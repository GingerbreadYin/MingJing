"""Export the cached BGE encoder and compare local CPU retrieval."""
import argparse
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimizer.isolation import confined_path
from optimizer.onnx_benchmark import BATCH_SIZE, OPSET, QUERY_PREFIX, THREADS, compare_rankings, rank_hits
from optimizer.retrieval import DEFAULT_CONFIG, build_chunks


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as file:
        return file_digest(file, 'sha256').hexdigest()


def worker(mode, source, target, env):
    started = perf_counter()
    subprocess.run([sys.executable, '-m', 'optimizer.onnx_benchmark', mode, str(source), str(target)],
                   cwd=ROOT, env=env | {'RAGOPT_BENCH_START': str(started)}, timeout=240, check=True)
    return (perf_counter() - started) * 1000


def normalize(vectors):
    if not np.isfinite(vectors).all():
        raise ValueError('Embedding 包含非有限数')
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError('Embedding 包含零向量')
    return vectors / norms[:, None]


def drift(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError('ONNX 向量维度与基线不同')
    first, second = normalize(reference.astype('float64')), normalize(candidate.astype('float64'))
    return {
        'shape': list(reference.shape),
        'max_absolute': float(np.max(np.abs(reference - candidate))),
        'mean_absolute': float(np.mean(np.abs(reference - candidate))),
        'min_cosine': float(np.min(np.clip(np.sum(first * second, axis=1), -1, 1))),
        'onnx_max_unit_norm_error': float(np.max(np.abs(np.linalg.norm(candidate, axis=1) - 1))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependency-dir', type=Path, default=ROOT / '.onnx-deps')
    args = parser.parse_args()
    config = DEFAULT_CONFIG.copy()
    cached = ROOT / '.cache/models/models--BAAI--bge-small-zh-v1.5'
    revision = (cached / 'refs/main').read_text(encoding='utf-8').strip()
    snapshot = confined_path(cached, 'snapshots/' + revision)
    run = confined_path(ROOT, f'runtime/onnx-runs/{uuid4().hex}')
    run.mkdir(parents=True, exist_ok=False)
    model = run / 'bge-small-zh-v1.5.onnx'
    env = os.environ | {'PYTHONPATH': os.pathsep.join([str(args.dependency_dir.resolve()), str(ROOT)]),
                        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                        'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': str(THREADS),
                        'OPENBLAS_NUM_THREADS': str(THREADS), 'PYTHONIOENCODING': 'utf-8'}
    export_ms = worker('export', snapshot, model, env)
    export = read(model.with_suffix('.json'))
    frozen_sha = digest(model)

    data = ROOT / 'data'
    splits = {name: read(data / f'{name}.json') for name in ('development', 'heldout', 'regression')}
    corpora = {name: build_chunks(read(data / f'{name}.json'), config) for name in ('corpus', 'original-corpus')}
    questions = [row['question'] for rows in splits.values() for row in rows]
    chunks = [chunk['text'] for corpus in corpora.values() for chunk in corpus]
    request = run / 'inputs.json'
    request.write_text(json.dumps({'snapshot': str(snapshot), 'onnx': str(model),
                                   'questions': questions, 'chunks': chunks, **export}, ensure_ascii=False),
                       encoding='utf-8')
    backend = {}
    embeddings = {}
    for name in ('torch', 'onnx'):
        path = run / f'{name}.json'
        wall = worker(name, request, path, env)
        backend[name] = read(path) | {'full_worker_ms': wall}
        with np.load(path.with_suffix('.npz')) as archive:
            embeddings[name] = {key: archive[key] for key in ('chunks', 'queries', 'first', 'dynamic_probes')}
    if digest(model) != frozen_sha:
        raise ValueError('ONNX 文件在验收期间变化')

    metrics = {}
    query_start = 0
    for name, cases in splits.items():
        original = name == 'regression'
        corpus = corpora['original-corpus' if original else 'corpus']
        corpus_start = len(corpora['corpus']) if original else 0
        results = {}
        for key, arrays in embeddings.items():
            queries_matrix = normalize(arrays['queries'][query_start:query_start + len(cases)])
            corpus_matrix = normalize(arrays['chunks'][corpus_start:corpus_start + len(corpus)])
            results[key] = [rank_hits(corpus, row, top_k=config['top_k'], context_chars=config['context_chars'])
                            for row in queries_matrix @ corpus_matrix.T]
        metrics[name] = compare_rankings(cases, results['torch'], results['onnx'], config['top_k'])
        query_start += len(cases)

    report = {
        'model': 'BAAI/bge-small-zh-v1.5',
        'revision': revision,
        'model_weights_sha256': digest(snapshot / 'model.safetensors'),
        'tokenizer_sha256': {name: digest(snapshot / name)
                             for name in ('tokenizer.json', 'tokenizer_config.json', 'vocab.txt', 'special_tokens_map.json')},
        'onnx': {'artifact': model.relative_to(ROOT).as_posix(), 'sha256': frozen_sha,
                 'tokenizer_artifact': (run / 'tokenizer').relative_to(ROOT).as_posix(),
                 'effective_tokenizer_sha256': {item.name: digest(item) for item in sorted((run / 'tokenizer').iterdir())},
                 'bytes': model.stat().st_size, 'opset': OPSET, 'precision': 'FP32', 'export_ms': export_ms, **export},
        'pooling': 'CLS + L2', 'query_prefix': QUERY_PREFIX, 'config': config,
        'dataset_sha256': {name: digest(data / f'{name}.json') for name in (*splits, *corpora)},
        'cpu': {'platform': platform.platform(), 'processor': platform.processor(),
                'logical_cores': os.cpu_count(), 'intra_op_threads': THREADS, 'inter_op_threads': 1},
        'workload': {'questions': len(questions), 'chunks': len(chunks), 'batch_size': BATCH_SIZE, 'api_calls': 0,
                     'cold_scope': '每个后端独立 Python 进程到首条编码完成，系统文件缓存未清除'},
        'timing': backend, 'retrieval': metrics,
        'embedding_drift': {name: drift(embeddings['torch'][name], embeddings['onnx'][name])
                            for name in ('queries', 'chunks', 'first', 'dynamic_probes')},
    }
    report['chunk_sequence_sha256'] = sha256(json.dumps(chunks, ensure_ascii=False).encode()).hexdigest()
    path = run / 'onnx-benchmark.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    path.replace(confined_path(ROOT, 'runtime/onnx-benchmark.json'))
    print(json.dumps({'artifact': report['onnx'], 'timing': {
        key: {k: v['p50'] if isinstance(v, dict) else v for k, v in item['timing'].items()}
        for key, item in backend.items()
    }}, ensure_ascii=False))


if __name__ == '__main__':
    main()
