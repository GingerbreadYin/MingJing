"""Export BGE and run isolated CPU inference measurements."""
import json
import os
from pathlib import Path
import sys
from time import perf_counter

from .evaluation import aggregate, score_case


QUERY_PREFIX = '为这个句子生成表示以用于检索相关文章：'
BATCH_SIZE = 32
THREADS = 4
OPSET = 17
INPUTS = ('input_ids', 'attention_mask', 'token_type_ids')


def rank_hits(chunks, scores, *, top_k, context_chars):
    order = sorted(range(len(chunks)), key=lambda index: scores[index], reverse=True)
    hits = []
    for index in order:
        chunk = chunks[index]
        length = len(chunk['text'])
        if length <= context_chars:
            hits.append(chunk)
            context_chars -= length
            if len(hits) == top_k:
                break
    return hits


def compare_rankings(cases, torch_hits, onnx_hits, top_k):
    rows = []
    for case, first, second in zip(cases, torch_hits, onnx_hits, strict=True):
        rows.append({
            'id': case['id'],
            'torch_ids': [chunk['id'] for chunk in first],
            'onnx_ids': [chunk['id'] for chunk in second],
            'torch': score_case(case, first, top_k),
            'onnx': score_case(case, second, top_k),
        })
    return {
        'cases': len(rows),
        'top_k_order_matches': sum(row['torch_ids'] == row['onnx_ids'] for row in rows),
        'top_k_set_matches': sum(set(row['torch_ids']) == set(row['onnx_ids']) for row in rows),
        'recall_regressions': [row['id'] for row in rows if row['torch']['recall'] is not None
                               and row['onnx']['recall'] < row['torch']['recall']],
        'metrics': {backend: aggregate([row[backend] for row in rows]) for backend in ('torch', 'onnx')},
        'rows': rows,
    }


def load_torch(snapshot, *, eager=False):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    kwargs = {'attn_implementation': 'eager'} if eager else {}
    return SentenceTransformer(str(snapshot), device='cpu', local_files_only=True, model_kwargs=kwargs).eval()


def export_model(snapshot, output):
    import onnx
    import torch

    model = load_torch(snapshot, eager=True)
    pooling = model[1]
    if pooling.pooling_mode != 'cls' or type(model[2]).__name__ != 'Normalize':
        raise ValueError('BGE 模块配置与导出约定不一致')

    class Embedding(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, input_ids, attention_mask, token_type_ids):
            # This BERT has no cache. A 4-D additive mask avoids tracing the
            # Transformers mask builder's Python branch on sequence length.
            mask = (1 - attention_mask[:, None, None, :].to(torch.float32)) * torch.finfo(torch.float32).min
            hidden = self.backbone(input_ids=input_ids, attention_mask=mask,
                                   token_type_ids=token_type_ids, return_dict=False)[0]
            return torch.nn.functional.normalize(hidden[:, 0], p=2, dim=1)

    encoder = Embedding(model[0].auto_model).eval()
    model.tokenizer.save_pretrained(output.parent / 'tokenizer')
    example = model.tokenizer(['测试问题', '用于验证动态形状的更长文本'], padding=True, return_tensors='pt')
    dynamic = {name: {0: 'batch', 1: 'sequence'} for name in INPUTS}
    dynamic['embedding'] = {0: 'batch'}
    with torch.inference_mode():
        torch.onnx.export(encoder, tuple(example[name] for name in INPUTS), str(output),
                          input_names=list(INPUTS), output_names=['embedding'], dynamic_axes=dynamic,
                          opset_version=OPSET, dynamo=False)
    onnx.checker.check_model(str(output))
    return {'max_seq_length': model.max_seq_length, 'embedding_dim': model.get_embedding_dimension()}


def prepare_encoder(backend, snapshot, onnx_path, max_seq_length):
    if backend == 'torch':
        import torch
        model = load_torch(snapshot)

        def encode(texts):
            return model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True)

        return encode, torch.__version__

    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(onnx_path.parent / 'tokenizer/tokenizer.json'))
    tokenizer.enable_truncation(max_length=max_seq_length)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id('[PAD]'), pad_token='[PAD]')
    options = ort.SessionOptions()
    options.intra_op_num_threads = THREADS
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=['CPUExecutionProvider'])

    def encode(texts):
        order = np.argsort([-len(text) for text in texts])
        batches = []
        for start in range(0, len(texts), BATCH_SIZE):
            encoded = tokenizer.encode_batch([texts[i] for i in order[start:start + BATCH_SIZE]])
            tensors = {
                'input_ids': np.array([row.ids for row in encoded], dtype=np.int64),
                'attention_mask': np.array([row.attention_mask for row in encoded], dtype=np.int64),
                'token_type_ids': np.array([row.type_ids for row in encoded], dtype=np.int64),
            }
            batches.append(session.run(['embedding'], tensors)[0])
        return np.concatenate(batches)[np.argsort(order)]

    return encode, ort.__version__


def benchmark(backend, request_path, output):
    import numpy as np

    request = json.loads(request_path.read_text(encoding='utf-8'))
    started = perf_counter()
    encode, version = prepare_encoder(backend, Path(request['snapshot']), Path(request['onnx']),
                                      request['max_seq_length'])
    loaded = perf_counter()
    queries = [QUERY_PREFIX + query for query in request['questions']]
    first = encode(queries[:1])
    first_done = perf_counter()
    timings = {'initialize_ms': (loaded - started) * 1000, 'first_query_ms': (first_done - loaded) * 1000}
    timings['process_to_first_ms'] = (first_done - float(os.environ['RAGOPT_BENCH_START'])) * 1000
    encode(queries[:8])
    warm = {'single_ms': ([queries[i % len(queries)]] for i in range(25)),
            'query_batch_ms': (queries for _ in range(5)),
            'corpus_batch_ms': (request['chunks'] for _ in range(5))}
    for label, runs in warm.items():
        samples = []
        for texts in runs:
            start = perf_counter()
            encode(texts)
            samples.append((perf_counter() - start) * 1000)
        timings[label] = {'p50': float(np.percentile(samples, 50)), 'p95': float(np.percentile(samples, 95)),
                          'samples': samples}
    probes = ['订单 ABC Café 如何退款？', '物流核查' * 600, '发票']
    np.savez(output.with_suffix('.npz'), queries=encode(queries), chunks=encode(request['chunks']),
             first=first, dynamic_probes=encode(probes))
    output.write_text(json.dumps({'runtime_version':version, 'timing':timings}), encoding='utf-8')


if __name__ == '__main__':
    mode, source, target = sys.argv[1:]
    if mode == 'export':
        output = Path(target)
        info = export_model(Path(source), output)
        output.with_suffix('.json').write_text(json.dumps(info), encoding='utf-8')
    else:
        benchmark(mode, Path(source), Path(target))
