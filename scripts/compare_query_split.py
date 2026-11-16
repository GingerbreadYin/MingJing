"""Reproduce the local single-query versus query-splitting experiment."""
import json
from pathlib import Path
import sys
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimizer.checkpoints import read_json, write_json
from optimizer.evaluation import aggregate, score_case
from optimizer.isolation import confined_path
from optimizer.retrieval import DEFAULT_CONFIG, run_retrieval


def summarize(cases, output, config, elapsed):
    comparison = output['query_split']
    rows = []
    for case, baseline, split in zip(cases, output['results'], comparison['cases'], strict=True):
        def evidence(hits):
            return {
                'scores': score_case(case, hits, config['top_k']),
                'context_chars': sum(len(hit['text']) for hit in hits),
                'hits': [{key: hit[key] for key in ('id', 'evidence_ids', 'text')} for hit in hits],
            }

        branches = [{'query': branch['query'], 'ranked_chunk_ids': [hit['id'] for hit in branch['hits']]}
                    for branch in split['branches']]
        rows.append({key: case[key] for key in ('id', 'type', 'question', 'required')} | {
            'branches': branches, 'baseline': evidence(baseline), 'split': evidence(split['hits']),
        })
    cohorts = {
        'all_development': rows,
        'multi_condition': [row for row in rows if row['type'] == 'multiple'],
        'actually_split': [row for row in rows if len(row['branches']) > 1],
    }
    return {
        'dataset': 'development',
        'config': config,
        'worker': {key: output['metadata'][key] for key in ('model', 'corpus_sha256', 'chunks', 'device', 'api_calls')},
        'langgraph': comparison['langgraph'],
        'timing': {key: comparison[key] for key in ('baseline_seconds', 'split_seconds')} | {
            'worker_wall_seconds': elapsed,
            'scope': '每组批量查询编码、余弦相似度及排序。拆分组另含拆分、图构建和调度。两组共用已加载的模型及语料索引。',
        },
        'queries': {key: comparison[key] for key in ('baseline_queries', 'split_queries', 'split_embeddings')},
        'metrics': {
            name: {arm: aggregate([row[arm]['scores'] for row in cohort]) for arm in ('baseline', 'split')}
            for name, cohort in cohorts.items()
        },
        'cases': rows,
    }


def main():
    corpus = read_json(ROOT / 'data/corpus.json')
    cases = read_json(ROOT / 'data/development.json')
    run = confined_path(ROOT, f'runtime/query-split-runs/{uuid4().hex}')
    config = DEFAULT_CONFIG.copy()
    start = perf_counter()
    output = run_retrieval(ROOT, corpus, [case['question'] for case in cases], config, run,
                           compare_query_split=True)
    report = summarize(cases, output, config, perf_counter() - start)
    staged_report = run / 'query-split.json'
    write_json(staged_report, report)
    staged_report.replace(confined_path(ROOT, 'runtime/query-split.json'))
    print(json.dumps({key: report[key] for key in ('metrics', 'queries', 'timing')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
