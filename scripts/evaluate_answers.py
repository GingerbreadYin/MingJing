"""Run the pinned answer-quality evaluation, with offline replay by default."""
import argparse
import asyncio
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimizer.answer_eval import evaluate_samples, load_samples, ragas_metric
from optimizer.isolation import confined_path
from optimizer.judge_transport import DeepSeekJudge, MODEL
from optimizer.store import Store


def replay(root):
    payloads = json.loads((root / 'data/answer-eval-replay.json').read_text(encoding='utf-8'))

    async def complete(case_id, stage, prompt, response_model):
        saved = payloads[case_id]
        if stage == 'rubric':
            raw = dict(zip(('grounded', 'complete', 'appropriate', 'reason'), saved['rubric'], strict=True))
        elif stage == 'statements':
            raw = {'statements': [item[0] for item in saved['claims']]}
        else:
            raw = {'statements': [{'statement': text, 'verdict': verdict, 'reason': '离线脚本响应'}
                                  for text, verdict in saved['claims']]}
        return response_model.model_validate(raw, strict=True)

    return complete


async def run(args):
    samples = load_samples(ROOT)
    recorded = replay(ROOT)
    # Check that the actual Ragas package and adapter load before reserving any provider request.
    ragas_metric(recorded, samples[0]['id'])
    if args.live_ledger is None:
        report = await evaluate_samples(samples, recorded, mode='offline_replay')
        report['provider_calls'] = []
    else:
        from dotenv import dotenv_values
        from httpx import AsyncClient
        from openai import AsyncOpenAI

        key = os.environ.get('DEEPSEEK_API_KEY')
        if not key and args.key_file:
            key = dotenv_values(args.key_file, interpolate=False).get('DEEPSEEK_API_KEY')
        if not key or key == 'not-configured':
            raise ValueError('未配置 DeepSeek 凭据')
        if not args.live_ledger.is_file():
            raise ValueError('需要现有项目的预算账本')
        store = Store(args.live_ledger.resolve())
        ledger_id = 'answer-eval-v17-acceptance'
        async with AsyncOpenAI(api_key=key, base_url='https://api.deepseek.com',
                               max_retries=0, timeout=40,
                               http_client=AsyncClient(follow_redirects=False)) as client:
            judge = DeepSeekJudge(client, store, ledger_id)
            report = await evaluate_samples(samples, judge, mode='deepseek_live')
        report.update(provider=MODEL, provider_calls=judge.calls, ledger_id=ledger_id,
                      run_estimate_cny=sum(call.get('estimate_cny', .1) for call in judge.calls))
    report['sample_sha256'] = sha256(json.dumps(samples, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    report['created_at'] = datetime.now(timezone.utc).isoformat()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live-ledger', type=Path, help='经授权的真实运行使用的现有 SQLite 预算账本')
    parser.add_argument('--key-file', type=Path, help='可选的现有本地凭据文件路径')
    args = parser.parse_args()
    report = asyncio.run(run(args))
    name = 'answer-evaluation.json' if args.live_ledger else 'answer-evaluation-offline.json'
    target = confined_path(ROOT, 'runtime/' + name)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = confined_path(ROOT, f'runtime/.answer-evaluation-{uuid4().hex}.tmp')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(target)
    print(json.dumps({'mode': report['mode'], 'provider_calls': len(report['provider_calls']),
                      'summary': report['summary'], 'output': str(target)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
