"""Kill a dedicated fixture process after development publication, then resume locally."""
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from optimizer import service as orchestration
from optimizer.store import Conflict, Store


def fixture_service(control):
    database = control.with_suffix('.db')
    orchestration.Store = lambda unused: Store(database)
    return orchestration.OptimizerService(ROOT)


def crash_fixture(control):
    service = fixture_service(control)
    task = service.create('v1.2 断点验收：基线等价候选，仅验证恢复机制', mode='offline', run=False)
    task_id = task['id']
    service._baseline(task_id)
    plan = service.propose(task_id, {}, '验收夹具使用同一检索配置，验证断点与回放。')
    service.approve(task_id, plan['plan_hash'], run=False)
    control.write_text(json.dumps({'task_id':task_id, 'plan_hash':plan['plan_hash']}), encoding='utf-8')
    record_checkpoint = service.store.record_checkpoint

    def crash_after_publish(task_id, stage, digest, **kwargs):
        if stage == 'development':
            os._exit(86)
        return record_checkpoint(task_id, stage, digest, **kwargs)

    service.store.record_checkpoint = crash_after_publish
    service.execute(task_id)
    raise RuntimeError('预期的故障注入未触发')


def main():
    os.environ.update(PYTHONIOENCODING='utf-8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      LANGSMITH_TRACING='false', LANGFUSE_TRACING='false')
    control = ROOT/'runtime'/f'v12-acceptance-{uuid4().hex}.json'
    control.parent.mkdir(exist_ok=True)
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--crash-fixture', str(control)],
                           cwd=ROOT, timeout=600)
    if child.returncode != 86:
        raise RuntimeError(f'故障夹具退出码为 {child.returncode}，预期 86')
    request = json.loads(control.read_text(encoding='utf-8'))
    service = fixture_service(control)
    try:
        task_id = request['task_id']
        before = service.replay(task_id)
        assert before['status'] == 'awaiting_resume', before['status']
        assert before['checkpoints']['development']['state'] == 'published', before['checkpoints']
        service.resume(**request, run=False)
        try:
            service.resume(**request, run=False)
        except Conflict:
            duplicate_resume_blocked = True
        else:
            raise AssertionError('重复续跑被接受')
        comparison = service.execute(task_id)
        after = service.replay(task_id)
        assert after['status'] == 'awaiting_retention', after['status']
        assert all(checkpoint['state'] == 'verified' for checkpoint in after['checkpoints'].values())
        assert before['checkpoints']['development']['digest'] == after['checkpoints']['development']['digest']
        starts = [row['stage'] for row in after['timeline'] if row['kind'] == 'stage_started']
        assert starts == ['build', 'development', 'verification'], starts
        assert after['development_metrics']['matches']
        assert not service.store.budget()['calls']
        assert all(comparison['retention_checks'].values()), comparison['retention_checks']
        service.retain(task_id, False)
        summary = {'task_id':task_id, 'crash_exit_code':child.returncode, 'before':before['checkpoints'],
                   'after':after['checkpoints'], 'stages_started':starts, 'duplicate_resume_blocked':duplicate_resume_blocked,
                   'metrics':after['development_metrics'], 'model_calls':0, 'fixture_final_status':'rejected',
                   'fixture_scope':'baseline equivalent; no retrieval improvement claim'}
        reports = ROOT/'reports'
        reports.mkdir(exist_ok=True)
        (reports/'v1.2-resume-acceptance.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False))
    finally:
        service.close()


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--crash-fixture':
        crash_fixture(Path(sys.argv[2]))
    else:
        main()
