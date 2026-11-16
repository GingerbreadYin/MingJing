"""Publish local stage outputs once and verify them before every reuse."""
import json
from pathlib import Path
import tempfile

from .artifacts import candidate_hash
from .isolation import confined_path
from .store import Conflict


STAGES = ('build', 'development', 'verification')


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def stage_directory(work, stage):
    if stage not in STAGES:
        raise ValueError('未知执行阶段')
    return confined_path(work, f'stages/{stage}')


def stage_identity(task, stage):
    return {'protocol':1, 'task_id':task['id'], 'plan_hash':task['approved_hash'],
            'snapshot':task['snapshot'], 'stage':stage}


def verify_stage(path, task, stage):
    try:
        manifest = read_json(confined_path(path, 'checkpoint.json'))
        payload = confined_path(path, 'payload')
        if not payload.is_dir():
            raise ValueError('检查点产物目录不存在')
        digest = candidate_hash(payload)
        expected = task.get('checkpoints', {}).get(stage, digest)
        if manifest != stage_identity(task, stage) | {'digest':digest} or expected != digest:
            raise ValueError('检查点指纹或审批身份不一致')
    except (OSError, ValueError) as exc:
        raise Conflict(f'阶段 {stage} 检查点校验失败：{exc}') from None
    return payload, digest


def run_stage(service, task_id, work, stage, produce):
    task = service._guard(task_id)
    path = stage_directory(work, stage)
    adopted = path.exists()
    if not adopted:
        if stage in task.get('checkpoints', {}):
            raise Conflict(f'阶段 {stage} 已登记但产物缺失')
        path.parent.mkdir(exist_ok=True)
        service.store.event(task_id, 'stage_started', {'stage':stage})
        # Attempts abandoned by a crash remain separate from published output.
        temporary = Path(tempfile.mkdtemp(prefix=f'.{stage}-', dir=path.parent))
        payload = confined_path(temporary, 'payload')
        payload.mkdir()
        produce(payload)
        service._guard(task_id)
        digest = candidate_hash(payload)
        write_json(confined_path(temporary, 'checkpoint.json'), stage_identity(task, stage) | {'digest':digest})
        temporary.rename(path)
        payload = confined_path(path, 'payload')
    else:
        payload, digest = verify_stage(path, task, stage)
    service.store.record_checkpoint(task_id, stage, digest, plan_hash=task['approved_hash'], adopted=adopted)
    return payload
