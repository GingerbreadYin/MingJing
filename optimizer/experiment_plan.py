"""Validated inputs and evidence criteria for one bounded experiment."""
from statistics import mean

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, str_strip_whitespace=True)

    hypothesis: str = Field(min_length=5, max_length=1000, description='待检验的检索失败原因')
    target_case_ids: list[str] = Field(min_length=1, max_length=8, description='开发集漏召回题 ID')
    min_recall_gain: float = Field(gt=0, le=1, description='目标题平均证据 Recall 的最小绝对增量，例如 0.10')

    @field_validator('target_case_ids')
    @classmethod
    def unique_targets(cls, ids):
        if len(set(ids)) != len(ids) or any(not item or len(item) > 120 for item in ids):
            raise ValueError('目标题 ID 必须有效且不重复')
        return ids


def build_experiment(spec, baseline, config, reference_config):
    rows = {row['id']: row['scores']['recall'] for row in baseline.get('cases', [])}
    if any(rows.get(case_id) is None or rows[case_id] >= 1 for case_id in spec.target_case_ids):
        raise ValueError('目标题必须属于本任务开发集中的漏召回题')
    changes = {name: {'before': reference_config[name], 'after': value}
               for name, value in config.items() if value != reference_config[name]}
    if len(changes) != 1:
        raise ValueError('每份实验计划必须且只能修改一个检索配置项')
    baseline_recall = mean(rows[case_id] for case_id in spec.target_case_ids)
    if baseline_recall + spec.min_recall_gain > 1 + 1e-9:
        raise ValueError('目标题召回阈值超过可达到的上限')
    return {
        'schema_version': 1,
        **spec.model_dump(),
        'changes': changes,
        'target_baseline_recall': baseline_recall,
        'budget': {name: config[name] for name in ('top_k', 'context_chars')},
    }


def assess_experiment(experiment, report):
    rows = {row['id']: row['scores']['recall'] for row in report['cases']}
    targets = experiment['target_case_ids']
    if any(rows.get(case_id) is None for case_id in targets):
        raise ValueError('候选评测缺少计划指定的开发集目标题')
    recall = mean(rows[case_id] for case_id in targets)
    baseline = experiment['target_baseline_recall']
    gain = recall - baseline
    minimum = experiment['min_recall_gain']
    return {'baseline_recall': baseline, 'candidate_recall': recall, 'recall_gain': gain,
            'min_recall_gain': minimum, 'passed': gain + 1e-9 >= minimum}
