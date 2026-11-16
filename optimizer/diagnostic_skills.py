"""Select and load an operator-reviewed guide for the diagnosis role."""
from pathlib import Path


GUIDES = {'missing-evidence', 'ranking-regression'}


def select_skill(task):
    previous = task.get('previous_round') or {}
    checks = previous.get('checks') or {}
    if (previous.get('config', {}).get('strategy') != 'vector'
            and (checks.get('development_recall_no_regression') is False
                 or checks.get('precision_no_regression') is False)):
        return 'ranking-regression'
    for row in (task.get('baseline') or {}).get('cases', []):
        recall = row['scores']['recall']
        if recall is not None and recall < 1:
            return 'missing-evidence'
    return None


def load_guide(name):
    if name not in GUIDES:
        raise ValueError('未知诊断指南')
    path = Path(__file__).parent/'skill_guides'/name/'SKILL.md'
    return path.read_text(encoding='utf-8')
