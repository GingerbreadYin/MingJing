"""Create an archived two-round example with traceable synthetic notes."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimizer.store import Store


def main():
    store = Store(ROOT / 'runtime/optimizer.db')
    goal = 'v1.6 验收示例 · 两轮记忆和来源追溯（模拟轮次）'
    existing = next((task for task in store.list() if task['goal'] == goal and task.get('previous_task_id')), None)
    if existing:
        print(json.dumps({'first': existing['previous_task_id'], 'second': existing['id']}))
        return
    parent = store.create(goal, 'synthetic-memory-example', iterative=True)
    first = parent['id']
    store.event(first, 'series_memory_note', {'text': '先检查物流停滞题的两组证据，保持 Top-2。', 'include_in_model': True})
    store.event(first, 'series_memory_note', {'text': '这条是本机备忘，用于演示模型可见性。', 'include_in_model': False})
    feedback = {
        'round': 1, 'metrics': {'recall': .5}, 'checks': {'development_recall_no_regression': True},
        'target_assessment': {'baseline_recall': .5, 'candidate_recall': .5, 'passed': False},
        'eligible': False,
        'case_deltas': [{'id': 'demo-13-combined', 'baseline_recall': .5, 'candidate_recall': .5}],
    }
    store.transition(first, {'created'}, 'awaiting_round_decision', plan={}, mode='offline', round_summary=feedback)
    second = store.create_replan(first, None, feedback)['id']
    store.event(second, 'series_memory_note', {'text': '对照轮次只变动一个检索配置项。', 'include_in_model': True})
    store.transition(second, {'created'}, 'rejected', test_fixture=False)
    print(json.dumps({'first': first, 'second': second}))


if __name__ == '__main__':
    main()
