"""Development-only feedback for a bounded experiment series."""
from .experiment_plan import assess_experiment


def development_feedback(task, report):
    baseline = task['baseline']
    assessment = assess_experiment(task['experiment_plan'], report)
    checks = {
        'development_recall_no_regression': report['metrics']['recall'] >= baseline['metrics']['recall'],
        'precision_no_regression': report['metrics']['precision'] >= baseline['metrics']['precision'],
        'approved_target_recall_gain': assessment['passed'],
    }
    before = {row['id']: row['scores']['recall'] for row in baseline['cases']}
    targets = set(task['experiment_plan']['target_case_ids'])
    deltas = []
    for row in report['cases']:
        old, new = before.get(row['id']), row['scores']['recall']
        if old is not None and new is not None and (row['id'] in targets or new != old):
            deltas.append({'id':row['id'], 'question':row['question'], 'baseline_recall':old,
                           'candidate_recall':new, 'target':row['id'] in targets})
    deltas.sort(key=lambda row: (not row['target'], row['candidate_recall'] - row['baseline_recall'], row['id']))
    return {'round':task['round_number'], 'config':task['plan'], 'baseline_metrics':baseline['metrics'],
            'metrics':report['metrics'], 'target_assessment':assessment, 'checks':checks,
            'eligible':all(checks.values()), 'case_deltas':deltas[:8]}
