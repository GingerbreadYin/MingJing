from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from optimizer.answer_eval import RubricResult, rubric_input, report_rows
from optimizer.judge_transport import DeepSeekJudge
from optimizer.store import Conflict, Store


class AnswerEvaluationTests(unittest.TestCase):
    def test_reference_labels_and_local_review_are_not_in_model_input(self):
        sample = {'question':'问题', 'response':'回答', 'reference':'参考答案',
                  'contexts':['条款正文'], 'expected':'pass', 'review':'本地标注原因'}
        sent = rubric_input(sample)
        self.assertEqual(set(sent), {'question','response','reference','contexts'})
        self.assertNotIn('本地标注原因', str(sent))

    def test_failures_are_not_scores_and_metric_divergence_is_explicit(self):
        rows = [
            {'id':'one','split':'calibration','expected':'fail','judge':{'verdict':'fail'},'faithfulness':1.0},
            {'id':'two','split':'check','expected':'pass','judge':{'verdict':'fail'},'faithfulness':0.25},
            {'id':'three','split':'check','expected':'pass','judge':None,'faithfulness':None},
        ]
        result = report_rows(rows)
        self.assertEqual(result['rubric_disagreements'], ['two'])
        self.assertEqual(result['faithful_but_rubric_failed'], ['one'])
        self.assertEqual(result['by_split']['check']['judge_scored'], 1)
        self.assertEqual(result['by_split']['check']['faithfulness_mean'], .25)


class JudgeBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_durable_limit_blocks_provider_even_with_a_new_judge(self):
        with TemporaryDirectory() as folder:
            store = Store(Path(folder) / 'budget.db')
            count = 0

            async def request(**kwargs):
                nonlocal count
                count += 1
                body = '{"grounded":true,"complete":true,"appropriate":true,"reason":"有据"}'
                return SimpleNamespace(
                    usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
                    choices=[SimpleNamespace(message=SimpleNamespace(content=body))],
                )

            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=request)))
            judge = DeepSeekJudge(client, store, 'sample-batch')
            for index in range(24):
                await judge(str(index), 'rubric', '固定样本', RubricResult)
            with self.assertRaises(Conflict):
                await DeepSeekJudge(client, store, 'sample-batch')('next', 'rubric', '固定样本', RubricResult)
            self.assertEqual(count, 24)
            self.assertEqual(len(store.budget()['calls']), 24)
