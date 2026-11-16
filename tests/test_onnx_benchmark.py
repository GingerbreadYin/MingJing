import unittest

from optimizer.onnx_benchmark import rank_hits, compare_rankings


class ONNXBenchmarkTests(unittest.TestCase):
    def test_ranking_preserves_stable_ties_and_character_budget(self):
        chunks = [{'id':'a','text':'too long'}, {'id':'b','text':'12'}, {'id':'c','text':'34'}]
        hits = rank_hits(chunks, [.9, .8, .8], top_k=2, context_chars=3)
        self.assertEqual([hit['id'] for hit in hits], ['b'])

    def test_reordering_and_recall_regression_are_separate(self):
        cases = [{'id':'one','required':[['yes']]}, {'id':'two','required':[['yes']]}]
        yes = {'id':'A','evidence_ids':['yes']}
        no = {'id':'B','evidence_ids':[]}
        result = compare_rankings(cases, [[yes,no],[yes]], [[no,yes],[no]], 2)
        self.assertEqual(result['top_k_order_matches'], 0)
        self.assertEqual(result['top_k_set_matches'], 1)
        self.assertEqual(result['recall_regressions'], ['two'])
