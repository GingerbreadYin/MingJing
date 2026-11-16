import unittest
from optimizer.evaluation import case_regressions


class RegressionComparisonTests(unittest.TestCase):
    def test_offsetting_gains_do_not_hide_broken_original_case(self):
        baseline={'cases':[{'id':'A','question':'A','scores':{'recall':1}}, {'id':'B','question':'B','scores':{'recall':0}}]}
        candidate={'cases':[{'id':'A','scores':{'recall':0}}, {'id':'B','scores':{'recall':1}}]}
        differences=case_regressions(baseline,candidate)
        self.assertEqual([r['id'] for r in differences],['A'])
