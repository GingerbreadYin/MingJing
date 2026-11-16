import unittest

from optimizer import query_split
from optimizer.retrieval import validate_config


class QuerySplitTests(unittest.TestCase):
    def test_candidate_flag_defaults_off_and_requires_plain_vector(self):
        self.assertFalse(validate_config({})['query_split'])
        self.assertTrue(validate_config({'query_split': True})['query_split'])
        for config in ({'query_split': 1}, {'query_split': True, 'strategy':'hybrid'},
                       {'query_split': True, 'rerank': True}):
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_splits_two_explicit_questions_and_keeps_ambiguous_text(self):
        self.assertEqual(query_split.split_query('物流停滞何时核查，核查后能否立即按丢件退款？'),
                         ['物流停滞何时核查', '物流停滞 核查后能否立即按丢件退款'])
        self.assertEqual(query_split.split_query('赠品退货何时登记，登记后能否补交材料？'),
                         ['赠品退货何时登记', '赠品退货 登记后能否补交材料'])
        self.assertEqual(query_split.split_query('合并发货何时核实，出库后与售后记录有什么限制？'),
                         ['合并发货何时核实', '出库后与售后记录有什么限制'])
        for question in ('若未收到包裹，怎么办？', '发票抬头和税号怎么填？'):
            self.assertEqual(query_split.split_query(question), [question])

    def test_send_merges_unique_hits_and_honors_both_limits(self):
        hits = {
            'one': [{'id': 'A', 'text': '123'}, {'id': 'B', 'text': '2345'}],
            'two': [{'id': 'A', 'text': '123'}, {'id': 'C', 'text': '23'}, {'id': 'D', 'text': '4'}],
        }
        graph = query_split.build_query_graph(hits.__getitem__, top_k=2, context_chars=5)
        both = graph.invoke({'queries': ['one', 'two']})
        self.assertEqual([hit['id'] for hit in both['hits']], ['A', 'C'])
        self.assertEqual([(branch['query'], len(branch['hits'])) for branch in both['branches']],
                         [('one', 2), ('two', 3)])
        single = graph.invoke({'queries': ['one']})
        self.assertEqual([hit['id'] for hit in single['hits']], ['A'])
        self.assertEqual([branch['query'] for branch in single['branches']], ['one'])
