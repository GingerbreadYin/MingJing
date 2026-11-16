import unittest
from optimizer.evaluation import score_case, aggregate


class EvidenceScoringTests(unittest.TestCase):
    def test_retrieved_fragments_jointly_cover_one_fact(self):
        from optimizer.retrieval import build_chunks
        docs=[{'id':'p','title':'政策','facts':[{'id':'p:a','text':'甲乙丙丁戊己庚辛壬癸甲乙丙丁戊己庚辛壬癸。'}]}]
        chunks=build_chunks(docs,{'chunking':'fixed','chunk_size':16})
        score=score_case({'required':[['p:a']]},chunks,len(chunks))
        self.assertEqual(score['recall'],1)
        self.assertTrue(score['complete'])

    def test_partial_multifact_recall_and_duplicate_chunks(self):
        case = {'required': [['p:a'], ['p:b']]}
        results = [{'evidence_ids': ['p:a']}, {'evidence_ids': ['p:a']}]
        result = score_case(case, results, 2)
        self.assertEqual(result['recall'], .5)
        self.assertEqual(result['precision'], 1)
        self.assertFalse(result['complete'])

    def test_alternative_evidence_counts_once(self):
        case = {'required': [['p:a', 'p:a-alternative'], ['p:b']]}
        result = score_case(case, [{'evidence_ids': ['p:a-alternative', 'p:b']}], 2)
        self.assertEqual(result['recall'], 1)
        self.assertEqual(result['mrr'], 1)

    def test_unanswerable_is_not_recall_zero(self):
        result = score_case({'required': []}, [{'evidence_ids': ['unrelated:a']}], 2)
        self.assertIsNone(result['recall'])
        self.assertEqual(result['false_evidence_rate'], 1)

    def test_aggregate_excludes_unanswerable_from_recall(self):
        scores = [score_case({'required': [['p:a']]}, [{'evidence_ids':['p:a']}], 2), score_case({'required': []}, [], 2)]
        self.assertEqual(aggregate(scores)['recall'], 1)
