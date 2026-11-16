import unittest, tempfile, subprocess, os
from pathlib import Path
from optimizer.retrieval import build_chunks, validate_config, run_retrieval


class ChunkMappingTests(unittest.TestCase):
    def test_weighted_rrf_respects_weights_and_preserves_equal_weight(self):
        from optimizer.rag_adapter import fuse_rrf
        dense, sparse = [.9, .1], [1., 9.]
        equal, _ = fuse_rrf(dense, sparse, 50)
        self.assertEqual(equal, [1/61 + 1/62] * 2)
        dense_first, _ = fuse_rrf(dense, sparse, 75)
        sparse_first, _ = fuse_rrf(dense, sparse, 25)
        self.assertGreater(dense_first[0], dense_first[1])
        self.assertLess(sparse_first[0], sparse_first[1])

    def test_custom_weight_requires_hybrid_and_integer_percent(self):
        self.assertEqual(validate_config({})['rrf_vector_pct'], 50)
        for config in ({'rrf_vector_pct':75}, {'strategy':'hybrid','rrf_vector_pct':True},
                       {'strategy':'hybrid','rrf_vector_pct':100}):
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_overlap_preserves_whole_fact_without_redundant_tail(self):
        docs = [{'id':'p','title':'标题', 'facts':[{'id':'p:a','text':'甲'*10},
                                                  {'id':'p:b','text':'乙'*14}]}]
        cfg = {'chunking':'fixed','chunk_size':20}
        plain = build_chunks(docs, cfg)
        overlapping = build_chunks(docs, cfg | {'chunk_overlap':4})
        self.assertEqual([c['start'] for c in plain], [0,20])
        self.assertEqual([c['start'] for c in overlapping], [0,16])
        self.assertIn('p:b', overlapping[1]['evidence_ids'])
        self.assertNotIn('p:b', plain[1]['evidence_ids'])

    def test_overlap_rejects_non_fixed_modes_and_nonpositive_stride(self):
        for cfg in ({'chunk_overlap':4}, {'chunking':'fixed','chunk_size':20,'chunk_overlap':20},
                    {'chunking':'fixed','chunk_overlap':True}):
            with self.assertRaises(ValueError):
                validate_config(cfg)

    def test_existing_report_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'report').mkdir()
            with self.assertRaises(FileExistsError):
                run_retrieval(root,[],[],{},root/'report')
            self.assertFalse((root/'report/retrieval-request.json').exists())

    @unittest.skipUnless(os.name=='nt','Windows junction regression')
    def test_report_writer_rejects_windows_junction(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            root=Path(tmp)
            result=subprocess.run(['cmd','/c','mklink','/J',str(root/'report'),outside],capture_output=True)
            if result.returncode:
                self.skipTest('当前系统不能创建测试目录联接')
            with self.assertRaises(ValueError):
                run_retrieval(root,[],[],{},root/'report')
            self.assertFalse((Path(outside)/'retrieval-request.json').exists())

    def test_chunk_labels_require_complete_evidence_span(self):
        docs=[{'id':'p','title':'测试政策','facts':[{'id':'p:a','text':'定制商品不支持无理由退货，但质量问题可以申请售后。'}]}]
        fragments=build_chunks(docs, {'chunking':'fixed','chunk_size':16})
        self.assertFalse(any('p:a' in c['evidence_ids'] for c in fragments))
        sections=build_chunks(docs, {'chunking':'section','chunk_size':300})
        self.assertEqual(sections[0]['evidence_ids'], ['p:a'])

    def test_config_cannot_inject_commands_or_unbounded_topk(self):
        for config in [{'strategy':'shell'}, {'top_k':9999}, {'command':'echo hello'}]:
            with self.assertRaises(ValueError):
                validate_config(config)
