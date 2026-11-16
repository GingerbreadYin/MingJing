import tempfile, unittest
from unittest.mock import patch
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient
from optimizer.routes import build_router
from optimizer.service import OptimizerService
from test_service import fake_evaluate


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        root=Path(self.tmp.name)
        (root/'baseline/rag').mkdir(parents=True)
        (root/'data').mkdir()
        for name in ['corpus','original-corpus','development','regression','heldout']:
            (root/f'data/{name}.json').write_text('[]',encoding='utf-8')
        self.service=OptimizerService(root,evaluator=fake_evaluate)
        app=FastAPI()
        app.include_router(build_router(self.service))
        self.client=TestClient(app)

    def tearDown(self):
        self.client.close()
        self.service.close()
        self.tmp.cleanup()

    def test_bad_goal_and_unknown_task_have_clear_errors(self):
        self.assertEqual(self.client.post('/api/rag-optimizer/tasks',json={'goal':''}).status_code,422)
        self.assertEqual(self.client.get('/api/rag-optimizer/tasks/missing').status_code,404)

    def test_query_split_report_is_read_only_and_absent_until_generated(self):
        endpoint = '/api/rag-optimizer/query-split'
        self.assertEqual(self.client.get(endpoint).status_code, 404)
        report = self.service.root / 'runtime/query-split.json'
        report.write_text('{"cases": []}', encoding='utf-8')
        self.assertEqual(self.client.get(endpoint).json(), {'cases': []})
        self.assertEqual(self.client.post(endpoint).status_code, 405)

    def test_memory_note_is_durable_and_private_by_default(self):
        task = self.service.create('备注测试', run=False, mode='offline', iterative=True)
        endpoint = f"/api/rag-optimizer/tasks/{task['id']}/memory"
        self.assertEqual(self.client.post(endpoint, json={'text': '  '}).status_code, 422)
        response = self.client.post(endpoint, json={'text': '本地备忘'})
        self.assertEqual(response.status_code, 201)
        entry = self.client.get(endpoint).json()['context']['entries'][0]
        self.assertEqual(entry['text'], '本地备忘')
        self.assertEqual(entry['source']['task_id'], task['id'])
        self.assertEqual(self.service.memory.view(task['id'], for_model=True)['context']['entries'], [])
        self.assertEqual(self.service.store.get(task['id'])['status'], 'created')

    def test_answer_report_prefers_real_results_and_cannot_trigger_calls(self):
        endpoint = '/api/rag-optimizer/answer-evaluation'
        self.assertEqual(self.client.get(endpoint).status_code, 404)
        runtime = self.service.root / 'runtime'
        (runtime / 'answer-evaluation-offline.json').write_text('{"mode":"offline_replay"}')
        self.assertEqual(self.client.get(endpoint).json()['mode'], 'offline_replay')
        (runtime / 'answer-evaluation.json').write_text('{"mode":"deepseek_live"}')
        self.assertEqual(self.client.get(endpoint).json()['mode'], 'deepseek_live')
        self.assertEqual(self.client.post(endpoint).status_code, 405)

    def test_onnx_report_route_only_reads_existing_file(self):
        endpoint = '/api/rag-optimizer/onnx-benchmark'
        self.assertEqual(self.client.get(endpoint).status_code, 404)
        (self.service.root / 'runtime/onnx-benchmark.json').write_text('{"precision":"FP32"}')
        self.assertEqual(self.client.get(endpoint).json(), {'precision':'FP32'})
        self.assertEqual(self.client.post(endpoint).status_code, 405)

    def test_cross_origin_mutations_are_rejected(self):
        response=self.client.post('/api/rag-optimizer/tasks',headers={'Origin':'https://untrusted.example'},json={'goal':'test'})
        self.assertEqual(response.status_code,403)

    def test_approval_and_retention_are_separate(self):
        task=self.service.create('检查检索',run=False,mode='offline')
        task=self.service.propose(task['id'],{},'仅供审批')
        response=self.client.post(f"/api/rag-optimizer/tasks/{task['id']}/retain",json={'keep':True})
        self.assertEqual(response.status_code,409)
        response=self.client.post(f"/api/rag-optimizer/tasks/{task['id']}/approve",json={'plan_hash':'stale'})
        self.assertEqual(response.status_code,409)

    def test_resume_route_validates_approval_and_replay_is_read_only(self):
        task = self.service.create('续跑验收', run=False, mode='offline')
        task_id = task['id']
        base = f'/api/rag-optimizer/tasks/{task_id}'
        plan = self.service.diagnose_offline(task_id)
        self.service.approve(task_id, plan['plan_hash'], run=False)
        self.service.store.recover_interrupted()
        count = len(self.service.store.events(task_id))
        replay = self.client.get(base+'/replay').json()
        self.assertEqual(replay['status'], 'awaiting_resume')
        self.assertEqual(replay['model_calls'], 0)
        self.assertEqual(len(self.service.store.events(task_id)), count)
        self.assertEqual(self.client.post(base+'/resume', json={'plan_hash':'stale'}).status_code, 409)
        with patch.object(self.service.pool, 'submit') as queue:
            self.assertEqual(self.client.post(base+'/resume', json={'plan_hash':plan['plan_hash']}).status_code, 200)
            self.assertEqual(self.client.post(base+'/resume', json={'plan_hash':plan['plan_hash']}).status_code, 409)
            queue.assert_called_once()

    def test_sse_reconnection_uses_last_event_id_and_keeps_task_scope(self):
        task = self.service.create('SSE 验收', run=False)
        task_id = task['id']
        stream = f'/api/rag-optimizer/tasks/{task_id}/stream?once=true'
        initial = self.client.get(stream)
        self.assertIn('text/event-stream', initial.headers['content-type'])
        cursor = max(int(line[4:]) for line in initial.text.splitlines() if line.startswith('id: '))
        other = self.service.create('另一任务', run=False)
        self.service.store.event(other['id'], 'unrelated', {})
        self.service.store.event(task_id, 'resumed-update', {})
        replayed = self.client.get(stream, headers={'Last-Event-ID':str(cursor)})
        self.assertIn('resumed-update', replayed.text)
        self.assertNotIn('unrelated', replayed.text)
        self.assertEqual(len([line for line in replayed.text.splitlines() if line.startswith('id: ')]), 1)
        self.assertEqual(self.client.get(stream, headers={'Last-Event-ID':'-1'}).status_code, 422)
