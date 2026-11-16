import json, tempfile, unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import shutil
from optimizer.service import OptimizerService
from optimizer.store import Conflict
from optimizer.artifacts import candidate_hash


def fake_evaluate(root, documents, cases, config, work, split, **kwargs):
    work.mkdir(parents=True,exist_ok=True)
    return {'split':split,'metrics':{'recall':.5,'precision':.5},
            'cases':[{'id':'dev-miss','question':'开发集漏召回题','scores':{'recall':.5}}] if split=='development' else [],
            'config':config}


class ServiceBoundaryTests(unittest.TestCase):
    def test_rrf_profile_binds_the_equal_weight_control(self):
        task = self.service.create('检验 RRF',run=False,mode='offline',iterative=True,rrf_experiment=True)
        self.service._baseline(task['id'])
        self.assertEqual(self.service.store.get(task['id'])['baseline']['config'], {'strategy':'hybrid'})
        experiment = {'hypothesis':'提高向量权重验证融合漏召回', 'target_case_ids':['dev-miss'], 'min_recall_gain':.1}
        candidate = {'strategy':'hybrid','rrf_vector_pct':75}
        with self.assertRaises(ValueError):
            self.service.propose(task['id'],candidate | {'top_k':3},'改两项',experiment)
        result = self.service.propose(task['id'],candidate,'单项权重对照',experiment)
        self.assertEqual(result['experiment_plan']['changes'], {'rrf_vector_pct':{'before':50,'after':75}})
        self.assertEqual(result['status'],'awaiting_approval')

    def test_overlap_uses_fixed_control_and_only_allows_overlap_change(self):
        def evaluator(*args, **kwargs):
            report = fake_evaluate(*args, **kwargs)
            report['cases'][0].update(required=[['fact']], results=[])
            return report

        self.service.evaluator = evaluator
        task = self.service.create('检验重叠',run=False,mode='offline',iterative=True,overlap_experiment=True)
        task = self.service.diagnose_offline(task['id'])
        self.assertEqual(task['experiment_plan']['changes'], {'chunk_overlap':{'before':0,'after':16}})
        self.assertEqual(task['baseline']['config'], {'chunking':'fixed','chunk_size':64})
        with self.assertRaises(ValueError):
            self.service.propose(task['id'],task['plan'] | {'top_k':3},'改两项',task['experiment_plan'])

    def prepared_candidate(self):
        task=self.service.create('验收保留',run=False)
        candidate=self.service.work(task['id'])/'candidate'
        candidate.mkdir()
        (candidate/'adapter.py').write_text('frozen code',encoding='utf-8')
        (candidate/'vector_index.pt').write_bytes(b'index-A')
        self.service.store.transition(task['id'],{'created'},'awaiting_retention',candidate={
            'candidate_hash':candidate_hash(candidate),'retention_checks':{'original_retrieval_no_regression':True}})
        return task,candidate

    def test_index_bytes_are_part_of_verified_candidate(self):
        task,candidate=self.prepared_candidate()
        before=candidate_hash(candidate)
        (candidate/'vector_index.pt').write_bytes(b'index-B')
        self.assertNotEqual(candidate_hash(candidate),before)


    def test_changed_candidate_is_not_retained(self):
        task,candidate=self.prepared_candidate()
        (candidate/'adapter.py').write_text('different code',encoding='utf-8')
        with self.assertRaises(Conflict):
            self.service.retain(task['id'],True)

    def test_successful_retention_keeps_frozen_code_and_original(self):
        task,candidate=self.prepared_candidate()
        before=(self.root/'baseline/rag/retrieval.py').read_bytes()
        retained=self.service.retain(task['id'],True)
        self.assertEqual(retained['status'],'retained')
        self.assertEqual(candidate_hash(Path(retained['retained_path'])),candidate_hash(candidate))
        self.assertEqual((self.root/'baseline/rag/retrieval.py').read_bytes(),before)

    def test_model_proposal_cannot_be_approved_before_finalization(self):
        task=self.service.create('检查检索',run=False)
        def fake_run(service,task_id):
            proposal=service.propose(task_id,{},'候选方案')
            with self.assertRaises(Conflict):
                service.approve(task_id,proposal['plan_hash'],run=False)
            return '诊断完成'
        with patch('optimizer.agent.run_diagnosis',fake_run):
            self.service.diagnose_deerflow(task['id'])
        self.assertEqual(self.service.store.get(task['id'])['status'],'awaiting_approval')

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        (self.root/'baseline/rag').mkdir(parents=True)
        (self.root/'baseline/rag/retrieval.py').write_text('original',encoding='utf-8')
        (self.root/'data').mkdir()
        for name,value in [('corpus',[]),('original-corpus',[]),('development',[]),('regression',[]),('heldout',[{'question':'SECRET-HOLDOUT','required':[]}])]:
            (self.root/f'data/{name}.json').write_text(json.dumps(value),encoding='utf-8')
        self.service=OptimizerService(self.root,evaluator=fake_evaluate)

    def tearDown(self):
        self.service.close()
        self.tmp.cleanup()

    def test_read_only_diagnosis_does_not_create_candidate(self):
        task=self.service.create('提高召回',run=False)
        self.service.diagnose_offline(task['id'])
        self.assertFalse((self.root/'experiments'/task['id']/'candidate').exists())
        self.assertNotIn('SECRET-HOLDOUT',json.dumps(self.service.agent_view(task['id'])))

    def test_source_change_rejects_approved_execution(self):
        task=self.service.create('提高召回',run=False)
        proposed=self.service.propose(task['id'],{'strategy':'hybrid'},'有证据的候选方案')
        (self.root/'baseline/rag/retrieval.py').write_text('changed',encoding='utf-8')
        with self.assertRaises(Conflict):
            self.service.approve(task['id'],proposed['plan_hash'],run=False)

    def test_agent_cannot_change_unapproved_candidate(self):
        task=self.service.create('提高召回',run=False)
        self.service.propose(task['id'],{'strategy':'hybrid'},'建议')
        with self.assertRaises(Conflict):
            self.service.execute(task['id'])

    def test_structured_plan_rejects_non_development_targets_and_multiple_changes(self):
        task = self.service.create('提高召回', run=False, mode='offline')
        original = self.service.diagnose_offline(task['id'])
        spec = {'hypothesis':'关键词可能补足漏召回', 'target_case_ids':['dev-miss'], 'min_recall_gain':0.1}
        with self.assertRaisesRegex(ValueError, '开发集中的漏召回题'):
            self.service.propose(task['id'], {'strategy':'hybrid'}, '候选理由', spec | {'target_case_ids':['SECRET-HOLDOUT']})
        with self.assertRaisesRegex(ValueError, '只能修改一个'):
            self.service.propose(task['id'], {'strategy':'hybrid', 'chunking':'section'}, '候选理由', spec)
        self.assertEqual(self.service.store.get(task['id'])['plan_hash'], original['plan_hash'])

    def test_approval_binds_target_and_failed_goal_blocks_retention(self):
        (self.root/'optimizer').mkdir()
        shutil.copyfile(Path(__file__).resolve().parents[1]/'optimizer/rag_adapter.py', self.root/'optimizer/rag_adapter.py')
        task = self.service.create('提高召回', run=False, mode='offline')
        task_id = task['id']
        original = self.service.diagnose_offline(task_id)
        spec = {'hypothesis':'关键词可能补足漏召回', 'target_case_ids':['dev-miss'], 'min_recall_gain':0.25}
        proposed = self.service.propose(task_id, {'strategy':'hybrid'}, '单变量对照', spec)
        with self.assertRaises(Conflict):
            self.service.approve(task_id, original['plan_hash'], run=False)
        self.assertFalse((self.service.work(task_id)/'candidate').exists())
        self.service.approve(task_id, proposed['plan_hash'], run=False)

        def graph_fixture(args, **kwargs):
            Path(args[3]).write_text('{"passed": true}', encoding='utf-8')
            return SimpleNamespace(returncode=0)

        with patch('optimizer.service.worker_python', return_value='mock-python'), patch('optimizer.service.subprocess.run', side_effect=graph_fixture):
            comparison = self.service.execute(task_id)
        config = json.loads((Path(comparison['candidate_path'])/'retrieval_config.json').read_text(encoding='utf-8'))
        self.assertEqual(config['strategy'], 'hybrid')
        self.assertEqual(comparison['experiment_assessment']['min_recall_gain'], 0.25)
        self.assertFalse(comparison['retention_checks']['approved_target_recall_gain'])
        self.assertTrue(all(v for k, v in comparison['retention_checks'].items() if k != 'approved_target_recall_gain'))
        with self.assertRaises(Conflict):
            self.service.retain(task_id, True)

    def test_changed_plan_contents_cannot_execute_after_approval(self):
        task = self.service.create('提高召回', run=False, mode='offline')
        task_id = task['id']
        proposal = self.service.diagnose_offline(task_id)
        self.service.approve(task_id, proposal['plan_hash'], run=False)
        changed = proposal['experiment_plan'] | {'min_recall_gain':0.01}
        self.service.store.transition(task_id, {'approved'}, 'approved', experiment_plan=changed)
        with self.assertRaisesRegex(Conflict, '计划与已记录版本不一致'):
            self.service.execute(task_id)
        self.assertFalse((self.service.work(task_id)/'candidate').exists())

    def approved_fixture(self):
        (self.root/'optimizer').mkdir()
        shutil.copyfile(Path(__file__).resolve().parents[1]/'optimizer/rag_adapter.py', self.root/'optimizer/rag_adapter.py')
        task = self.service.create('恢复验收', run=False, mode='offline')
        proposal = self.service.diagnose_offline(task['id'])
        return self.service.approve(task['id'], proposal['plan_hash'], run=False)

    def test_recovery_adopts_published_stage_without_repeating_evaluation(self):
        task = self.approved_fixture()
        task_id = task['id']
        save = self.service.store.record_checkpoint

        def interrupt_after_publish(task_id, stage, digest, **kwargs):
            if stage == 'development':
                raise SystemExit('crash after publish and before DB commit')
            return save(task_id, stage, digest, **kwargs)

        with patch.object(self.service.store, 'record_checkpoint', side_effect=interrupt_after_publish):
            with self.assertRaises(SystemExit):
                self.service.execute(task_id)
        self.service.close()
        self.service = OptimizerService(self.root, evaluator=fake_evaluate)
        self.assertEqual(self.service.store.get(task_id)['status'], 'awaiting_resume')
        self.service.resume(task_id, task['plan_hash'], run=False)
        with self.assertRaises(Conflict):
            self.service.resume(task_id, task['plan_hash'], run=False)

        def graph_fixture(args, **kwargs):
            Path(args[3]).write_text('{"passed": true}', encoding='utf-8')
            return SimpleNamespace(returncode=0)

        with patch.object(self.service, 'evaluator', wraps=fake_evaluate) as evaluator, \
             patch('optimizer.service.worker_python', return_value='mock-python'), \
             patch('optimizer.service.subprocess.run', side_effect=graph_fixture):
            self.service.execute(task_id)
        self.assertNotIn('development', [call.args[5] for call in evaluator.call_args_list])
        replay = self.service.replay(task_id)
        self.assertEqual(replay['status'], 'awaiting_retention')
        self.assertEqual(replay['checkpoints']['development']['state'], 'verified')
        self.assertTrue(any(event['kind'] == 'stage_adopted' for event in replay['timeline']))

    def test_second_service_cannot_recover_a_live_owners_tasks(self):
        task = self.service.create('已有执行器', run=False)
        with self.assertRaises(Conflict):
            OptimizerService(self.root, evaluator=fake_evaluate)
        self.assertEqual(self.service.store.get(task['id'])['status'], 'created')

    def test_tampered_checkpoint_is_visible_in_replay_and_blocked_on_resume(self):
        task = self.approved_fixture()
        task_id = task['id']
        with patch.object(self.service, '_develop_candidate', side_effect=SystemExit('crash before publish')):
            with self.assertRaises(SystemExit):
                self.service.execute(task_id)
        checkpoint = self.service.work(task_id)/'stages/build/payload/candidate/retrieval_config.json'
        checkpoint.write_text('{}', encoding='utf-8')
        self.service.close()
        self.service = OptimizerService(self.root, evaluator=fake_evaluate)
        count = len(self.service.store.events(task_id))
        replay = self.service.replay(task_id)
        self.assertEqual(replay['checkpoints']['build']['state'], 'invalid')
        self.assertEqual(len(self.service.store.events(task_id)), count)
        self.service.resume(task_id, task['plan_hash'], run=False)
        with self.assertRaisesRegex(Conflict, '检查点校验失败'):
            self.service.execute(task_id)

    def test_iterative_round_replans_from_dev_without_running_heldout(self):
        (self.root/'optimizer').mkdir()
        shutil.copyfile(Path(__file__).resolve().parents[1]/'optimizer/rag_adapter.py', self.root/'optimizer/rag_adapter.py')
        task = self.service.create('验证重规划', run=False, mode='offline', iterative=True)
        first = self.service.diagnose_offline(task['id'])
        self.service.approve(first['id'], first['plan_hash'], run=False)
        with patch.object(self.service, 'evaluator', wraps=fake_evaluate) as calls:
            self.service.execute(first['id'])
        self.assertEqual([call.args[5] for call in calls.call_args_list], ['development'])
        self.assertEqual(self.service.store.get(first['id'])['status'], 'awaiting_round_decision')
        with self.assertRaises(Conflict):
            self.service.finalize(first['id'], first['plan_hash'], run=False)
        second = self.service.replan(first['id'], first['plan_hash'], run=False)
        with self.assertRaises(Conflict):
            self.service.replan(first['id'], first['plan_hash'], run=False)
        with patch.object(self.service, 'evaluator', side_effect=AssertionError('baseline rerun')):
            second = self.service.diagnose_offline(second['id'])
        self.assertNotEqual(second['plan'], first['plan'])
        self.assertEqual(second['status'], 'awaiting_approval')
        self.assertEqual(second['round_number'], 2)
        self.assertEqual(second['series_id'], first['id'])
        self.assertIn('dev-miss', str(self.service.agent_view(second['id'])['previous_round']))
        self.assertNotIn('SECRET-HOLDOUT', str(self.service.agent_view(second['id'])))
        with self.assertRaises(Conflict):
            self.service.approve(second['id'], first['plan_hash'], run=False)
        self.service.store.reserve(first['id'], 2, task_cap=3)
        with self.assertRaisesRegex(Conflict, '实验系列'):
            self.service.store.reserve(second['id'], 2, task_cap=3)

    def test_finalizing_eligible_round_reuses_dev_and_closes_replanning(self):
        (self.root/'optimizer').mkdir()
        shutil.copyfile(Path(__file__).resolve().parents[1]/'optimizer/rag_adapter.py', self.root/'optimizer/rag_adapter.py')
        task = self.service.create('冻结验收', run=False, mode='offline', iterative=True)
        plan = self.service.diagnose_offline(task['id'])
        task_id, plan_hash = plan['id'], plan['plan_hash']
        self.service.approve(task_id, plan_hash, run=False)

        def good_development(*args, **kwargs):
            report = fake_evaluate(*args, **kwargs)
            self.assertEqual(args[5], 'development')
            report['metrics']['recall'] = 1
            report['cases'][0]['scores']['recall'] = 1
            return report

        with patch.object(self.service, 'evaluator', side_effect=good_development):
            self.assertTrue(self.service.execute(task_id)['eligible'])
        self.service.finalize(task_id, plan_hash, run=False)
        with self.assertRaises(Conflict):
            self.service.replan(task_id, plan_hash, run=False)

        def graph_fixture(args, **kwargs):
            Path(args[3]).write_text('{"passed": true}', encoding='utf-8')
            return SimpleNamespace(returncode=0)

        with patch.object(self.service, 'evaluator', wraps=fake_evaluate) as calls, \
             patch('optimizer.service.worker_python', return_value='mock-python'), \
             patch('optimizer.service.subprocess.run', side_effect=graph_fixture):
            comparison = self.service.execute(task_id)
        self.assertEqual([call.args[5] for call in calls.call_args_list],
                         ['heldout', 'heldout', 'original-regression', 'original-regression'])
        self.assertTrue(all(comparison['retention_checks'].values()))
        self.assertEqual(self.service.store.get(task_id)['status'], 'awaiting_retention')
