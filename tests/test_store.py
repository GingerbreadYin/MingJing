import tempfile, unittest
from pathlib import Path
from optimizer.store import Store, Conflict


class ApprovalTests(unittest.TestCase):
    def test_restart_marks_lost_created_and_approved_queue_entries(self):
        other=self.store.create('queued','snapshot-1')
        task=self.store.propose(self.task['id'],{},'方案')
        self.store.approve(task['id'],task['plan_hash'])
        self.store.recover_interrupted()
        self.assertEqual(self.store.get(other['id'])['status'],'interrupted')
        self.assertEqual(self.store.get(task['id'])['status'],'interrupted')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name)/'state.db')
        self.task = self.store.create('诊断漏召回', 'snapshot-1')

    def tearDown(self):
        self.temp.cleanup()

    def test_cannot_execute_before_approval(self):
        self.store.propose(self.task['id'], {'strategy':'hybrid'}, '依据')
        with self.assertRaises(Conflict):
            self.store.claim_execution(self.task['id'])

    def test_approval_is_version_bound_and_execution_claim_is_once(self):
        task = self.store.propose(self.task['id'], {'strategy':'hybrid'}, '依据')
        with self.assertRaises(Conflict):
            self.store.approve(task['id'], 'wrong')
        self.store.approve(task['id'], task['plan_hash'])
        self.store.claim_execution(task['id'])
        with self.assertRaises(Conflict):
            self.store.claim_execution(task['id'])

    def test_recovery_does_not_replay_running_work(self):
        task = self.store.propose(self.task['id'], {'strategy':'hybrid'}, '依据')
        self.store.approve(task['id'], task['plan_hash'])
        self.store.claim_execution(task['id'])
        self.store.recover_interrupted()
        self.assertEqual(self.store.get(task['id'])['status'], 'interrupted')

    def test_cancel_revokes_approval(self):
        task = self.store.propose(self.task['id'], {'strategy':'hybrid'}, '依据')
        self.store.approve(task['id'], task['plan_hash'])
        self.store.cancel(task['id'])
        with self.assertRaises(Conflict):
            self.store.claim_execution(task['id'])

    def test_interrupted_model_reservation_stays_charged_and_is_marked_uncertain(self):
        self.store.reserve(self.task['id'], 0.15)
        self.store.recover_interrupted()
        self.assertEqual(self.store.get(self.task['id'])['status'], 'interrupted')
        budget = self.store.budget()
        self.assertEqual(budget['committed_estimate_cny'], 0.15)
        self.assertEqual(budget['calls'][0]['status'], 'uncertain')
