import tempfile, unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from optimizer.store import Store, Conflict


class BudgetTests(unittest.TestCase):
    def test_per_task_spending_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp)/'state.db')
            store.reserve('task',2,task_cap=3)
            with self.assertRaises(Conflict):
                store.reserve('task',2,task_cap=3)

    def test_physical_call_count_is_transactionally_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp)/'state.db')
            first = store.create('first round','snapshot',iterative=True)
            second = store.create('second round','snapshot')
            store.transition(second['id'],{'created'},'created',series_id=first['series_id'])
            for task_id in (first['id'],second['id']):
                for _ in range(30):
                    store.reserve(task_id,.01,task_cap=5)
            with self.assertRaisesRegex(Conflict, '60 次'):
                store.reserve(second['id'],.01,task_cap=5)

    def test_concurrent_reservations_cannot_overbook_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp)/'state.db')
            def reserve():
                try:
                    return store.reserve('test',50,70)
                except Conflict:
                    return None
            with ThreadPoolExecutor(max_workers=2) as pool:
                results=list(pool.map(lambda _:reserve(),range(2)))
            self.assertEqual(sum(bool(r) for r in results),1)
            self.assertEqual(store.budget()['committed_estimate_cny'],50)

    def test_uncertain_call_retains_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp)/'state.db')
            call=store.reserve('test',1)
            store.settle(call,None)
            self.assertEqual(store.budget()['committed_estimate_cny'],1)
