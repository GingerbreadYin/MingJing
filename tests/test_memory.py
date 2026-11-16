from pathlib import Path
import tempfile
import unittest

from optimizer.memory import SessionMemory, encoded_context
from optimizer.store import Store


class MemoryTests(unittest.TestCase):
    def test_reads_ancestor_sources_without_crossing_series_or_audience(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'optimizer.db'
            store = Store(path)
            parent = store.create('第一轮', 'fixture', iterative=True)
            other = store.create('其他实验', 'fixture', iterative=True)
            for task, note, public in [(parent, '固定 Top-2', True), (parent, '本机私有备忘', False),
                                       (other, '其他实验内容', True)]:
                store.event(task['id'], 'series_memory_note', {'text': note, 'include_in_model': public})
            store.transition(parent['id'], {'created'}, 'awaiting_round_decision', plan={}, mode='offline',
                             round_summary={'metrics': {'recall': .5}, 'checks': {'passed': False},
                                            'case_deltas': [{'id': 'demo-13-combined'}]})
            store.create_replan(parent['id'], None, {})
            child = next(task for task in store.list() if task.get('previous_task_id') == parent['id'])
            store.event(child['id'], 'series_memory_note', {'text': '后续轮次备注', 'include_in_model': True})
            rebuilt = SessionMemory(Store(path))
            result = rebuilt.view(child['id'], for_model=True)
            prompt = encoded_context(result['context'])
            self.assertIn('固定 Top-2', prompt)
            self.assertIn('demo-13-combined', prompt)
            self.assertNotIn('其他实验内容', prompt)
            self.assertNotIn('本机私有备忘', prompt)
            for entry in result['context']['entries']:
                source = entry['source']
                self.assertTrue(any(event['seq'] == source['seq'] for event in store.events(source['task_id'])))
            parent_view = encoded_context(rebuilt.view(parent['id'])['context'])
            self.assertIn('本机私有备忘', parent_view)
            self.assertNotIn('后续轮次备注', parent_view)

    def test_memory_has_hard_character_limit_and_keeps_recent_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / 'optimizer.db')
            task_id = store.create('压缩', 'fixture', iterative=True)['id']
            for i in range(20):
                store.event(task_id, 'series_memory_note', {'text': f'备注{i} ' + '证据' * 270})
            context = SessionMemory(store).view(task_id)['context']
            self.assertLessEqual(len(encoded_context(context)), 1000)
            self.assertGreater(len(context['entries']), 1)
            self.assertIn('备注19', context['entries'][0]['text'])
            self.assertTrue(context['entries'][0]['truncated'])
            self.assertEqual(context['omitted'], 20 - len(context['entries']))
