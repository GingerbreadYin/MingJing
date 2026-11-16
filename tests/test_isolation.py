import tempfile, unittest
from pathlib import Path
from optimizer.isolation import confined_path, snapshot_hash


class IsolationTests(unittest.TestCase):
    def test_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            root=Path(tmp)
            try:
                (root/'linked').symlink_to(outside,target_is_directory=True)
            except OSError:
                self.skipTest('当前 Windows 账户没有创建符号链接权限')
            with self.assertRaises(ValueError):
                confined_path(root,'linked/file.txt')

    def test_traversal_and_absolute_path_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for value in ['../outside.txt', str(Path(tmp).parent/'outside.txt')]:
                with self.assertRaises(ValueError):
                    confined_path(Path(tmp), value)

    def test_source_changes_invalidate_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'policy.md').write_text('A', encoding='utf-8')
            before=snapshot_hash(root)
            (root/'policy.md').write_text('B', encoding='utf-8')
            self.assertNotEqual(snapshot_hash(root), before)
