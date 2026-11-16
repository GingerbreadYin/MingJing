"""Filesystem confinement and immutable source fingerprints."""
import hashlib
from pathlib import Path


def confined_path(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or '..' in value.parts or ':' in relative or not relative or relative == '.':
        raise ValueError('只允许实验目录内的相对路径')
    root = root.resolve(strict=True)
    candidate = root/value
    current = candidate
    while current != root:
        if current.is_symlink() or (hasattr(current, 'is_junction') and current.is_junction()):
            raise ValueError('不允许符号链接或目录联接')
        current = current.parent
    if not candidate.resolve().is_relative_to(root):
        raise ValueError('路径越出实验目录')
    return candidate


def snapshot_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if any(part in {'.git','.venv','__pycache__','.model-cache','.pytest_cache'} for part in path.relative_to(root).parts):
            continue
        if path.is_symlink() or (hasattr(path,'is_junction') and path.is_junction()):
            raise ValueError('快照中不能包含链接')
        if path.is_file() and path.name != '.env':
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b'\0')
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()
