"""Freeze candidate implementation so retention cannot follow mutable host code."""
from pathlib import Path
import hashlib, inspect, json, shutil
from .retrieval import validate_config, DEFAULT_CONFIG, build_chunks
from .isolation import confined_path


def candidate_hash(candidate: Path,include_index=True):
    digest=hashlib.sha256()
    for p in sorted(candidate.rglob('*')):
        if p.is_symlink() or (hasattr(p,'is_junction') and p.is_junction()):
            raise ValueError('候选不允许链接')
        if p.is_file() and '__pycache__' not in p.parts and (include_index or p.name!='vector_index.pt'):
            digest.update(p.relative_to(candidate).as_posix().encode())
            digest.update(b'\0')
            digest.update(hashlib.sha256(p.read_bytes()).digest())
    return digest.hexdigest()


def freeze_adapter(root: Path,candidate: Path, *, query_split=False):
    package=confined_path(candidate,'candidate_retrieval')
    package.mkdir()
    (package/'__init__.py').write_text('',encoding='utf-8')
    source=(root/'optimizer/rag_adapter.py').read_text(encoding='utf-8')
    source=source.replace('from .retrieval import validate_config','from .config import validate_config')
    (package/'adapter.py').write_text(source,encoding='utf-8')
    (package/'config.py').write_text('DEFAULT_CONFIG = '+repr(DEFAULT_CONFIG)+'\n\n'+inspect.getsource(validate_config),encoding='utf-8')
    if query_split:
        shutil.copyfile(root/'optimizer/query_split.py', package/'query_split.py')


def write_chunks(candidate: Path,documents: list[dict],config: dict):
    chunks=build_chunks(documents,config)
    for chunk in chunks:
        chunk.update(source=chunk['title'],section=chunk['title'])
    confined_path(candidate,'chunks.json').write_text(json.dumps(chunks,ensure_ascii=False,indent=2),encoding='utf-8')


def record_diff(baseline: Path,candidate: Path,output: Path):
    import difflib
    diffs=[]
    for p in sorted(candidate.rglob('*')):
        if not p.is_file() or p.suffix not in {'.py','.json','.md'} or '__pycache__' in p.parts:
            continue
        rel=p.relative_to(candidate)
        old=baseline/rel
        before=old.read_text(encoding='utf-8') if old.exists() else ''
        after=p.read_text(encoding='utf-8')
        if before!=after:
            diffs.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='baseline/'+rel.as_posix(),tofile='candidate/'+rel.as_posix()))
    output.write_text(''.join(diffs),encoding='utf-8')
