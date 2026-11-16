"""Verify adapter baseline rankings against the original retrieval.py function."""
from pathlib import Path
import importlib.util, json, os, sys, uuid

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from sentence_transformers import SentenceTransformer
from optimizer.retrieval import run_retrieval
source=root/'baseline/rag/retrieval.py'
spec=importlib.util.spec_from_file_location('original_retrieval',source)
original=importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
chunks=json.loads((root/'baseline/rag/chunks.json').read_text(encoding='utf-8'))
docs=json.loads((root/'data/original-corpus.json').read_text(encoding='utf-8'))
cases=json.loads((root/'data/regression.json').read_text(encoding='utf-8'))
model=SentenceTransformer('BAAI/bge-small-zh-v1.5',device='cpu',cache_folder=str(root/'.cache/models'),local_files_only=True)
resources={'model':model,'chunks':chunks,'vectors':model.encode([c['text'] for c in chunks],convert_to_tensor=True,show_progress_bar=False)}
output=run_retrieval(root,docs,[c['question'] for c in cases],{},root/'reports'/('baseline-equivalence-'+uuid.uuid4().hex))
rows=[]
for case,hits in zip(cases,output['results'],strict=True):
    actual=[c['id'] for c in original.rank_chunks(case['question'],resources)[:2]]
    adapted=[c['evidence_ids'][0] for c in hits]
    rows.append({'id':case['id'],'original':actual,'adapted_evidence':adapted,'same':actual==adapted})
report={'cases':rows,'all_equal':all(r['same'] for r in rows),'provider_api_calls':0,'scope':'Original rank_chunks executed with actual BGE; same text/model/query prefix, top-2'}
(root/'reports/baseline-equivalence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
assert report['all_equal'],rows
print('原检索函数与适配基线 Top-2 排序逐题一致：'+str(len(rows))+' 题')
