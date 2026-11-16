"""Actual candidate RAG retrieval adapter, preserving the original API."""
from pathlib import Path
import json
from time import perf_counter


def load_resources(project_dir: Path):
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from .retrieval import validate_config
    raw=json.loads((project_dir/'retrieval_config.json').read_text(encoding='utf-8'))
    cache=raw.pop('model_cache')
    config=validate_config(raw)
    chunks=json.loads((project_dir/'chunks.json').read_text(encoding='utf-8'))
    model=SentenceTransformer('BAAI/bge-small-zh-v1.5',device='cpu',cache_folder=cache,local_files_only=True)
    import torch, hashlib
    source_hash=hashlib.sha256((project_dir/'chunks.json').read_bytes()).hexdigest()
    index_path=project_dir/'vector_index.pt'
    if index_path.is_symlink():
        raise ValueError('索引文件不允许符号链接')
    index=torch.load(index_path,map_location='cpu',weights_only=True) if index_path.exists() else None
    if index and index.get('source_sha256')==source_hash and index.get('model_name')=='BAAI/bge-small-zh-v1.5':
        vectors=index['vectors']
    else:
        vectors=model.encode([c['text'] for c in chunks],convert_to_tensor=True,show_progress_bar=False)
        torch.save({'vectors':vectors,'source_sha256':source_hash,'model_name':'BAAI/bge-small-zh-v1.5'},index_path)
    return {'chunks':chunks,'model':model,'vectors':vectors,'config':config,
            'reranker':CrossEncoder('BAAI/bge-reranker-base',device='cpu',cache_folder=cache,local_files_only=True) if config['rerank'] else None}


def rank_chunks(question, resources):
    if resources['config'].get('query_split', False):
        started = perf_counter()
        from .query_split import split_query

        queries = split_query(question)
        if len(queries) > 1:
            selected = rank_split_queries(queries, resources)
        else:
            selected = rank_single_query(question, resources)
            resources['last_query_split'] = {'branches': [{'query': question,
                                                            'ranked_ids': [chunk['id'] for chunk in selected]}]}
        resources['last_query_split']['elapsed_ms'] = (perf_counter() - started) * 1000
        return selected
    return rank_single_query(question, resources)


def rank_split_queries(queries, resources):
    from sentence_transformers import util
    from .query_split import build_query_graph

    config = resources['config']
    chunks = resources['chunks']
    vectors = resources['model'].encode(['为这个句子生成表示以用于检索相关文章：'+q for q in queries],
                                         convert_to_tensor=True, show_progress_bar=False)
    scores = dict(zip(queries, util.cos_sim(vectors, resources['vectors']).tolist(), strict=True))

    def retrieve(query):
        order = sorted(range(len(chunks)), key=scores[query].__getitem__, reverse=True)
        return [chunks[i] for i in order if len(chunks[i]['text']) <= config['context_chars']][:config['candidate_k']]

    expanded_k = min(config['top_k']+2,10)
    graph = build_query_graph(retrieve, top_k=expanded_k, context_chars=config['context_chars'])
    result = graph.invoke({'queries':queries}, config={'max_concurrency':3})
    selected = result['hits']
    top = selected[:config['top_k']]
    # Branch ordering is recorded independently of the final merged ranking.
    resources['last_query_split'] = {'branches': [
        {'query': branch['query'], 'ranked_ids': [hit['id'] for hit in branch['hits']]}
        for branch in result['branches']
    ]}
    resources['last_stages'] = {'initial_candidates':top, 'post_ranking':top,
                                'budgeted_context':top, 'potential_expanded_context':selected}
    return selected


def fuse_rrf(dense, sparse, vector_pct):
    scores = [0.] * len(dense)
    orders = {}
    for name, values, weight in (('vector', dense, vector_pct), ('bm25', sparse, 100-vector_pct)):
        order = sorted(range(len(values)), key=values.__getitem__, reverse=True)
        for rank, index in enumerate(order, 1):
            scores[index] += (weight / 50) / (60 + rank)
        orders[name] = order
    return scores, orders


def rank_single_query(question, resources):
    import jieba
    from rank_bm25 import BM25Okapi
    from sentence_transformers import util
    chunks,config=resources['chunks'],resources['config']
    query=resources['model'].encode('为这个句子生成表示以用于检索相关文章：'+question,convert_to_tensor=True)
    dense=util.cos_sim(query,resources['vectors'])[0].tolist()
    tokenize=lambda text:[w for w in jieba.lcut(text) if w.strip() and w.isalnum()]
    sparse=BM25Okapi([tokenize(c['text']) for c in chunks]).get_scores(tokenize(question)).tolist()
    orders = {}
    if config['strategy']=='vector':
        scores=dense
    elif config['strategy']=='bm25':
        scores=sparse
    else:
        scores, orders = fuse_rrf(dense, sparse, config.get('rrf_vector_pct', 50))
    order=sorted(range(len(chunks)),key=lambda i:scores[i],reverse=True)
    initial=[chunks[i] for i in order[:config['top_k']]]
    if resources['reranker']:
        order=order[:config['candidate_k']]
        values=resources['reranker'].predict([(question,chunks[i]['text']) for i in order],show_progress_bar=False)
        order=[i for _,i in sorted(zip(values,order),key=lambda pair:pair[0],reverse=True)]
    post_ranking=[chunks[i] for i in order[:config['top_k']]]
    selected=[]
    remaining=config['context_chars']
    for i in order:
        if len(chunks[i]['text'])>remaining:
            continue
        remaining-=len(chunks[i]['text'])
        selected.append(chunks[i])
    resources['last_stages']={'initial_candidates':initial,'post_ranking':post_ranking,'budgeted_context':selected[:config['top_k']],'potential_expanded_context':selected[:min(config['top_k']+2,10)]}
    if orders:
        resources['last_stages']['rrf_trace'] = {
            name: [chunks[i]['id'] for i in route[:config['top_k']]]
            for name, route in orders.items()
        } | {'fused': [chunk['id'] for chunk in initial]}
    return selected


def format_evidence(chunks):
    return '\n'.join(f"[{c['id']}]\n来源：{c['source']}，章节：{c['section']}\n{c['text']}" for c in chunks)
