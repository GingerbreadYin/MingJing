"""Fixed local worker; input paths are supplied only by the trusted service."""
from pathlib import Path
import hashlib, json, sys, time


def main():
    import jieba, torch
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer, CrossEncoder, util
    from optimizer.rag_adapter import fuse_rrf
    request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    chunks,queries,config=request['chunks'],request['queries'],request['config']
    if request.get('candidate_path'):
        # Score the exact frozen files that will be retained, not a rebuilt
        # approximation of their config. The subprocess is fresh per run.
        candidate=Path(request['candidate_path'])
        sys.path.insert(0,str(candidate))
        import retrieval as candidate_retrieval
        resources=candidate_retrieval.load_retriever()
        if {'query_split':False, 'chunk_overlap':0, 'rrf_vector_pct':50, **resources['config']}!=config:
            raise RuntimeError('候选配置与批准配置不一致')
        results, stages, traces, timings = [], [], [], []
        for question in queries:
            started = time.perf_counter()
            results.append(candidate_retrieval.rank_chunks(question,resources)[:config['top_k']])
            timings.append((time.perf_counter() - started) * 1000)
            stages.append(resources['last_stages'])
            if config['query_split']:
                traces.append(resources['last_query_split'])
        output={'results':results,'stages':stages,'metadata':{'model':'BAAI/bge-small-zh-v1.5','device':'cpu','api_calls':0,
                 'candidate_artifacts':str(candidate),'queries':len(queries),'chunks':len(resources['chunks']),
                 'implementation':'frozen candidate_retrieval.adapter'}}
        output['metadata']['query_latency_ms'] = timings
        if config['query_split']:
            output['query_split_traces'] = traces
        Path(sys.argv[2]).write_text(json.dumps(output,ensure_ascii=False),encoding='utf-8')
        return
    cache=request['cache']
    index_dir=Path(request['index_dir'])
    index_dir.mkdir(parents=True,exist_ok=True)
    model_name='BAAI/bge-small-zh-v1.5'
    model=SentenceTransformer(model_name,device='cpu',cache_folder=cache,local_files_only=True)
    fingerprint=hashlib.sha256(json.dumps(chunks,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    index_path=index_dir/'vectors.pt'
    vectors=model.encode([c['text'] for c in chunks],convert_to_tensor=True,show_progress_bar=False)
    torch.save({'vectors':vectors,'corpus_sha256':fingerprint,'model_name':model_name},index_path)
    def similarities(questions):
        embeddings = model.encode(['为这个句子生成表示以用于检索相关文章：'+q for q in questions],
                                  convert_to_tensor=True, show_progress_bar=False)
        return util.cos_sim(embeddings, vectors).tolist()

    tokenize=lambda text:[word for word in jieba.lcut(text) if word.strip() and word.isalnum()]
    bm25=BM25Okapi([tokenize(c['text']) for c in chunks])
    reranker=CrossEncoder('BAAI/bge-reranker-base',device='cpu',cache_folder=cache,local_files_only=True) if config['rerank'] else None
    def rank(question, dense, limit):
        orders = {}
        sparse=bm25.get_scores(tokenize(question)).tolist()
        if config['strategy']=='vector':
            scores=dense
        elif config['strategy']=='bm25':
            scores=sparse
        else:
            scores, orders = fuse_rrf(dense, sparse, config['rrf_vector_pct'])
        order=sorted(range(len(chunks)),key=lambda i:scores[i],reverse=True)
        initial=[chunks[i] for i in order[:config['top_k']]]
        if reranker:
            order=order[:config['candidate_k']]
            values=reranker.predict([(question,chunks[i]['text']) for i in order],show_progress_bar=False)
            for i,value in zip(order,values,strict=True):
                scores[i]=float(value)
            order.sort(key=lambda i:scores[i],reverse=True)
        post_ranking=[chunks[i] for i in order[:config['top_k']]]
        hits=[]
        remaining=config['context_chars']
        for idx in order:
            if len(hits)>=limit:
                break
            # Never truncate and still claim evidence is complete.
            if len(chunks[idx]['text'])>remaining:
                continue
            remaining-=len(chunks[idx]['text'])
            hits.append(chunks[idx]|{'score':float(scores[idx]),'vector_score':float(dense[idx]),'bm25_score':float(sparse[idx])})
        expanded=[]
        remaining=config['context_chars']
        for idx in order:
            if len(expanded)>=min(config['top_k']+2,10):
                break
            if len(chunks[idx]['text'])<=remaining:
                expanded.append(chunks[idx])
                remaining-=len(chunks[idx]['text'])
        stages = {'initial_candidates':initial,'post_ranking':post_ranking,'budgeted_context':hits,'potential_expanded_context':expanded}
        if orders:
            stages['rrf_trace'] = {name:[chunks[i]['id'] for i in route[:config['top_k']]]
                                   for name, route in orders.items()} | {'fused':[chunk['id'] for chunk in initial]}
        return hits, stages

    started = time.perf_counter()
    sims = similarities(queries)
    start = time.perf_counter()
    results, stages = zip(*(rank(question, dense, config['top_k'])
                            for question, dense in zip(queries, sims, strict=True))) if queries else ([], [])
    output={'results':results,'stages':stages,'metadata':{'model':model_name,'corpus_sha256':fingerprint,'chunks':len(chunks),'queries':len(queries),'device':'cpu','ranking_seconds':time.perf_counter()-start,'index':str(index_path),'api_calls':0,'reranker':'BAAI/bge-reranker-base' if reranker else None}}
    if request.get('compare_query_split'):
        baseline_seconds = time.perf_counter() - started
        output['query_split'] = compare_split(queries, config, similarities, rank, baseline_seconds)
    Path(sys.argv[2]).write_text(json.dumps(output,ensure_ascii=False),encoding='utf-8')


def compare_split(questions, config, similarities, rank, baseline_seconds):
    from importlib.metadata import version
    from optimizer.query_split import build_query_graph, split_query

    started = time.perf_counter()
    plans = [split_query(question) for question in questions]
    unique_queries = list(dict.fromkeys(query for plan in plans for query in plan))
    scores = dict(zip(unique_queries, similarities(unique_queries), strict=True))

    def retrieve(query):
        return rank(query, scores[query], max(config['candidate_k'], config['top_k']))[0]

    graph = build_query_graph(retrieve, top_k=config['top_k'], context_chars=config['context_chars'])
    comparisons = [graph.invoke({'queries': plan}, config={'max_concurrency': 3}) for plan in plans]
    return {
        'cases': [{'hits': row['hits'], 'branches': row['branches']} for row in comparisons],
        'baseline_seconds': baseline_seconds,
        'split_seconds': time.perf_counter() - started,
        'baseline_queries': len(questions),
        'split_queries': sum(map(len, plans)),
        'split_embeddings': len(unique_queries),
        'langgraph': version('langgraph'),
    }


if __name__=='__main__':
    main()
