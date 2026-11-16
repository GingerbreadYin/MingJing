from pathlib import Path
from time import perf_counter
import json

import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util


def tokenize(text):
    return [word for word in jieba.lcut(text) if word.isalnum()]


project_dir = Path(__file__).parent
records = json.loads(
    (project_dir / "knowledge.json").read_text(encoding="utf-8")
)
policies = [record["text"] for record in records]

# 旧六题加上新增六题，正确依据由政策内容确定
cases = [
    ("因商品质量问题退货，谁承担运费？", "P003"),
    ("收到的东西有质量缺陷，寄回去的邮费谁出？", "P003"),
    ("个人原因退货，运费谁承担？", "P004"),
    ("买回来后不喜欢了，寄回去的邮费谁出？", "P004"),
    ("定制商品存在质量问题，可以申请售后吗？", "P002"),
    ("普通商品未使用且包装完整，签收后几天内可以无理由退货？", "P001"),
    ("退货商品已经验收通过，退款几天原路退回？", "P005"),
    ("仅退款申请审核通过，要寄回商品吗？退款多久退回？", "P006"),
    ("订单还没发货，可以修改收货地址吗？", "P007"),
    ("订单已经发货，可以直接修改收货地址吗？", "P008"),
    ("偏远地区发货后预计几天送达？", "P009"),
    ("申请电子发票后，多久会发送？", "P012"),
]

model = SentenceTransformer(
    "BAAI/bge-small-zh-v1.5",
    device="cpu",
    cache_folder=str(project_dir / ".model-cache"),
)
reranker = CrossEncoder(
    "BAAI/bge-reranker-base",
    device="cpu",
    cache_folder=str(project_dir / ".model-cache"),
)
policy_vectors = model.encode(policies, convert_to_tensor=True)
bm25 = BM25Okapi([tokenize(policy) for policy in policies])

# 每种方案分别累计第一条命中数、前两条命中数
totals = {
    "向量": [0, 0],
    "混合": [0, 0],
    "混合+重排": [0, 0],
}

for question, expected in cases:
    # 向量检索
    query = "为这个句子生成表示以用于检索相关文章：" + question
    query_vector = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_vector, policy_vectors)[0]
    vector_order = [int(i) for i in scores.argsort(descending=True)]

    # BM25 检索
    bm25_scores = bm25.get_scores(tokenize(question))
    bm25_order = sorted(
        range(len(policies)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )
    bm25_candidates = [i for i in bm25_order if bm25_scores[i] > 0][:3]

    # 两路各取最多三条，进行 RRF 融合
    rrf_scores = {}
    for candidates in [bm25_candidates, vector_order[:3]]:
        for rank, index in enumerate(candidates, start=1):
            rrf_scores[index] = rrf_scores.get(index, 0) + 1 / (60 + rank)

    hybrid_order = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

    # 对融合后的同一组候选重排
    candidates = hybrid_order[:6]
    pairs = [(question, policies[i]) for i in candidates]
    rerank_start = perf_counter()
    rerank_scores = reranker.predict(pairs)
    rerank_seconds = perf_counter() - rerank_start
    print(f"\n重排 {len(pairs)} 条候选，耗时：{rerank_seconds:.3f} 秒")
    reranked_pairs = sorted(
        zip(candidates, rerank_scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    reranked_order = [index for index, score in reranked_pairs]

    results = {
        "向量": vector_order[:2],
        "混合": hybrid_order[:2],
        "混合+重排": reranked_order[:2],
    }

    print(f"\n问题：{question} | 正确依据：{expected}")

    for name, indices in results.items():
        ids = [records[i]["id"] for i in indices]
        hit1 = bool(ids) and ids[0] == expected
        hit2 = expected in ids
        totals[name][0] += int(hit1)
        totals[name][1] += int(hit2)
        print(f"{name}：{ids} | Hit@1={hit1}，Hit@2={hit2}")

print(f"\n汇总：{len(cases)} 题，{len(policies)} 条政策")

for name, counts in totals.items():
    print(
        f"{name}："
        f"Hit@1={counts[0] / len(cases):.1%}，"
        f"Hit@2={counts[1] / len(cases):.1%}"
    )