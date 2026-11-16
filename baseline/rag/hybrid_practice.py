from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util


def tokenize(text):
    return [word for word in jieba.lcut(text) if word.isalnum()]


project_dir = Path(__file__).parent
policy_text = (project_dir / "policies.txt").read_text(encoding="utf-8")
policies = [line.strip() for line in policy_text.splitlines() if line.strip()]

# 1. 准备两种检索方法
bm25 = BM25Okapi([tokenize(policy) for policy in policies])

model = SentenceTransformer(
    "BAAI/bge-small-zh-v1.5",
    device="cpu",
    cache_folder=str(project_dir / ".model-cache"),
)
policy_vectors = model.encode(policies, convert_to_tensor=True)

question = input("请输入你的问题：")

# 2. BM25 取前三条正分候选
bm25_scores = bm25.get_scores(tokenize(question))
bm25_order = sorted(
    range(len(policies)),
    key=lambda index: bm25_scores[index],
    reverse=True,
)
bm25_candidates = [
    index for index in bm25_order if bm25_scores[index] > 0
][:3]

# 3. 向量检索取前三条候选
query = "为这个句子生成表示以用于检索相关文章：" + question
question_vector = model.encode(query, convert_to_tensor=True)
vector_scores = util.cos_sim(question_vector, policy_vectors)[0]
vector_candidates = [
    int(index)
    for index in vector_scores.argsort(descending=True)[:3]
]

# 4. RRF 融合：用政策位置作为键，累计两份排名的贡献
rrf_scores = {}

for candidates in [bm25_candidates, vector_candidates]:
    for rank, index in enumerate(candidates, start=1):
        rrf_scores[index] = rrf_scores.get(index, 0) + 1 / (60 + rank)

hybrid_order = sorted(
    rrf_scores,
    key=lambda index: rrf_scores[index],
    reverse=True,
)

# 5. 展示两路排名和融合结果
print("\nBM25：", [f"P{i + 1:03d}" for i in bm25_candidates])
print("向量：", [f"P{i + 1:03d}" for i in vector_candidates])
print("\n融合后的前三条：")

for index in hybrid_order[:3]:
    print(
        f"[P{index + 1:03d}] "
        f"RRF {rrf_scores[index]:.5f} | {policies[index]}"
    )

# 6. 对融合结果中的最多六条候选进行重排
print("\n正在加载重排模型，首次运行需要下载……")

reranker = CrossEncoder(
    "BAAI/bge-reranker-base",
    device="cpu",
    cache_folder=str(project_dir / ".model-cache"),
)

candidate_indices = hybrid_order[:6]
pairs = [(question, policies[index]) for index in candidate_indices]
rerank_scores = reranker.predict(pairs)

# 将政策位置与重排得分对应，再按得分排序
reranked = sorted(
    zip(candidate_indices, rerank_scores),
    key=lambda item: float(item[1]),
    reverse=True,
)

print("\n重排后的前三条：")

for index, score in reranked[:3]:
    print(f"[P{index + 1:03d}] 重排得分 {float(score):.4f} | {policies[index]}")