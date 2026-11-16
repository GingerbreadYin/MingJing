from pathlib import Path

from sentence_transformers import SentenceTransformer, util


# 1. 读取政策，每个非空行作为一条资料
project_dir = Path(__file__).parent
policy_text = (project_dir / "policies.txt").read_text(encoding="utf-8")
policies = [line.strip() for line in policy_text.splitlines() if line.strip()]

# 2. 加载本地向量模型，首次运行时需要下载
print("正在加载向量模型，首次运行需要下载……")

model = SentenceTransformer(
    "BAAI/bge-small-zh-v1.5",
    device="cpu",
    cache_folder=str(project_dir / ".model-cache"),
)

# 3. 将政策和问题分别转换为向量
policy_vectors = model.encode(policies, convert_to_tensor=True)

question = input("请输入你的问题：")
query_text = "为这个句子生成表示以用于检索相关文章：" + question
question_vector = model.encode(query_text, convert_to_tensor=True)

# 4. 计算问题与每条政策的相似度，并从高到低排序
scores = util.cos_sim(question_vector, policy_vectors)[0]
ranked_indices = scores.argsort(descending=True)

# 5. 显示所有政策的排名，方便观察
print("\n向量检索排名：")

for rank, index in enumerate(ranked_indices, start=1):
    index = int(index)
    print(f"{rank}. 得分 {float(scores[index]):.4f} | {policies[index]}")