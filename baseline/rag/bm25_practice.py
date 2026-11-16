from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi


# 问题与政策使用同一种分词方式，去掉标点和空白
def tokenize(text):
    return [word for word in jieba.lcut(text) if word.isalnum()]


# 1. 读取政策
project_dir = Path(__file__).parent
policy_text = (project_dir / "policies.txt").read_text(encoding="utf-8")
policies = [line.strip() for line in policy_text.splitlines() if line.strip()]

# 2. 将政策分词，建立 BM25 检索对象
tokenized_policies = [tokenize(policy) for policy in policies]
retriever = BM25Okapi(tokenized_policies)

# 3. 对问题分词，计算每条政策的分数
question = input("请输入你的问题：")
question_words = tokenize(question)
scores = retriever.get_scores(question_words)

# 4. 按分数从高到低排序，仅展示正分候选
ranked_indices = sorted(
    range(len(policies)),
    key=lambda index: scores[index],
    reverse=True,
)
matched_indices = [index for index in ranked_indices if scores[index] > 0]

print("问题分词：", question_words)
print("\nBM25 前三条结果：")

for index in matched_indices[:3]:
    print(f"[P{index + 1:03d}] 得分 {scores[index]:.4f} | {policies[index]}")

if not matched_indices:
    print("没有正分匹配结果")