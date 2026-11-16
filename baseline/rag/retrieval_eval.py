from pathlib import Path

from sentence_transformers import SentenceTransformer, util


project_dir = Path(__file__).parent
policy_text = (project_dir / "policies.txt").read_text(encoding="utf-8")
policies = [line.strip() for line in policy_text.splitlines() if line.strip()]

# 每题的 expected 是人工确定的正确政策编号
cases = [
    {"question": "因商品质量问题退货，谁承担运费？", "expected": "P003"},
    {"question": "收到的东西有质量缺陷，寄回去的邮费谁出？", "expected": "P003"},
    {"question": "个人原因退货，运费谁承担？", "expected": "P004"},
    {"question": "买回来后不喜欢了，寄回去的邮费谁出？", "expected": "P004"},
    {"question": "定制商品存在质量问题，可以申请售后吗？", "expected": "P002"},
    {"question": "普通商品未使用且包装完整，签收后几天内可以无理由退货？", "expected": "P001"},
]

model = SentenceTransformer(
    "BAAI/bge-small-zh-v1.5",
    device="cpu",
    cache_folder=str(project_dir / ".model-cache"),
)

policy_vectors = model.encode(policies, convert_to_tensor=True)

hit1_count = 0
hit2_count = 0

for case in cases:
    query = "为这个句子生成表示以用于检索相关文章：" + case["question"]
    question_vector = model.encode(query, convert_to_tensor=True)

    scores = util.cos_sim(question_vector, policy_vectors)[0]
    top_indices = scores.argsort(descending=True)[:2]
    retrieved_ids = [f"P{int(index) + 1:03d}" for index in top_indices]

    hit1 = case["expected"] == retrieved_ids[0]
    hit2 = case["expected"] in retrieved_ids

    hit1_count += int(hit1)
    hit2_count += int(hit2)

    print(f"\n问题：{case['question']}")
    print(f"正确依据：{case['expected']}，检索结果：{retrieved_ids}")
    print(f"第一条命中：{hit1}，前两条命中：{hit2}")

print(f"\n共评测 {len(cases)} 个问题")
print(f"Hit@1：{hit1_count / len(cases):.1%}")
print(f"Hit@2：{hit2_count / len(cases):.1%}")