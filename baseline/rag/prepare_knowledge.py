import json
from pathlib import Path


project_dir = Path(__file__).parent
source_path = project_dir / "policies.txt"
target_path = project_dir / "knowledge.json"

# 防止重复运行时覆盖已经维护好的知识库
if target_path.exists():
    raise SystemExit("knowledge.json 已存在，本次不覆盖。")

lines = source_path.read_text(encoding="utf-8").splitlines()
policies = [line.strip() for line in lines if line.strip()]

records = [
    {
        "id": f"P{index:03d}",
        "text": text,
        "source": source_path.name,
    }
    for index, text in enumerate(policies, start=1)
]

target_path.write_text(
    json.dumps(records, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"已生成 knowledge.json，共 {len(records)} 条政策")