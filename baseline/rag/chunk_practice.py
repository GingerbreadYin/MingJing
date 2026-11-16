import json
from pathlib import Path


project_dir = Path(__file__).parent

# 明确指定文档 ID，保留原有退货文档的编号
documents = [
    {
        "file": "return_policy_demo.md",
        "document_id": "return_policy_v1",
    },
    {
        "file": "invoice_policy_demo.md",
        "document_id": "invoice_policy_v1",
    },
]

chunks = []

for document in documents:
    source_path = project_dir / document["file"]
    text = source_path.read_text(encoding="utf-8")
    sections = [
        part.strip()
        for part in text.split("\n\n")
        if part.strip()
    ]

    for number, section in enumerate(sections, start=1):
        title = section.splitlines()[0].removeprefix("## ").strip()

        chunks.append({
            "id": f"{document['document_id']}-{number:03d}",
            "document_id": document["document_id"],
            "source": source_path.name,
            "section": title,
            "text": section,
        })

    print(f"{document['file']}：生成 {len(sections)} 块")

(project_dir / "chunks.json").write_text(
    json.dumps(chunks, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"共生成 {len(chunks)} 个文本块")