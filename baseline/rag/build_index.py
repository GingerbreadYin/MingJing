import hashlib
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


project_dir = Path(__file__).parent
source_path = project_dir / "chunks.json"
source_bytes = source_path.read_bytes()
chunks = json.loads(source_bytes)

model_name = "BAAI/bge-small-zh-v1.5"
model = SentenceTransformer(
    model_name,
    device="cpu",
    cache_folder=str(project_dir / ".model-cache"),
)

vectors = model.encode(
    [chunk["text"] for chunk in chunks],
    convert_to_tensor=True,
)

torch.save(
    {
        "model_name": model_name,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "vectors": vectors,
    },
    project_dir / "vector_index.pt",
)

print(f"已保存 {len(chunks)} 个文本块的向量")