import hashlib
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util


def load_retriever():
    """读取资料、索引并加载模型，返回检索所需资源。"""
    project_dir = Path(__file__).parent
    source_bytes = (project_dir / "chunks.json").read_bytes()
    chunks = json.loads(source_bytes)

    index = torch.load(
        project_dir / "vector_index.pt",
        map_location="cpu",
        weights_only=True,
    )

    if index["source_sha256"] != hashlib.sha256(source_bytes).hexdigest():
        raise RuntimeError("文档块已变化，请重新运行 build_index.py")

    model_name = "BAAI/bge-small-zh-v1.5"

    if index["model_name"] != model_name:
        raise RuntimeError("索引模型与问题向量模型不一致")

    model = SentenceTransformer(
        model_name,
        device="cpu",
        cache_folder=str(project_dir / ".model-cache"),
    )

    return {
        "chunks": chunks,
        "model": model,
        "vectors": index["vectors"],
    }


def rank_chunks(question, resources):
    """计算相关性，返回从高到低排列的完整文档块记录。"""
    query = "为这个句子生成表示以用于检索相关文章：" + question
    query_vector = resources["model"].encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_vector, resources["vectors"])[0]
    indices = scores.argsort(descending=True)

    return [resources["chunks"][int(index)] for index in indices]


def format_evidence(chunks):
    """将文档块整理成带编号、来源和章节的参考资料。"""
    return "\n".join(
        f"[{chunk['id']}]\n"
        f"来源：{chunk['source']}，章节：{chunk['section']}\n"
        f"{chunk['text']}"
        for chunk in chunks
    )