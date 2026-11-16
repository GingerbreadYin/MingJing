from retrieval import load_retriever
from rag_graph import build_rag_graph


resources = load_retriever()
graph = build_rag_graph(resources)

question = input("请输入你的问题：")
result = graph.invoke({"question": question})

print("\n最终状态：", result["status"])
print("检索轮次：", result["rounds"])
print("回答：", result["answer"])