from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from router import route_question
from retrieval import load_retriever, rank_chunks, format_evidence


# 描述状态可能包含的字段；字段可以由后续节点补充
class RagState(TypedDict, total=False):
    question: str
    strategy: str
    answer: str
    evidence: str


def build_graph(resources):
    def route_node(state):
        print("执行节点：route")
        return {"strategy": route_question(state["question"])}

    def direct_node(state):
        print("执行节点：direct")
        return {"answer": "你好，我可以帮你查询店铺政策。"}

    def retrieve_node(state):
        print("执行节点：retrieve")
        ranked = rank_chunks(state["question"], resources)
        return {"evidence": format_evidence(ranked[:2])}

    def choose_path(state):
        return state["strategy"]

    builder = StateGraph(RagState)

    builder.add_node("route", route_node)
    builder.add_node("direct", direct_node)
    builder.add_node("retrieve", retrieve_node)

    builder.add_edge(START, "route")

    builder.add_conditional_edges(
        "route",
        choose_path,
        {
            "DIRECT": "direct",
            "RAG": "retrieve",
        },
    )

    builder.add_edge("direct", END)
    builder.add_edge("retrieve", END)

    return builder.compile()


# 直接运行本文件时才执行下面的交互代码
if __name__ == "__main__":
    resources = load_retriever()
    graph = build_graph(resources)

    question = input("请输入你的问题：")
    result = graph.invoke({"question": question})

    print("\n最终状态：")
    print(result)