from typing import TypedDict
from citations import validate_citations

from langgraph.graph import StateGraph, START, END

from router import route_question
from retrieval import rank_chunks, format_evidence
from evidence import check_evidence
from generation import generate_answer


class RagState(TypedDict, total=False):
    question: str
    strategy: str
    ranked_chunks: list[dict]
    selected_chunks: list[dict]
    evidence: str
    assessment: dict
    rounds: int
    status: str
    answer: str
    final_answer_usage: dict | None


def build_rag_graph(resources):
    def route_node(state):
        print("执行节点：route")
        question = state["question"].strip()
        if not question:
            raise ValueError("问题不能为空")
        return {
            "question": question,
            "strategy": route_question(question),
        }

    def direct_node(state):
        print("执行节点：direct")
        return {
            "status": "DIRECT",
            "answer": "你好，我可以帮你查询店铺政策。",
            "rounds": 0,
            "selected_chunks": [],
            "final_answer_usage": None,
        }

    def retrieve_node(state):
        print("执行节点：retrieve")
        ranked = rank_chunks(state["question"], resources)
        selected = ranked[:2]
        return {
            "ranked_chunks": ranked,
            "selected_chunks": selected,
            "evidence": format_evidence(selected),
            "rounds": 1,
        }

    def check_node(state):
        print(f"执行节点：check，第 {state['rounds']} 轮")
        assessment = check_evidence(state["question"], state["evidence"])
        print("检查结果：", assessment["status"], assessment["reason"])
        return {"assessment": assessment}

    def expand_node(state):
        print("执行节点：expand")
        selected = state["ranked_chunks"][:4]
        return {
            "selected_chunks": selected,
            "evidence": format_evidence(selected),
            "rounds": state["rounds"] + 1,
        }

    def generate_node(state):
        print("执行节点：generate")
        answer, usage = generate_answer(state["question"], state["evidence"])
        validate_citations(answer, state["selected_chunks"])
        return {
            "status": "ANSWERED",
            "answer": answer,
            "final_answer_usage": usage,
        }

    def clarify_node(state):
        print("执行节点：clarify")
        return {
            "status": "NEED_CLARIFICATION",
            "answer": state["assessment"]["question"],
            "final_answer_usage": None,
        }

    def insufficient_node(state):
        print("执行节点：insufficient")
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "answer": "两轮检索后仍缺少必要依据，无法确定，请向客服核实。",
            "final_answer_usage": None,
        }

    def after_route(state):
        return state["strategy"]

    def after_check(state):
        status = state["assessment"]["status"]
        if status == "ANSWERABLE":
            return "generate"
        if status == "NEED_CLARIFICATION":
            return "clarify"
        if status == "NEED_RETRIEVAL":
            return "expand" if state["rounds"] < 2 else "insufficient"
        raise ValueError(f"未知证据状态：{status}")

    builder = StateGraph(RagState)

    nodes = {
        "route": route_node,
        "direct": direct_node,
        "retrieve": retrieve_node,
        "check": check_node,
        "expand": expand_node,
        "generate": generate_node,
        "clarify": clarify_node,
        "insufficient": insufficient_node,
    }

    for name, function in nodes.items():
        builder.add_node(name, function)

    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route", after_route,
        {"DIRECT": "direct", "RAG": "retrieve"},
    )
    builder.add_edge("retrieve", "check")
    builder.add_conditional_edges(
        "check", after_check,
        {
            "generate": "generate",
            "clarify": "clarify",
            "expand": "expand",
            "insufficient": "insufficient",
        },
    )
    builder.add_edge("expand", "check")

    for name in ["direct", "generate", "clarify", "insufficient"]:
        builder.add_edge(name, END)

    return builder.compile()