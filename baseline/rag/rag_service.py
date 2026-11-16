from router import route_question
from retrieval import load_retriever, rank_chunks, format_evidence
from evidence import check_evidence
from generation import generate_answer


def make_result(status, answer, rounds, chunks, usage=None):
    return {
        "status": status,
        "answer": answer,
        "retrieval_rounds": rounds,
        "retrieved_sources": [
            {
                "id": chunk["id"],
                "source": chunk["source"],
                "section": chunk["section"],
            }
            for chunk in chunks
        ],
        "final_answer_usage": usage,
    }


def ask_rag(question, resources=None):
    question = question.strip()

    if not question:
        raise ValueError("问题不能为空")

    if route_question(question) == "DIRECT":
        return make_result(
            "DIRECT",
            "你好，我可以帮你查询店铺的退货、退款和售后政策。",
            0,
            [],
        )

    # 允许调用方传入已经加载的资源，避免每次重新加载模型
    if resources is None:
        resources = load_retriever()

    ranked_chunks = rank_chunks(question, resources)

    rounds = 1
    selected_chunks = ranked_chunks[:2]
    evidence = format_evidence(selected_chunks)
    assessment = check_evidence(question, evidence)

    if assessment["status"] == "NEED_RETRIEVAL":
        rounds = 2
        selected_chunks = ranked_chunks[:4]
        evidence = format_evidence(selected_chunks)
        assessment = check_evidence(question, evidence)

    if assessment["status"] == "NEED_CLARIFICATION":
        return make_result(
            "NEED_CLARIFICATION",
            assessment["question"],
            rounds,
            selected_chunks,
        )

    if assessment["status"] == "NEED_RETRIEVAL":
        return make_result(
            "INSUFFICIENT_EVIDENCE",
            "两轮检索后仍缺少必要依据，无法确定，请向客服核实。",
            rounds,
            selected_chunks,
        )

    answer, usage = generate_answer(question, evidence)

    return make_result(
        "ANSWERED",
        answer,
        rounds,
        selected_chunks,
        usage,
    )