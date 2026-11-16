def route_question(question):
    normalized = question.strip().rstrip("！!。.?？")

    if normalized in {"你好", "您好", "嗨", "hello", "Hello"}:
        return "DIRECT"

    return "RAG"