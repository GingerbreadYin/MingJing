from unittest.mock import patch

import rag_graph


def assessment(status):
    return {
        "status": status,
        "reason": "测试判断原因",
        "question": "请说明商品损坏原因。" if status == "NEED_CLARIFICATION" else "",
    }


# 只提供测试所需的数据结构，不使用真实政策内容
chunks = [
    {
        "id": f"test-{i}",
        "source": "test.md",
        "section": f"测试章节{i}",
        "text": f"测试条款{i}",
    }
    for i in range(1, 5)
]

# 名称、问题、依次返回的检查结果、预期状态、预期轮次
cases = [
    ("直接回复", "你好", [], "DIRECT", 0),
    ("一轮回答", "测试政策问题", ["ANSWERABLE"], "ANSWERED", 1),
    ("需要澄清", "测试政策问题", ["NEED_CLARIFICATION"], "NEED_CLARIFICATION", 1),
    ("补检后回答", "测试政策问题",
     ["NEED_RETRIEVAL", "ANSWERABLE"], "ANSWERED", 2),
    ("补检后仍不足", "测试政策问题",
     ["NEED_RETRIEVAL", "NEED_RETRIEVAL"], "INSUFFICIENT_EVIDENCE", 2),
]

for name, question, checks, expected_status, expected_rounds in cases:
    with (
        patch.object(rag_graph, "rank_chunks", return_value=chunks) as rank,
        patch.object(
            rag_graph,
            "check_evidence",
            side_effect=[assessment(status) for status in checks],
        ) as check,
        patch.object(
            rag_graph,
            "generate_answer",
                return_value=("测试回答[test-1]", None),
        ) as generate,
    ):
        graph = rag_graph.build_rag_graph(resources={})
        result = graph.invoke(
            {"question": question},
            config={"recursion_limit": 20},
        )

        assert result["status"] == expected_status, name
        assert result["rounds"] == expected_rounds, name
        assert check.call_count == len(checks), name
        assert generate.call_count == int(expected_status == "ANSWERED"), name
        assert rank.call_count == int(expected_rounds > 0), name

        if expected_rounds == 2:
            assert len(result["selected_chunks"]) == 4, name

    print(f"通过：{name}")

print("全部图路径检查通过")