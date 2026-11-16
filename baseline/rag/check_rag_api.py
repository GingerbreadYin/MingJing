from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import rag_api


test_resources = {"test": True}
fake_graph = Mock()
fake_graph.invoke.return_value = {
    "status": "DIRECT",
    "answer": "测试回复",
    "rounds": 0,
    "selected_chunks": [],
    "final_answer_usage": None,
}

with (
    patch.object(rag_api, "load_retriever", return_value=test_resources),
    patch.object(rag_api, "build_rag_graph", return_value=fake_graph) as build,
):
    with TestClient(rag_api.app) as client:
        assert client.get("/health").status_code == 200

        for body in [{}, {"question": ""}, {"question": "   "}]:
            response = client.post("/chat", json=body)
            assert response.status_code == 422

        fake_graph.invoke.assert_not_called()

        response = client.post("/chat", json={"question": "  你好  "})
        assert response.status_code == 200
        assert response.json() == {
            "status": "DIRECT",
            "answer": "测试回复",
            "retrieval_rounds": 0,
            "retrieved_sources": [],
            "final_answer_usage": None,
        }

        fake_graph.invoke.assert_called_once_with({"question": "你好"})
        build.assert_called_once_with(test_resources)
                # 模拟图执行过程中模型请求失败
        fake_graph.invoke.side_effect = rag_api.ModelRequestError(
            "测试用模型故障"
        )

        response = client.post("/chat", json={"question": "退款多久退回？"})

        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "MODEL_REQUEST_FAILED"

        # 单次问答失败后，服务仍然能够响应健康检查
        assert client.get("/health").status_code == 200

print("接口检查通过：正常请求、输入校验、模型故障转换均符合预期")