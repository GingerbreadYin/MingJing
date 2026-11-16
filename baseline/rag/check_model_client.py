from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError

import model_client


def http_error(code):
    return HTTPError(
        "https://api.deepseek.com/chat/completions",
        code,
        "测试错误",
        None,
        None,
    )


cases = [
    ("401 不重试", [http_error(401)], 1, 0, True),
    ("402 不重试", [http_error(402)], 1, 0, True),
    (
        "429 后恢复",
        [http_error(429), BytesIO(b'{"ok": true}')],
        2, 1, False,
    ),
    (
        "503 达到上限",
        [http_error(503), http_error(503), http_error(503)],
        3, 2, True,
    ),
]

for name, events, calls, waits, should_fail in cases:
    with (
        patch.object(model_client, "load_dotenv"),
        patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}),
        patch.object(model_client, "urlopen", side_effect=events) as send,
        patch.object(model_client.time, "sleep") as sleep,
    ):
        failed = False

        try:
            result = model_client.call_deepseek({"messages": []})
        except model_client.ModelRequestError:
            failed = True

        assert failed == should_fail, name
        assert send.call_count == calls, name
        assert sleep.call_count == waits, name

        if not should_fail:
            assert result == {"ok": True}, name

    print(f"通过：{name}")

print("模型请求重试检查通过")