import json
import os
import random
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


class ModelRequestError(RuntimeError):
    """向调用方报告模型请求失败。"""


def call_deepseek(payload):
    load_dotenv(Path(__file__).parent / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY")

    if not api_key:
        raise ModelRequestError("没有读取到 DeepSeek 密钥，请检查 .env")

    request = Request(
        url="https://api.deepseek.com/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    max_attempts = 3
    retryable_codes = {429, 500, 502, 503, 504}

    for attempt in range(max_attempts):
        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))

        except HTTPError as error:
            if error.code not in retryable_codes:
                raise ModelRequestError(
                    f"模型请求失败，HTTP {error.code}，未自动重试"
                ) from error

            if attempt == max_attempts - 1:
                raise ModelRequestError(
                    f"模型请求尝试 {max_attempts} 次后仍失败，HTTP {error.code}"
                ) from error

        except (URLError, TimeoutError) as error:
            if attempt == max_attempts - 1:
                raise ModelRequestError(
                    f"模型请求尝试 {max_attempts} 次后仍出现网络错误或超时"
                ) from error

        delay = 2 ** attempt + random.uniform(0, 0.5)
        print(f"模型请求暂时失败，{delay:.2f} 秒后重试")
        time.sleep(delay)