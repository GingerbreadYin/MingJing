import re


class InvalidAnswerError(RuntimeError):
    """模型回答未通过输出校验。"""


def validate_citations(answer, chunks):
    if not isinstance(answer, str) or not answer.strip():
        raise InvalidAnswerError("模型回答为空")

    # 提取当前项目约定格式的引用编号
    cited_ids = set(re.findall(r"\[([A-Za-z0-9_-]+)\]", answer))
    allowed_ids = {chunk["id"] for chunk in chunks}

    if not cited_ids:
        raise InvalidAnswerError("政策回答缺少引用")

    unknown_ids = cited_ids - allowed_ids

    if unknown_ids:
        raise InvalidAnswerError(
            f"回答包含未提供的引用编号：{sorted(unknown_ids)}"
        )