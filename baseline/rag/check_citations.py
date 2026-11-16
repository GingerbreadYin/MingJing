from citations import InvalidAnswerError, validate_citations


chunks = [{"id": "return_policy_v1-004"}]

validate_citations(
    "退款将在三个工作日内原路退回。[return_policy_v1-004]",
    chunks,
)
print("通过：合法引用")

invalid_answers = [
    ("缺少引用", "退款将在三个工作日内原路退回。"),
    ("未知编号", "退款将在三个工作日内退回。[unknown-001]"),
    (
        "混入未知编号",
        "退款说明[return_policy_v1-004]，其他说明[unknown-001]",
    ),
    ("空回答", ""),
]

for name, answer in invalid_answers:
    try:
        validate_citations(answer, chunks)
    except InvalidAnswerError:
        print(f"通过：已拦截{name}")
    else:
        raise AssertionError(f"未拦截：{name}")

print("引用检查全部通过")