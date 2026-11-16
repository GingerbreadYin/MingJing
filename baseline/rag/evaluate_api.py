import json
import re
from pathlib import Path
from time import perf_counter
import sys
import httpx


project_dir = Path(__file__).parent
cases_path = project_dir / (
    sys.argv[1] if len(sys.argv) > 1 else "e2e_cases.json"
)
cases = json.loads(cases_path.read_text(encoding="utf-8"))
reports = []

for case in cases:
    print(f"\n正在测试 {case['id']}：{case['question']}")
    started = perf_counter()

    try:
        response = httpx.post(
            "http://127.0.0.1:8010/chat",
            json={"question": case["question"]},
            timeout=180,
            trust_env=False,
        )
        response.raise_for_status()
        result = response.json()

    except httpx.HTTPError as error:
        reports.append({
            "id": case["id"],
            "automatic_pass": False,
            "error": str(error),
            "elapsed_seconds": round(perf_counter() - started, 3),
        })
        print("请求失败：", error)
        continue

    answer = result.get("answer", "")
    status_ok = result.get("status") == case["expected_status"]

    # 所有要求的引用，是否出现在回答中
    required_ok = all(
        f"[{citation}]" in answer
        for citation in case["required_citations"]
    )

    # 回答中出现的资料编号，是否属于本次检索来源
    used_ids = set(re.findall(r"\[([A-Za-z0-9_-]+)\]", answer))
    supplied_ids = {
        source["id"]
        for source in result.get("retrieved_sources", [])
    }
    unknown_ids = sorted(used_ids - supplied_ids)
    citations_ok = required_ok and not unknown_ids

    automatic_pass = status_ok and citations_ok

    reports.append({
        "id": case["id"],
        "question": case["question"],
        "expected_status": case["expected_status"],
        "status_ok": status_ok,
        "citations_ok": citations_ok,
        "unknown_citations": unknown_ids,
        "automatic_pass": automatic_pass,
        "elapsed_seconds": round(perf_counter() - started, 3),
        "manual_review_criteria": case["review"],
        "result": result,
    })

    print("实际状态：", result.get("status"))
    print("自动检查：", "通过" if automatic_pass else "未通过")
    print("回答：", answer)
    print("人工复核要点：", case["review"])

output_path = project_dir / f"e2e_report_{cases_path.stem}.json"
output_path.write_text(
    json.dumps(reports, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

passed = sum(report["automatic_pass"] for report in reports)
print(f"\n自动检查通过：{passed}/{len(cases)}")
print(f"完整结果已保存到 {output_path.name}，仍需人工核对回答。")