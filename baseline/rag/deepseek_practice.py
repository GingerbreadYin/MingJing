import json

from rag_service import ask_rag


question = input("请输入你的问题：")
result = ask_rag(question)

print("\n处理结果：")
print(json.dumps(result, ensure_ascii=False, indent=2))