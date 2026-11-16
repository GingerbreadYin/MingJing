from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_graph import build_rag_graph
from rag_service import make_result
from retrieval import load_retriever
import logging
from citations import InvalidAnswerError
from model_client import ModelRequestError

logger = logging.getLogger(__name__)
@asynccontextmanager
async def lifespan(app):
    print("正在加载检索资源并构建图……")
    resources = load_retriever()
    app.state.rag_graph = build_rag_graph(resources)
    print("RAG 图准备完成")
    yield


app = FastAPI(title="电商政策 RAG API", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(body: ChatRequest):
    question = body.question.strip()

    if not question:
        raise HTTPException(status_code=422, detail="问题不能为空白")

    try:
        state = app.state.rag_graph.invoke({"question": question})
    except ModelRequestError:
        logger.exception("处理问答时，模型请求失败")
        raise HTTPException(
            status_code=503,
            detail={
                "code": "MODEL_REQUEST_FAILED",
                "message": "模型服务当前不可用，本次未能完成处理。",
            },
        ) from None
    except InvalidAnswerError:
        logger.exception("模型回答未通过引用校验")
        raise HTTPException(
            status_code=502,
            detail={
                "code": "INVALID_MODEL_OUTPUT",
                "message": "本次回答未通过引用校验，暂未返回。",
            },
        ) from None
    return make_result(
        state["status"],
        state["answer"],
        state["rounds"],
        state["selected_chunks"],
        state.get("final_answer_usage"),
    )