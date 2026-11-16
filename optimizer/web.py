"""Small standalone development host; production entry uses upstream Gateway."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from .service import get_service
from .routes import build_router


@asynccontextmanager
async def lifespan(app):
    yield
    get_service().close()


app=FastAPI(title='RAG 检索诊断与优化',lifespan=lifespan)
app.include_router(build_router(get_service()))


@app.get('/')
def index():
    return RedirectResponse('/api/rag-optimizer/ui')


@app.get('/health')
def health():
    return {'status':'ok','mode':'development-host'}
