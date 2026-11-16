# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
RUN pip install --no-cache-dir uv==0.11.1 \
    && useradd --uid 10001 --create-home ragopt

WORKDIR /app
COPY --from=deerflow backend/ /app/upstream/backend/
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /app/upstream/backend && uv sync --locked --no-dev --extra redis

COPY baseline/rag/requirements-rag.txt /tmp/requirements-rag.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,from=cpu_wheels,target=/opt/cpu-wheels \
    uv venv /app/.venv-retrieval \
    && torch_package='torch==2.14.1+cpu' \
    && local_wheel='/opt/cpu-wheels/torch-2.14.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl' \
    && if [ -f "$local_wheel" ]; then \
         echo "5a6363570c753812540a05eb82380e329469cbe668643e88111414c12627711f  $local_wheel" | sha256sum -c - \
         && torch_package="$local_wheel"; \
       fi \
    && uv pip install --python /app/.venv-retrieval/bin/python "$torch_package" --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --python /app/.venv-retrieval/bin/python -r /tmp/requirements-rag.txt \
    && /app/.venv-retrieval/bin/python -c 'import torch, sentence_transformers; from langgraph.types import Send' \
    && touch /app/.venv-retrieval/READY

COPY --chown=ragopt:ragopt optimizer/ ./optimizer/
COPY --chown=ragopt:ragopt scripts/ ./scripts/
COPY --chown=ragopt:ragopt data/ ./data/
COPY --chown=ragopt:ragopt baseline/ ./baseline/
RUN mkdir -p /app/runtime /app/experiments /app/retained /app/.cache/models \
    && chown ragopt:ragopt /app /app/runtime /app/experiments /app/retained /app/.cache/models
USER ragopt
CMD ["/app/upstream/backend/.venv/bin/python", "/app/scripts/compose_gateway.py"]
