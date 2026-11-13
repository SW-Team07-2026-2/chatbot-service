# syntax=docker/dockerfile:1
# 빌드: docker buildx build --platform linux/arm64 -t <ecr>/indig-chatbot:$(git rev-parse HEAD) --push .
# 🔴 arm64 — 운영 노드가 t4g (Graviton) 입니다. 파이썬 휠이 아키텍처별이라 빌드 단계도 대상 아키텍처에서 돕니다
FROM python:3.13-slim AS build
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.13-slim
RUN useradd -r -u 10001 app
WORKDIR /app
COPY --from=build /app/.venv ./.venv
COPY app ./app
COPY gateway_entry.py ./
# 루트 파일시스템 읽기 전용 — .pyc 를 쓰지 않습니다 (쓰기는 /tmp 만)
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
USER 10001
EXPOSE 8000
# gateway_entry — 기존 app.main:app 에 Gateway 경로(/api/chatbot)만 덧붙인 진입점
CMD ["uvicorn", "gateway_entry:app", "--host", "0.0.0.0", "--port", "8000"]
