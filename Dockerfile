FROM python:3.13-slim AS builder

WORKDIR /app

# uv 설치 및 의존성 동기화
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 패키지 명세 파일 복사
COPY pyproject.toml uv.lock ./

# 가상환경 의존성 설치
RUN uv sync --frozen --no-install-project --no-dev

# 런타임 단계
FROM python:3.13-slim

WORKDIR /app

# 보안을 위한 non-root 사용자 생성
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 가상환경 복사
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# 애플리케이션 소스 코드 복사
COPY app ./app

# 소유권 변경 및 유저 전환
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 컨테이너 헬스체크 (추가 바이너리 없이 내장 urllib 사용)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
