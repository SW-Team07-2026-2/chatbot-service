# INDIG 챗봇 - 배포 · Gateway 연결

> `app/` 코드는 그대로이고, 배포에 필요한 파일만 추가했습니다 (`gateway_entry.py` · `Dockerfile` · `.github/workflows/ci.yml`).

## 1. Gateway 경로

운영에서는 Gateway(`https://api.indig.co.kr`)가 `/api/chatbot/**` 를 **경로 변경 없이** 챗봇으로 넘깁니다.
`gateway_entry.py` 가 기존 `/chat` 라우터를 `/api/chatbot` 아래에도 열어 줍니다. 핸들러 · 인증은 기존과 같습니다.

| 로컬 (`uvicorn app.main:app`) | 운영 (Gateway 경유) |
| :--- | :--- |
| `GET /chat` | `GET /api/chatbot/chat` |
| `POST /chat/upload` | `POST /api/chatbot/chat/upload` |
| `POST /chat` | `POST /api/chatbot/chat` |

Gateway 를 지날 때 지켜야 하는 것:

- 로그인 쿠키(`AT`)가 있어야 합니다 (`credentials: 'include'`). 없으면 Gateway 가 401 을 냅니다.
- `POST` 는 `XSRF-TOKEN` 쿠키 값을 `X-XSRF-TOKEN` 헤더로 같이 보내야 합니다. 없으면 403 입니다.
- `/api/chatbot/**` 전체가 사용자당 **분당 5회** 한도를 같이 씁니다 (기록 조회 · 업로드 · 질문 합산). 넘으면 429 입니다.
- 응답은 **30초**에서 끊깁니다. 요청 본문은 **1 MiB** 까지입니다 (더 큰 업로드는 413).

## 2. 운영 환경 변수

값은 manifests 레포 `services/chatbot/` 에 있습니다. 키 2개는 클러스터 Secret `app-secrets` 에서 들어옵니다.

| 변수 | 운영 값 |
| :--- | :--- |
| `ENVIRONMENT` | `prod` |
| `CORS_ORIGINS` | `https://indig.co.kr` |
| `REDIS_URL` | `redis://redis-chatbot:6379/0` (챗봇 전용 Redis) |
| `BACKEND_URL` | `https://api.indig.co.kr` |
| `LLM_MODEL` | `google:gemini-flash-lite-latest` |
| `GEMINI_API_KEY` | Secret `app-secrets` 의 `chatbot.gemini.api-key` |
| `EXA_API_KEY` | Secret `app-secrets` 의 `chatbot.exa.api-key` |

## 3. 배포 흐름

1. 작업 브랜치 → `develop` PR (앱 기동 확인만 돕니다 — `import gateway_entry`)
2. `develop` → `main` 머지 → CI 가 arm64 이미지를 ECR `indig-chatbot:<커밋 SHA>` 로 올리고, manifests 레포의 이미지 태그를 고쳐 커밋합니다
3. ArgoCD 가 manifests `main` 을 읽어 클러스터에 반영합니다

컨테이너는 UID 10001 · 루트 파일시스템 읽기 전용으로 돕니다. 파일은 `/tmp` 에만 쓸 수 있습니다.

로컬에서 이미지 확인:

```sh
docker build -t indig-chatbot:local .
docker run --rm -p 8000:8000 --env-file .env indig-chatbot:local
```
