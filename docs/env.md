# INDIG 챗봇 - 환경 변수

> !!! 모든 값 필수 !!!

## 빠른 시작 (복사)

```ini
# 1. 서버 기본 설정
ENVIRONMENT=prod # 실행 환경: dev(개발) 또는 prod(운영)
CORS_ORIGINS=http://... # 프론트엔드 CORS 허용 주소 (쉼표로 구분)
REDIS_URL=redis://... # Redis 연결 주소
BACKEND_URL=http://... # 메인 백엔드(Gateway) 주소

# 2. AI 및 외부 API 설정
LLM_MODEL=google:gemini-flash-lite-latest # 사용할 LLM 모델 식별자 (PydanticAI 형식)
GEMINI_API_KEY=...당신의_키 # Google Gemini API 키 
EXA_API_KEY=...당신의_키 # Exa AI 실시간 웹 검색 API 키 
```

---

## 환경 변수 상세 설명

| 변수명 | 예시값 | 설명 |
| :--- | :--- | :--- |
| **`ENVIRONMENT`** | `dev` | 서버의 구동 환경입니다. 값에 `dev`가 포함되면 Swagger 문서(`/docs`, `/redoc`)가 활성화되며, `prod`가 포함되면 운영 보안을 위해 비활성화됩니다. |
| **`CORS_ORIGINS`** | `http://localhost:5173` | AI 챗봇과 통신을 허용할 프론트엔드 도메인(Origin) 목록입니다. 여러 개일 경우 쉼표(`,`)로 구분합니다. |
| **`REDIS_URL`** | `redis://localhost:6379/0` | 유저 대화 히스토리 및 장기 기억을 저장하는 Redis 연결 문자열입니다. |
| **`BACKEND_URL`** | `http://localhost:8080` | 메인 백엔드(Gateway) 주소입니다. 챗봇 요청이 들어올 때 유저의 토큰(`AT`)이 진짜 로그인 상태인지 `GET /api/auth/me`를 호출해 검증합니다. |
| **`LLM_MODEL`** |`google:gemini-flash-lite-latest` | 대화 생성에 사용할 LLM 모델입니다. 기본값으로 가볍고 빠른 Gemini Flash-Lite 모델을 권장합니다. |
| **`GEMINI_API_KEY`** |`AIzaSy...` | 구글 AI Gemini 호출에 사용되는 API 키입니다. 텍스트 대화뿐만 아니라 유저가 올린 첨부 파일(이미지/PDF/음원/동영상) 분석에도 함께 쓰입니다. |
| **`EXA_API_KEY`** |  `...` | AI가 최신 인디 밴드 공연 정보나 아티스트 뉴스를 인터넷에서 실시간 검색할 때 사용하는 검색 엔진 API 키입니다. |