# INDIG 챗봇 - API 명세서

> 모든 `/chat` 요청은 로그인 필수 (`AT` 쿠키 또는 `Authorization: Bearer <token>` 헤더)  
> 비로그인 또는 토큰 만료 시 `401 Unauthorized` 반환

## 엔드포인트 요약

| Method | Endpoint | 통신 방식 | 설명 |
| :--- | :--- | :---: | :--- |
| **`GET`** | `/chat` | REST (JSON) | 대화 기록 조회 (기록 없을 시 첫인사 자동 생성) |
| **`POST`** | `/chat/upload` | Multipart | 파일(이미지/PDF/음원 등) Gemini 임시 업로드 (48시간 보관) |
| **`POST`** | `/chat` | **SSE** | 사용자 질문/첨부파일 전송 및 실시간 AI 답변 스트리밍 |
| **`GET`** | `/health/liveness` | REST (JSON) | 서버 생존 상태 확인 |
| **`GET`** | `/health/readiness` | REST (JSON) | Redis 연결 및 서비스 준비 상태 확인 |

---

## 1. GET /chat (대화 기록 조회)

- 최근 20개(10턴)의 대화 기록을 반환합니다.
- 대화 기록이 없으면 딘디의 첫인사를 Redis에 자동 생성 후 반환합니다.

### Response (`200 OK`)
```json
{
  "count": 1,
  "total_count": 1,
  "has_truncated": false,
  "messages": [
    {
      "id": "hist-bot-1",
      "role": "bot",
      "text": "안녕하세요! 인디 밴드 플랫폼 **INDIG**의 AI 챗봇 **딘디**입니다.",
      "time": "오후 3:20"
    }
  ]
}
```

---

## 2. POST /chat/upload (파일 업로드)

- Google Gemini File API에 업로드되며 48시간 동안 무료 임시 보관됩니다.
- 응답받은 객체를 그대로 `POST /chat`의 `attachments`에 넣어 전송합니다.

### Request (`multipart/form-data`)
- `file`: 업로드할 파일 (최대 20MB)
  - 지원 포맷: 이미지(PNG/JPG/WEBP 등), 문서(PDF/TXT/MD/CSV/JSON), 오디오(MP3/WAV/M4A), 비디오(MP4)

### Response (`200 OK`)
```json
{
  "file_uri": "https://generativelanguage.googleapis.com/v1beta/files/...",
  "file_name": "poster.png",
  "media_type": "image/png",
  "url": "https://generativelanguage.googleapis.com/v1beta/files/...",
  "name": "poster.png",
  "type": "image/png"
}
```

---

## 3. POST /chat (실시간 답변 스트리밍)

- 질문 및 첨부파일을 전송하고 AI 답변을 실시간 SSE(Server-Sent Events)로 수신합니다.
- 이전 턴에서 업로드한 파일도 멀티턴으로 기억하여 답변에 반영합니다.

### Request (`application/json`)
```json
{
  "message": "이 포스터에 적힌 공연 날짜가 언제야?",
  "attachments": [
    {
      "file_uri": "https://generativelanguage.googleapis.com/v1beta/files/...",
      "file_name": "poster.png",
      "media_type": "image/png"
    }
  ]
}
```
> 첨부파일이 없을 경우 `attachments` 필드는 생략하거나 빈 배열(`[]`)로 보냅니다.

### Response (`text/event-stream`)

| 이벤트 (`event`) | 데이터 (`data`) 예시 | 설명 |
| :--- | :--- | :--- |
| **`status`** | `{"status": "thinking", "message": "답변을 생각하고 있어요..."}`<br>`{"status": "tool_calling", "tool": "search_web", "message": "..."}` | AI 사고 상태 또는 백엔드 도구(스킬) 실행 안내 |
| **`text_delta`** | `{"content": "공연은 11월 "}` | 생성 중인 AI 답변 글자 조각 |
| **`action`** | `{"action": "test_popup", ...}` | UI 액션 신호 (말풍선 하단 버튼/모달 렌더링용) |
| **`done`** | `{"status": "completed"}` | 답변 스트리밍 완료 |
| **`error`** | `{"error": "답변 생성 중 문제가 발생했습니다.", "code": "..."}` | 스트리밍 처리 오류 |

---

## 4. 헬스 체크

### GET /health/liveness
- 서버 프로세스 동작 확인
```json
{
  "status": "UP"
}
```

### GET /health/readiness
- Redis 연결 정상 시 `200 OK`, 연결 불가 시 `503 Service Unavailable`
```json
{
  "status": "UP",
  "components": {
    "redis": "UP"
  }
}
```