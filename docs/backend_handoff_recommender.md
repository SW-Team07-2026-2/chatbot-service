# 백엔드 연동 요청서 — 개인화 추천 · 시맨틱 검색 (`/api/recommender`)

> **작성일**: 2026-11-27 · **대상 레포**: `chatbot-service` (PR #10 `feat/recommender-and-chatbot-improvements`)  
> **기준 백엔드 문서**: `v2_API_명세서.md` (2026-11-26)  
> **요약**: 기존 `chatbot-service` 컨테이너(포트 `8000`)에 개인화 추천 및 시맨틱 검색 모듈(`/api/recommender/**`)을 추가했습니다. DB(PostgreSQL) 직접 연결은 없으며, **Gateway 라우팅 + `/ingest/members/**` HMAC 키 발급 + 환경변수 3종 추가**만 해주시면 바로 운영 연동됩니다.

---

## 1. 백엔드 / 인프라 요청 사항 체크리스트

### ① [필수] Gateway 라우팅 & 인증 · 한도 설정 (`/api/recommender/**`)
기존 `/api/chatbot/**`는 **분당 5회 + 전원 로그인 필수**로 묶여 있어 추천 API에 그대로 쓸 수 없습니다.  
`/api/recommender/**`를 **경로 변경 없이** `chatbot-service:8000`으로 넘기되, 아래처럼 권한과 한도를 분리해 주세요.

| 경로 | 메서드 | 인증 (Gateway) | CSRF | 권장 호출 한도 | 응답 타임아웃 | 비고 |
|---|---|---|---|---|---|---|
| `/api/recommender/status` | `GET` | **공개** | 불필요 | 분당 60회 | 5초 | 준비 상태 헬스 확인 |
| `/api/recommender/popular/{tab}` | `GET` | **공개** | 불필요 | 분당 300회 | 10초 | 비로그인 인기 목록 (`artists`·`physical-albums`·`goods`·`concerts`) |
| `/api/recommender/similar/{targetType}/{itemId}` | `GET` | **공개 (AT 쿠키 있으면 전달)** | 불필요 | 분당 300회 | 10초 | 상세 하단 특수 추천 (로그인 시 미세 개인화) |
| `/api/recommender/onboarding/similar` | `GET` | **공개** | 불필요 | 분당 120회 | 10초 | 가입 전 밴드 고르기 체인 추천 (가입 전이라 AT 없음) |
| `/api/recommender/semantic-search` | `POST` | **공개 (AT 쿠키 있으면 전달)** | 선택/권장 | 분당 60회 | 15초 | 자연어 시맨틱 검색 (챗봇 내부에서는 HTTP 없이 직접 호출) |
| `/api/recommender/{tab}` | `GET` | **로그인 (`AT`)** | 불필요 | 분당 300회 | 10초 | 개인화 전체 추천 (새로고침 잦음 · 실제 응답 <20ms) |
| `/api/recommender/feedback` | `POST` | **로그인 (`AT`)** | **필수 (`X-XSRF-TOKEN`)** | 분당 60회 | 5초 | 실시간 행동·관심없음 피드백 (`/api/members/me/metrics`와 동일 수준) |
| `/api/recommender/admin/**` | `GET`/`POST`/`DELETE` | **외부 차단 (403/404)** | — | — | 30초 | **클러스터 내부 / Studio 터널 전용** (`X-Admin-Secret` 검증) |

> ⚠️ **중요**: `/api/recommender/admin/**`는 외부 인터넷(Cloudflare/Gateway)에서 접근할 수 없도록 막아 주세요. 관리자(Studio) 또는 클러스터 내부에서만 `chatbot-service:8000`으로 직접 호출합니다.

---

### ② [필수] `/ingest/members/**` HMAC 키 발급 & K8s Secret 추가
`v2_API_명세서.md` 9장에 미발급 상태로 적혀 있던 회원 데이터 키(`ingest.member-keys.*`)를 발급해 `chatbot-service` 환경변수(Secret `app-secrets`)에 넣어 주세요.

| 환경변수명 | 필수 여부 | 용도 | 비고 |
|---|---|---|---|
| `INGEST_MEMBER_KEY_ID` | 운영 권장 (미설정 시 Redis 단독 폴백) | Gateway `ingest.member-keys.*` 키 ID | `GET/PUT /ingest/members/{userId}/embedding`, `GET .../metrics` 호출용 |
| `INGEST_MEMBER_SECRET` | 운영 권장 (미설정 시 Redis 단독 폴백) | 위 키의 HMAC-SHA256 시크릿 | Secret `app-secrets` 주입 |
| `RECOMMENDER_ADMIN_SECRET` | 운영 필수 | `/api/recommender/admin/**` `X-Admin-Secret` 검증용 | 미설정 시 운영(`prod`)에서 관리자 API `503` 차단 |
| `REC_ARTIFACT_DIR` | 선택 (기본값 `/tmp/indig_recommender`) | 로컬 디스크 스냅샷 경로 | 읽기 전용 루트 FS 호환 (`/tmp`만 사용) |

#### 🔍 확인 부탁드릴 점 (HMAC 서명 헤더명)
현재 `chatbot-service` ([app/recommender/ingest_client.py](../app/recommender/ingest_client.py))는 `v2_API_명세서.md` 9장 기준으로 아래 헤더명과 서명 문자열을 사용합니다. Gateway `IngestFilter`의 실제 헤더명과 다른 부분이 있으면 알려 주세요(바로 맞추겠습니다).
- **헤더**: `X-Ingest-Key-Id`, `X-Ingest-Timestamp`, `X-Ingest-Nonce`, `X-Ingest-Signature`
- **서명 문자열**: `METHOD\nPATH+QUERY\nTIMESTAMP\nNONCE\nSHA256(BODY_HEX)`

---

### ③ [확인] `/ingest/members/**` 요청 · 응답 포맷 확인
`chatbot-service`가 백엔드와 주고받는 JSON 모양입니다.

1. **`PUT /ingest/members/{userId}/embedding`** (15분 주기 Write-Back 백업 · `TEXT` 저장 전제)
   ```json
   {
     "vector": [0.12, -0.04, "... (128d float)"],
     "artist_affinity": { "1": 4.5, "7": 2.1 },
     "genre_affinity": { "INDIE_ROCK": 3.5 },
     "preferred_regions": { "SEOUL": 2.0 },
     "impression_counts": { "CONCERT:201": 2 },
     "revisit_counts": { "CONCERT:201": 3 },
     "purchased_items": ["CONCERT:9"],
     "disliked_items": [],
     "seen_fresh_items": ["PHYSICAL_ALBUM:11"],
     "updated_at": 1764200000.0
   }
   ```
   - `GET /ingest/members/{userId}/embedding` 응답은 `result`가 ① 위 JSON 객체 그대로, ② JSON 문자열(`str`), ③ `{"embedding": ...}` 형태 모두 자동 파싱되도록 구현해 두었습니다.
   - 만약 `PUT` 요청 본문을 `{"embedding": "<json string>"}`처럼 감싸서 보내야 한다면 말씀해 주세요.

2. **`GET /ingest/members/{userId}/metrics?limit=100`** (Redis/임베딩 둘 다 없을 때 초기 취향 복구용)
   - `result.items` 또는 `result.events` 또는 `result` 배열의 각 항목에서 아래 4개 필드를 읽습니다:
     `targetType`, `targetId`, `metricType`, `metricValue` (v2 명세서 2-6 `MetricEvent` 기준).

---

### ④ [선택 · 권장] Studio(관리자) 연동 & 인프라 리소스 안내

#### A. 신규 아티스트/앨범/상품/공연 등록 시 즉시 반영 (선택)
- `chatbot-service`는 **1시간마다** 백엔드 공개 API(`/api/catalog/**`, `/api/concerts`)를 `embed_hash` 증분 동기화합니다(바뀐 항목만 임베딩).
- 관리자가 Studio에서 승인/등록한 직후 **1초 만에** 추천·검색에 반영하고 싶다면, 승인 처리 후 비동기로 아래 내부 API를 1회 호출해 주세요(실패해도 다음 주기 동기화로 복구됨):
  - 등록/수정: `POST http://chatbot-service:8000/api/recommender/admin/items/{targetType}/{itemId}/upsert` (`Header: X-Admin-Secret`)
  - 삭제/비공개: `DELETE http://chatbot-service:8000/api/recommender/admin/items/{targetType}/{itemId}` (`Header: X-Admin-Secret`)
  - `targetType`: `ARTIST` | `PHYSICAL_ALBUM` | `GOODS` | `CONCERT`

#### B. K8s 배포 · 메모리 · 멀티 Pod 호환성
- **컨테이너 경량 유지**: 운영 의존성에는 `torch` / `faiss-cpu`를 넣지 않고 **`numpy`만 추가**했습니다(`t4g` arm64 / `amd64` 멀티아치 100% 호환, 읽기 전용 루트 FS `/tmp` 호환).
- **멀티 Pod(Replica ≥ 2) 지원**: 인덱스 스냅샷을 기존 `redis-chatbot`의 `rec:index:snapshot` / `rec:index:version` 키에 저장·공유합니다. Pod 재시작/증설 시 백엔드 조회·임베딩 없이 Redis 스냅샷을 즉시 로드합니다.
- **Redis 메모리 확인**: 기존 챗봇 키(`{user}:chat`, `{user}:ltm`)와 완전 분리되며, 아이템 1,000건 기준 스냅샷 약 2~3 MB + 유저 상태/캐시 포함해 **`redis-chatbot`에 최소 30~50 MB 여유**만 있으면 충분합니다.
- **기존 백엔드 공개 API 호출량**: 1시간에 1회 페이징 조회(`size=50`)만 수행하므로 Gateway/서비스 부하 영향은 거의 없습니다.

---

## 2. 추천 API 요약 (프론트 · Gateway 공유용)

### 공통 목록 파라미터 (`GET /api/recommender/{tab}`, `GET /api/recommender/popular/{tab}`)
- `tab`: `artists` | `physical-albums` | `goods` | `concerts`
- 페이징: `page` (0부터, 기본 0) · `size` (기본 20, 최대 50)
- **v2 칩 필터 100% 호환** (같은 칩 OR, 다른 칩 AND, 콤마 또는 반복 파라미터 모두 지원):
  - 아티스트: `genre`, `gender`, `composition`
  - 피지컬 앨범: `format`, `minPrice`, `maxPrice`(≥150000 상한 없음), `excludeSoldOut`
  - 굿즈: `category`, `minPrice`, `maxPrice`, `excludeSoldOut`
  - 공연: `saleStatus`, `type`, `region`, `genre`

### 응답 형식 (`PageResult` 호환)
```json
{
  "items": [
    {
      "item_id": 201,
      "target_type": "CONCERT",
      "score": 0.8421,
      "reason_type": "FAVORITE_ARTIST",
      "reason_text": "자주 본 한로로 관련 추천",
      "data": { "...기존 v2 목록 항목(ArtistItem / PhysicalItem / GoodsItem / ConcertSummary)..." }
    }
  ],
  "page": 0,
  "size": 20,
  "hasNext": true,
  "degraded": false
}
```
> 인덱스 초기화 전이거나 장애 시에도 500 대신 `200 OK` + `{ "items": [], "degraded": true }`를 반환합니다(프론트는 기존 전체 목록 API로 안전하게 폴백 가능).

### 관리자 조작 API 목록 (`/api/recommender/admin/**`, 내부 전용 + `X-Admin-Secret`)
| Method | Endpoint | 설명 |
|---|---|---|
| `GET` | `/api/recommender/admin/status` | 카테고리별 인덱싱 건수, 스냅샷 버전, 마지막 동기화/Write-Back 시각 |
| `POST` | `/api/recommender/admin/sync` | 전체 카탈로그·공연 즉시 증분 재동기화 |
| `POST` | `/api/recommender/admin/items/{targetType}/{itemId}/upsert` | 단일 아이템 1건 즉시 임베딩·인덱스 반영 및 전 Pod 전파 |
| `DELETE` | `/api/recommender/admin/items/{targetType}/{itemId}` | 단일 아이템 즉시 제거 및 전 Pod 전파 |
| `POST` | `/api/recommender/admin/reload-artifact` | 오프라인 PyTorch 재학습 아티팩트 무중단 핫 리로드 및 전 Pod 전파 |
| `POST` | `/api/recommender/admin/flush-writeback` | Redis dirty 유저 벡터 즉시 백엔드 DB 백업 |
| `POST` | `/api/recommender/admin/cache/clear` | 인기 목록 Redis 캐시 즉시 초기화 |
