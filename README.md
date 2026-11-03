# Indie Band Platform - AI Agent Service (`chat-ai`)

인디 밴드 검색, 공연 일정 조회, 노치(Notch) UI 실시간 인터랙션 및 승인 카드를 제공하는 AI 에이전트 마이크로서비스입니다.

## 1. 아키텍처 개요

- **프레임워크**: Python 3.13 + FastAPI + Uvicorn
- **AI / LLM**: PydanticAI 기반 (제미나이, 클로드)
- **통신 프로토콜**: Server-Sent Events (SSE) 기반 실시간 스트리밍
- **에이전트 패턴**: ReAct (도구 판단 → 실행 → 결과 취합 → 최종 응답)
- **인증 및 보안**: JWT 패스스루, Human-in-the-loop 승인 카드 (환각 방지)
- **메모리 관리**: Redis (채팅 기록 및 상태 관리)

## 사용법


`uv run uvicorn app.main:app --reload --port 8000`