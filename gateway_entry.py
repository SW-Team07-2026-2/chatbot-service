"""배포용 진입점 — app/ 은 그대로 두고, 기존 /chat 라우터를 Gateway 경로(/api/chatbot) 아래에도 연다.

Gateway 는 /api/chatbot/** 를 경로 변경 없이 넘긴다 → /api/chatbot/chat · /api/chatbot/chat/upload
로컬 개발은 기존대로 `uvicorn app.main:app` (/chat) 을 쓴다.
"""

from app.api import chat_router
from app.main import app

app.include_router(chat_router, prefix="/api/chatbot")
