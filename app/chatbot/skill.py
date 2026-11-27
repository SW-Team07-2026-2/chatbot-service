import asyncio
import re
from typing import Literal
from exa_py import AsyncExa
from google.genai import types

from .agent import MODEL_NAME, skill
from .history import create_uploaded_file_part, get_raw_chat_history
from .memory import add_user_memory
from .uploader import get_genai_client
from app.config import constant, logger, setting
from app.recommender import service as recommender_service
from app.server.auth import get_current_user_auth, get_current_user_uuid

_background_tasks: set[asyncio.Task] = set()


async def drain_background_tasks(timeout: float = 5.0) -> None:
    """서버 종료 시 진행 중인 백그라운드 작업 완료 대기"""
    if not _background_tasks:
        return
    pending = list(_background_tasks)
    logger.info(f"백그라운드 작업 {len(pending)}건 완료 대기 중...")
    _, still_pending = await asyncio.wait(pending, timeout=timeout)
    for task in still_pending:
        task.cancel()



@skill("팝업을 띄우고 있어요")
def open_test_popup() -> dict:
    """사용자에게 테스트 팝업을 띄워줄 때 호출하라."""
    return {
        "action": "test_popup",
        "message_for_ai": f"ai가 하는 말 아래쪽에 팝업이 준비됐다. 사용자에게 자연스럽게 확인을 유도하며 친절하게 안내해라.",
    }

@skill("BGM 제어 중")
def control_bgm(bgm_action: Literal["stop", "start"]):
    """사용자 브라우저에서 나오는 BGM을 제어한다."""
    return {
        "action": f"{bgm_action}_bgm",
        "message_for_ai": f"BGM을 {'정지' if bgm_action == 'stop' else '재생'}하였다. 사용자에게 자연스럽게 안내해라.",

    }
    
    

# 프론트
# ===========================================================================================================
# 백엔드


@skill("새롭게 기억하는 중")
async def update_long_term_memory(information: str) -> str:
    """유저의 음악 취향이나 중요한 개인 정보를 장기 기억으로 새로 저장한다.

    유저가 좋아하는 밴드, 선호하는 음악 장르, 자주 가는 지역(예: 홍대) 등 
    앞으로의 대화와 맞춤형 추천에 도움이 되는 핵심 취향을 밝혔을 때만 호출하라.
    단순 인사나 일상 대화, 일회성 질문에는 절대 호출하지 마라.
    저장할 내용은 핵심만 간결하게 100자 이내(최대 200자)로 요약해 전달하라.
    """
    user_uuid = get_current_user_uuid()
    auth = get_current_user_auth()
    task = asyncio.create_task(add_user_memory(information, user_uuid=user_uuid, auth=auth))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return "유저의 새로운 취향을 장기 기억에 성공적으로 보관했습니다. 유저에게는 이를 의식하지 않고 자연스럽게 대화를 이어가세요."


_exa: AsyncExa | None = None


def get_exa_client() -> AsyncExa:
    """Exa 웹 검색 클라이언트 지연 초기화"""
    global _exa
    if _exa is None:
        _exa = AsyncExa(setting.EXA_API_KEY)
    return _exa


@skill("인터넷 검색 중")
async def web_search(query: str) -> dict:
    """query를 이용하여 인터넷 검색한다."""
    try:

        result = await get_exa_client().search(
            query,
            num_results=5,
            output_schema={"type": "text"},
            stream=False,
            system_prompt="한국 연예계(가수, 아이돌, 밴드, 노래) 관련 주제에 대해 중점적으로 답한다. 공식 발표나 신뢰할 수 있는 매체를 우선하고, 루머나 확인되지 않은 내용은 사실처럼 쓰지 않는다. 날짜가 중요한 정보는 기준 시점을 함께 밝힌다.",
            type="auto",
            user_location="KR",
            contents={
                "text": {
                    "max_characters": 300,
                    "verbosity": "full",
                },
                "highlights": {
                    "max_characters": 300,
                },
                "max_age_hours": 168,
            },
        )

        output_content = re.compile(r"\s*\[[0-9]\]").sub("", result.output.content).strip()
        grounding = getattr(result.output, "grounding", None) or []
        confidence = grounding[0].confidence if grounding else None

    except Exception as e:
        logger.warning(f"[웹 검색 오류] query={query}: {e}")
        return {
            "message_for_ai": "웹 검색 도중 에러가 발생하였다. 몇번 더 시도해보고 그래도 안되면 현재 웹 검색 기능에 문제가 있음을 알리고 기존 지식을 활용하여 친절히 답하라.",
        }

    response: dict = {"result": output_content}
    if confidence is not None:
        response["confidence"] = confidence
    return response


@skill("이전 첨부파일을 다시 확인하는 중")
async def read_previous_attachment(question: str, file_name: str = "") -> dict:
    """이전 대화에서 사용자가 올렸던 첨부파일의 세부 내용을 다시 읽어 확인한다.

    대화 기록에 `[이전 첨부파일: ...]`로만 남아 있고 원본 파일이 현재 문맥에 없는 경우,
    사용자가 해당 파일의 내용을 다시 묻거나 세부 확인을 요청할 때만 호출하라.
    file_name은 특정 파일명이 언급된 경우 입력하고, 생략하면 가장 최근 첨부파일을 확인한다.
    """
    user_uuid = get_current_user_uuid().strip()
    if not user_uuid:
        return {"error": "사용자 정보를 확인할 수 없어 첨부파일을 조회하지 못했습니다."}

    try:
        raw_history = await get_raw_chat_history(user_uuid)
    except Exception as e:
        logger.warning(f"[이전 첨부파일 조회 실패] ({user_uuid}): {e}")
        return {"error": "대화 기록을 불러오지 못해 이전 첨부파일을 확인할 수 없습니다."}

    target_att: dict | None = None
    target_name_query = str(file_name or "").strip().lower()

    for item in reversed(raw_history):
        if item.get("role") != "user":
            continue
        for att in reversed(item.get("attachments") or []):
            if not isinstance(att, dict):
                continue
            att_name = str(att.get("file_name") or att.get("name") or "").strip()
            if not target_name_query or target_name_query in att_name.lower():
                target_att = att
                break
        if target_att is not None:
            break

    if target_att is None:
        return {
            "error": "현재 대화 기록에서 해당 첨부파일을 찾지 못했습니다. 사용자에게 파일을 다시 올려달라고 안내하라."
        }

    file_part = create_uploaded_file_part(target_att)
    if file_part is None:
        return {
            "error": "유효하지 않은 첨부파일 경로입니다. 사용자에게 파일을 다시 올려달라고 안내하라."
        }

    resolved_name = str(target_att.get("file_name") or target_att.get("name") or "첨부파일")
    prompt = (
        f"첨부된 파일('{resolved_name}')의 내용을 바탕으로 아래 질문에 정확하고 간결하게 답해라.\n"
        f"파일에 없는 내용은 추측하지 말고 확인되지 않는다고 밝혀라.\n\n"
        f"질문: {question}"
    )

    try:
        res = await get_genai_client().aio.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=prompt), file_part],
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1024,
                http_options=types.HttpOptions(
                    timeout=constant.LLM_TIMEOUT_MS,
                    retry_options=types.HttpRetryOptions(attempts=constant.LLM_RETRY_ATTEMPTS),
                ),
            ),
        )
        analysis = (res.text or "").strip()
        if not analysis:
            return {"error": "첨부파일에서 답변할 내용을 추출하지 못했습니다."}
        return {"file_name": resolved_name, "analysis": analysis}
    except Exception as e:
        logger.warning(f"[이전 첨부파일 재분석 실패] ({user_uuid}, {resolved_name}): {e}")
        return {
            "error": "첨부파일 보관 기간이 만료되었거나 읽을 수 없습니다. 사용자에게 파일을 다시 업로드해 달라고 친절히 안내하라."
        }


@skill("분위기에 맞는 인디 음악·공연을 찾고 있어요")
async def semantic_search_catalog(
    query: str,
    categories: list[Literal["artists", "physical-albums", "goods", "concerts"]] | None = None,
    size: int = 5,
) -> dict:
    """사용자의 기분·상황·분위기·장르 자연어 키워드로 INDIG 내 아티스트, 음반, 굿즈, 공연을 시맨틱 검색한다.

    예: 사용자가 '기분 우울할 때 들을 만한 아티스트 추천해줘'라고 물으면
    query='우울, 회복, 안정, 비, 잔잔한', categories=['artists']로 호출하라.
    """
    user_uuid = get_current_user_uuid().strip() or None
    capped_size = max(1, min(int(size or 5), 10))
    res = await recommender_service.semantic_search(
        query=query,
        categories=categories,
        size=capped_size,
        user_uuid=user_uuid,
    )
    if not res.items:
        return {
            "items": [],
            "degraded": res.degraded,
            "message_for_ai": "INDIG 내부 카탈로그에서 조건에 맞는 항목을 찾지 못했다. 필요하면 웹 검색을 보조로 활용하거나 다른 키워드로 안내하라.",
        }
    return {
        "query": res.query,
        "degraded": res.degraded,
        "items": [
            {
                "item_id": it.item_id,
                "target_type": it.target_type,
                "score": it.score,
                "reason": it.reason_text,
                "data": it.data,
            }
            for it in res.items
        ],
    }



