import re
from typing import Literal
from exa_py import AsyncExa
from pydantic_ai import RunContext

from .agent import skill
from .memory import add_user_memory
from app.config import setting



@skill("팝업을 띄우고 있어요... 🎫")
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
async def update_long_term_memory(ctx: RunContext[str], information: str) -> str:
    """유저의 음악 취향이나 중요한 개인 정보를 장기 기억으로 새로 저장한다.

    유저가 좋아하는 밴드, 선호하는 음악 장르, 자주 가는 지역(예: 홍대) 등 
    앞으로의 대화와 맞춤형 추천에 도움이 되는 핵심 취향을 밝혔을 때만 호출하라.
    단순 인사나 일상 대화, 일회성 질문에는 절대 호출하지 마라.
    """
    user_uuid = ctx.deps
    print(f"[장기 기억 저장] 유저({user_uuid}): {information}")
    await add_user_memory(user_uuid, information)
    return f"유저의 새로운 취향({information})을 장기 기억에 성공적으로 보관했습니다. 유저에게는 이를 의식하지 않고 자연스럽게 대화를 이어가세요."


_exa = AsyncExa(setting.EXA_API_KEY)

@skill("인터넷 검색 중")
async def web_search(query: str) -> dict:
    """query를 이용하여 인터넷 검색한다."""
    try:
         
        result = await _exa.search(
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
        confidence = result.output.grounding[0].confidence

    except:
        return {
            "message_for_ai": "웹 검색 도중 에러가 발생하였다. 몇번 더 시도해보고 그래도 안되면 현재 웹 검색 기능에 문제가 있음을 알리고 기존 지식을 활용하여 친절히 답하라.",
        }
    
    return {
        "result": output_content,
        "confidence": confidence
    }
