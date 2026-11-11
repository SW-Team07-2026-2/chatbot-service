from app.config import logger
from app.server.redis import get_client

# TODO: 나중에 백앤드 쪽 장기기억 api 열리면 업그레이드

MEMORY_TTL: int = 60 * 60 * 24 * 30  # 장기 기억 TTL (30일)
MAX_MEMORY_ITEMS: int = 15  # 최대 보관 기억 갯수

# 초기 유저용 기본 데모 기억 (첫 대화 시 자동 장착)
DEFAULT_INITIAL_MEMORIES: list[str] = [
    "강렬한 하드 록과 헤비메탈 사운드를 가장 선호함 (기록: 3개월 전)",
    "홍대 인근 라이브클럽 공연장을 자주 관람함 (기록: 1개월 전)",
    "요즘은 차분하고 감성적인 재즈 팝과 어쿠스틱 음악에 푹 빠졌다고 언급함 (기록: 최근, 최우선 반영)",
]


def make_memory_key(user_uuid: str) -> str:
    return f"{{{user_uuid.strip()}}}:memory"


async def get_user_memories(user_uuid: str) -> list[str]:
    """Redis에서 유저의 장기 기억 목록 조회 (기록 없으면 기본 취향 데이터 초기화 후 반환)"""
    if not user_uuid:
        return []
    try:
        r = get_client()
        key = make_memory_key(user_uuid)
        items = await r.lrange(key, 0, -1)
        if not items:
            # 초기 데모 기억 등록
            async with r.pipeline(transaction=True) as p:
                for mem in DEFAULT_INITIAL_MEMORIES:
                    p.rpush(key, mem)
                p.expire(key, MEMORY_TTL)
                await p.execute()
            return DEFAULT_INITIAL_MEMORIES.copy()

        return items
    except Exception as e:
        logger.warning(f"[장기 기억 조회 실패] ({user_uuid}): {e}")
        return []


async def add_user_memory(user_uuid: str, information: str) -> None:
    """새로운 개인 취향/정보를 유저의 장기 기억에 추가 저장"""
    if not user_uuid or not information or not information.strip():
        return
    try:
        r = get_client()
        key = make_memory_key(user_uuid)
        async with r.pipeline(transaction=True) as p:
            p.rpush(key, information.strip())
            p.ltrim(key, -MAX_MEMORY_ITEMS, -1)
            p.expire(key, MEMORY_TTL)
            await p.execute()
        logger.info(f"[장기 기억 저장 완료] ({user_uuid}): {information.strip()}")
    except Exception as e:
        logger.error(f"[장기 기억 저장 실패] ({user_uuid}): {e}")


def format_memory_for_system_prompt(memories: list[str]) -> str:
    """시스템 프롬프트에 주입할 XML 포맷 텍스트 생성"""
    if not memories:
        return ""
    lines = "\n".join(f"- {m}" for m in memories)
    return f"""
<user_memory>
[사용자의 이전 취향 및 기억 정보]
{lines}

지침:
1. 위 내용은 사용자와의 이전 대화에서 수집된 실제 취향, 선호 장르, 관심사입니다.
2. 각 항목의 시간 흐름을 살피고, 가장 최근에 언급된 취향을 최우선으로 반영하여 맞춤형으로 대화하세요.
3. 시스템 내부 태그나 '기억에 적혀있다'는 사실을 직접 언급하지 말고, 원래 유저를 잘 알고 있던 것처럼 자연스럽게 응대하세요.
</user_memory>
"""

