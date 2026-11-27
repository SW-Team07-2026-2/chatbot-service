"""추천 시스템 Redis 저장소 (유저 취향 상태 · Write-Back dirty · 후보군 캐시 · 실시간 트렌드)."""

import json
from typing import Any

from app.config import constant, logger
from app.server.redis import get_client
from .schemas import RecommendCategory, UserTasteState

__all__ = [
    "make_user_taste_key",
    "make_dirty_users_key",
    "make_candidate_cache_key",
    "make_candidate_lock_key",
    "make_popular_cache_key",
    "make_fandom_trend_key",
    "make_global_trend_key",
    "make_index_snapshot_key",
    "make_index_version_key",
    "get_user_taste",
    "save_user_taste",
    "pop_dirty_users",
    "get_cached_candidates",
    "cache_candidates",
    "invalidate_user_candidates",
    "acquire_candidate_lock",
    "release_candidate_lock",
    "get_cached_popular",
    "cache_popular",
    "record_trend_event",
    "get_fandom_trends",
    "get_global_trends",
    "save_index_snapshot",
    "get_index_version",
    "load_index_snapshot",
]

ALL_CATEGORIES: tuple[RecommendCategory, ...] = (
    "artists",
    "physical-albums",
    "goods",
    "concerts",
)


def make_user_taste_key(user_uuid: str) -> str:
    return f"{{{user_uuid.strip()}}}:rec:taste"


def make_dirty_users_key() -> str:
    return "rec:dirty_users"


def make_candidate_cache_key(user_uuid: str, category: RecommendCategory) -> str:
    return f"{{{user_uuid.strip()}}}:rec:cand:{category}"


def make_candidate_lock_key(user_uuid: str, category: RecommendCategory) -> str:
    return f"{{{user_uuid.strip()}}}:rec:cand:{category}:lock"


def make_popular_cache_key(category: RecommendCategory) -> str:
    return f"rec:popular:{category}"


def make_fandom_trend_key(artist_id: int) -> str:
    return f"rec:trend:fan_of:{artist_id}"


def make_global_trend_key(category: RecommendCategory) -> str:
    return f"rec:trend:global:{category}"


def make_index_snapshot_key() -> str:
    return "rec:index:snapshot"


def make_index_version_key() -> str:
    return "rec:index:version"


async def get_user_taste(user_uuid: str) -> UserTasteState | None:
    """Redis에서 유저 취향 상태 조회"""
    try:
        raw = await get_client().get(make_user_taste_key(user_uuid))
        if not raw:
            return None
        return UserTasteState.model_validate_json(raw)
    except Exception as e:
        logger.warning(f"[추천 유저 상태 조회 실패] ({user_uuid}): {e}")
        return None


async def save_user_taste(
    user_uuid: str,
    state: UserTasteState,
    *,
    mark_dirty: bool = True,
) -> None:
    """Redis에 유저 취향 상태 저장 및 Write-Back dirty 마킹"""
    key = make_user_taste_key(user_uuid)
    payload = state.model_dump_json()
    try:
        r = get_client()
        await r.set(key, payload, ex=constant.REC_USER_TASTE_TTL)
        if mark_dirty:
            await r.sadd(make_dirty_users_key(), user_uuid.strip())
    except Exception as e:
        logger.warning(f"[추천 유저 상태 저장 실패] ({user_uuid}): {e}")


async def pop_dirty_users(limit: int = 100) -> list[str]:
    """Write-Back 대상 dirty 유저 ID 목록 조회 및 제거"""
    try:
        r = get_client()
        key = make_dirty_users_key()
        members = await r.smembers(key)
        if not members:
            return []
        batch = list(members)[:limit]
        await r.srem(key, *batch)
        return batch
    except Exception as e:
        logger.warning(f"[Write-Back dirty 유저 조회 실패]: {e}")
        return []


async def get_cached_candidates(
    user_uuid: str,
    category: RecommendCategory,
) -> list[dict[str, Any]] | None:
    """F5 새로고침 방어용 Top-50 후보군 캐시 조회"""
    try:
        raw = await get_client().get(make_candidate_cache_key(user_uuid, category))
        if not raw:
            return None
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else None
    except Exception as e:
        logger.debug(f"[후보군 캐시 조회 실패] ({user_uuid}, {category}): {e}")
        return None


async def cache_candidates(
    user_uuid: str,
    category: RecommendCategory,
    candidates: list[dict[str, Any]],
) -> None:
    """Top-50 후보군 Redis 캐싱"""
    try:
        await get_client().set(
            make_candidate_cache_key(user_uuid, category),
            json.dumps(candidates, ensure_ascii=False),
            ex=constant.REC_CANDIDATE_TTL,
        )
    except Exception as e:
        logger.debug(f"[후보군 캐시 저장 실패] ({user_uuid}, {category}): {e}")


async def invalidate_user_candidates(user_uuid: str) -> None:
    """강한 행동 발생 시 해당 유저의 전 카테고리 후보군 캐시를 즉시 무효화"""
    try:
        r = get_client()
        for cat in ALL_CATEGORIES:
            await r.delete(make_candidate_cache_key(user_uuid, cat))
    except Exception as e:
        logger.debug(f"[후보군 캐시 무효화 실패] ({user_uuid}): {e}")


async def acquire_candidate_lock(user_uuid: str, category: RecommendCategory) -> bool:
    """F5 Cache Stampede 방지용 분산 락 획득"""
    try:
        return bool(
            await get_client().set(
                make_candidate_lock_key(user_uuid, category),
                "1",
                nx=True,
                ex=constant.REC_CANDIDATE_LOCK_TTL_SECONDS,
            )
        )
    except Exception:
        return False


async def release_candidate_lock(user_uuid: str, category: RecommendCategory) -> None:
    try:
        await get_client().delete(make_candidate_lock_key(user_uuid, category))
    except Exception:
        pass


async def get_cached_popular(category: RecommendCategory) -> list[dict[str, Any]] | None:
    try:
        raw = await get_client().get(make_popular_cache_key(category))
        if not raw:
            return None
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else None
    except Exception:
        return None


async def cache_popular(category: RecommendCategory, items: list[dict[str, Any]]) -> None:
    try:
        await get_client().set(
            make_popular_cache_key(category),
            json.dumps(items, ensure_ascii=False),
            ex=constant.REC_POPULAR_CACHE_TTL,
        )
    except Exception:
        pass


async def record_trend_event(
    *,
    category: RecommendCategory,
    item_key: str,
    delta: float,
    fan_artist_ids: list[int] | None = None,
) -> None:
    """실시간 팬덤/전체 트렌드 ZSET 증감 반영 (음수 회피 신호도 그대로 반영)"""
    try:
        r = get_client()
        global_key = make_global_trend_key(category)
        # 전체 급상승은 긍정 신호만 누적하고, 팬덤 트렌드는 음수 페널티도 반영
        if delta > 0:
            await r.zincrby(global_key, delta, item_key)
            await r.expire(global_key, 60 * 60 * 24)

        for aid in (fan_artist_ids or [])[:5]:
            fan_key = make_fandom_trend_key(int(aid))
            await r.zincrby(fan_key, delta, item_key)
            await r.expire(fan_key, 60 * 60 * 24)
    except Exception as e:
        logger.debug(f"[트렌드 반영 실패] ({item_key}): {e}")


async def get_fandom_trends(artist_id: int) -> dict[str, float]:
    """특정 아티스트 팬덤 트렌드(양수/음수 포함) 조회"""
    try:
        pairs = await get_client().zrange(
            make_fandom_trend_key(artist_id),
            0,
            -1,
            withscores=True,
        )
        return {str(k): float(s) for k, s in pairs}
    except Exception:
        return {}


async def get_global_trends(category: RecommendCategory, top_k: int = 30) -> dict[str, float]:
    """카테고리별 전체 급상승 트렌드 상위 조회"""
    try:
        pairs = await get_client().zrange(
            make_global_trend_key(category),
            0,
            -1,
            withscores=True,
        )
        sorted_pairs = sorted(pairs, key=lambda kv: float(kv[1]), reverse=True)[:top_k]
        return {str(k): float(s) for k, s in sorted_pairs}
    except Exception:
        return {}


async def save_index_snapshot(snapshot_b64: str) -> int:
    """K8s 멀티 Pod 공유용 인덱스 스냅샷을 Redis에 저장하고 버전을 1 증가"""
    try:
        r = get_client()
        await r.set(make_index_snapshot_key(), snapshot_b64)
        new_version = int(await r.incr(make_index_version_key()))
        return new_version
    except Exception as e:
        logger.warning(f"[인덱스 스냅샷 Redis 저장 실패]: {e}")
        return 0


async def get_index_version() -> int:
    """Redis에 올라간 최신 인덱스 버전 번호 조회"""
    try:
        raw = await get_client().get(make_index_version_key())
        return int(raw) if raw else 0
    except Exception:
        return 0


async def load_index_snapshot() -> tuple[int, str] | None:
    """Redis에서 최신 인덱스 스냅샷과 버전 조회"""
    try:
        r = get_client()
        snapshot = await r.get(make_index_snapshot_key())
        if not snapshot:
            return None
        ver_raw = await r.get(make_index_version_key())
        version = int(ver_raw) if ver_raw else 1
        return version, str(snapshot)
    except Exception as e:
        logger.warning(f"[인덱스 스냅샷 Redis 조회 실패]: {e}")
        return None
