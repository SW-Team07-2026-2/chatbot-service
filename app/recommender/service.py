"""개인화 추천 · 인기 · 특수추천 · 시맨틱 검색 · 실시간 피드백 핵심 서비스."""

import asyncio
from datetime import datetime, timezone
import math
import random
import time
from typing import Any

import numpy as np

from app.config import constant, logger, setting, to_kst
from . import ingest_client, store
from .catalog_sync import sync_catalog_to_index, upsert_single_item_from_backend
from .embedding import embed_query, l2_normalize
from .index import ItemIndex, get_item_index
from .schemas import (
    CATEGORY_TO_TARGET_TYPE,
    FeedbackRequest,
    ItemRecord,
    ReasonType,
    RecommendCategory,
    RecommendationPage,
    RecommendedItem,
    SemanticSearchResponse,
    TARGET_TYPE_TO_CATEGORY,
    TargetType,
    UserTasteState,
)

__all__ = [
    "startup",
    "shutdown",
    "get_status",
    "get_admin_status",
    "get_recommendations",
    "get_popular",
    "get_similar_items",
    "get_onboarding_similar",
    "semantic_search",
    "apply_feedback",
    "sync_catalog_items",
    "upsert_item_from_backend",
    "remove_item_from_index",
    "reload_artifact",
    "clear_popular_cache",
    "flush_dirty_user_embeddings",
]

METRIC_WEIGHTS: dict[str, float] = {
    "DETAIL_VIEW": 1.0,
    "PREVIEW_PLAY": 1.5,
    "RECOMMEND_CLICK": 2.0,
    "DWELL": 2.5,
    "SEARCH_CLICK": 2.5,
    "SEARCH": 2.0,
    "SHARE": 3.0,
    "EXTERNAL_LINK_CLICK": 2.0,
    "CART_ADD": 3.5,
    "FOLLOW": 4.5,
    "PURCHASE": 5.0,
    "BOOKING": 5.0,
    "CART_REMOVE": -1.0,
    "SKIP": -1.5,
    "BOUNCE": -2.0,
    "UNFOLLOW": -4.0,
    "DISLIKE": -5.0,
}

REVISIT_METRICS: set[str] = {
    "DETAIL_VIEW",
    "DWELL",
    "PREVIEW_PLAY",
    "RECOMMEND_CLICK",
    "SEARCH_CLICK",
    "CART_ADD",
    "SHARE",
}

CACHE_INVALIDATING_METRICS: set[str] = {
    "DETAIL_VIEW",
    "DWELL",
    "CART_ADD",
    "FOLLOW",
    "UNFOLLOW",
    "PURCHASE",
    "BOOKING",
    "DISLIKE",
    "BOUNCE",
}


_scheduler_tasks: list[asyncio.Task] = []
_last_synced_at: float | None = None
_last_writeback_at: float | None = None


async def _writeback_loop() -> None:
    """주기적으로 Redis dirty 유저 벡터를 백엔드 DB로 Write-Back"""
    while True:
        await asyncio.sleep(constant.REC_WRITEBACK_INTERVAL_SECONDS)
        try:
            flushed = await flush_dirty_user_embeddings()
            if flushed:
                logger.info(f"[Recommender Write-Back] {flushed}건 백업 완료")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[Recommender Write-Back 루프 오류]: {e}")


async def _catalog_sync_loop(*, run_immediately: bool = False) -> None:
    """기동 직후(아티팩트 부재 시) 및 주기적으로 카탈로그 자동 동기화"""
    if run_immediately:
        try:
            await sync_catalog_items()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[Recommender 초기 카탈로그 동기화 실패]: {e}")

    while True:
        await asyncio.sleep(constant.REC_CATALOG_SYNC_INTERVAL_SECONDS)
        try:
            await sync_catalog_items()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[Recommender 주기 카탈로그 동기화 오류]: {e}")


async def startup() -> None:
    """서버 시작 시 디스크 아티팩트 로드 및 백그라운드 동기화/Write-Back 루프 시작"""
    idx = get_item_index()
    loaded = idx.load(setting.REC_ARTIFACT_DIR)
    if loaded:
        logger.info(f"Recommender 준비 완료 ({idx.item_count}건)")
    else:
        logger.info(
            f"Recommender 아티팩트 없음 ({setting.REC_ARTIFACT_DIR}) — 백그라운드 초기 동기화 대기"
        )

    if not _scheduler_tasks:
        _scheduler_tasks.append(asyncio.create_task(_writeback_loop()))
        _scheduler_tasks.append(
            asyncio.create_task(
                _catalog_sync_loop(
                    run_immediately=(not loaded and constant.REC_AUTO_SYNC_ON_STARTUP)
                )
            )
        )


async def shutdown() -> None:
    """서버 종료 시 스케줄러 중단 및 남은 dirty 유저 벡터 Write-Back"""
    while _scheduler_tasks:
        task = _scheduler_tasks.pop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    try:
        flushed = await flush_dirty_user_embeddings(limit=50)
        if flushed:
            logger.info(f"Recommender 종료 전 Write-Back 완료: {flushed}건")
    except Exception as e:
        logger.warning(f"Recommender 종료 정리 중 경고: {e}")


def get_status() -> dict[str, Any]:
    idx = get_item_index()
    return {
        "status": "READY" if idx.is_ready else "NOT_READY",
        "index_ready": idx.is_ready,
        "item_count": idx.item_count,
        "ingest_configured": ingest_client.is_ingest_configured(),
    }


def get_admin_status() -> dict[str, Any]:
    """관리자 대시보드용 상세 상태"""
    idx = get_item_index()
    by_type: dict[str, int] = {}
    for rec in idx.all_records:
        by_type[rec.target_type] = by_type.get(rec.target_type, 0) + 1
    return {
        **get_status(),
        "counts_by_type": by_type,
        "artifact_dir": setting.REC_ARTIFACT_DIR,
        "scheduler_running": any(not t.done() for t in _scheduler_tasks),
        "last_synced_at": to_kst(_last_synced_at).isoformat(timespec="seconds") if _last_synced_at else None,
        "last_writeback_at": to_kst(_last_writeback_at).isoformat(timespec="seconds") if _last_writeback_at else None,
    }


def _time_decay_factor(updated_at: float, now_ts: float, half_life_hours: float = 72.0) -> float:
    if updated_at <= 0:
        return 1.0
    elapsed_hours = max(0.0, (now_ts - updated_at) / 3600.0)
    return 0.5 ** (elapsed_hours / half_life_hours)


def _is_fresh(released_at: str | None, now_ts: float, days: int = 7) -> bool:
    if not released_at:
        return False
    try:
        text = released_at.strip()
        if len(text) == 10:
            dt = datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
        else:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        age_days = (now_ts - dt.timestamp()) / 86400.0
        return -30.0 <= age_days <= float(days)
    except ValueError:
        return False


def _context_boost(rec: ItemRecord, state: UserTasteState | None) -> float:
    """시간대(심야/주말) 및 유저 선호 지역 맥락 가산점"""
    now_kst = to_kst()
    hour = now_kst.hour
    weekday = now_kst.weekday()  # 4=금, 5=토, 6=일
    boost = 0.0

    moods_set = set(rec.moods)
    if hour >= 23 or hour < 4:
        if moods_set & {"CALM", "DREAMY", "DARK", "WARM"} or rec.target_type == "PHYSICAL_ALBUM":
            boost += 0.08
    if weekday in (4, 5, 6):
        if rec.target_type == "CONCERT" or (moods_set & {"ENERGETIC", "BRIGHT", "GROOVY"}):
            boost += 0.08

    if state and rec.region and state.preferred_regions:
        reg_score = state.preferred_regions.get(rec.region, 0.0)
        if reg_score > 0:
            boost += min(0.10, 0.04 * math.log1p(reg_score))

    return boost


def _genre_match_ratio(rec: ItemRecord, state: UserTasteState) -> float:
    """아이템 장르와 유저 genre_affinity의 정합성 (0.0 ~ 1.0)"""
    if not rec.genres or not state.genre_affinity:
        return 0.5
    max_aff = max((v for v in state.genre_affinity.values() if v > 0), default=0.0)
    if max_aff <= 0:
        return 0.5
    matched = max((state.genre_affinity.get(g, 0.0) for g in rec.genres), default=0.0)
    return max(0.0, min(1.0, matched / max_aff))


def _apply_diversity_cap(
    scored: list[dict[str, Any]],
    *,
    sticky_keys: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    """페이지 내 동일 아티스트 도배 방지 (Diversity Cap)"""
    selected: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    artist_counts: dict[int, int] = {}

    for entry in scored:
        aids: list[int] = entry.get("artist_ids") or []
        is_sticky = entry["item_key"] in sticky_keys
        cap = (
            constant.REC_MAX_STICKY_ARTIST_PER_PAGE
            if is_sticky
            else constant.REC_MAX_ARTIST_PER_PAGE
        )
        if aids and any(artist_counts.get(aid, 0) >= cap for aid in aids):
            overflow.append(entry)
            continue
        selected.append(entry)
        for aid in aids:
            artist_counts[aid] = artist_counts.get(aid, 0) + 1
        if len(selected) >= limit:
            break

    if len(selected) < limit and overflow:
        selected.extend(overflow[: limit - len(selected)])
    return selected


def _to_recommended_item(entry: dict[str, Any]) -> RecommendedItem:
    return RecommendedItem(
        item_id=int(entry["item_id"]),
        target_type=entry["target_type"],
        score=round(float(entry["score"]), 4),
        reason_type=entry["reason_type"],
        reason_text=entry.get("reason_text"),
        data=entry.get("data") or {},
    )


async def _load_or_init_user_taste(user_uuid: str) -> UserTasteState:
    """Redis 우선 조회 -> 백엔드 DB 복구 -> 빈 상태 반환"""
    state = await store.get_user_taste(user_uuid)
    if state is not None:
        return state

    backed_up = await ingest_client.fetch_member_embedding(user_uuid)
    if backed_up:
        try:
            restored = UserTasteState.model_validate(backed_up)
            await store.save_user_taste(user_uuid, restored, mark_dirty=False)
            return restored
        except Exception as e:
            logger.debug(f"[회원 벡터 복원 스킵] ({user_uuid}): {e}")

    return UserTasteState(updated_at=time.time())


def _pick_reason(
    rec: ItemRecord,
    *,
    s_taste: float,
    s_artist_aligned: float,
    s_fandom: float,
    s_global: float,
    s_fresh: float,
) -> tuple[ReasonType, str]:
    primary_artist = rec.artist_names[0] if rec.artist_names else ""
    components = [
        ("FAVORITE_ARTIST", s_artist_aligned * 0.20, f"자주 본 {primary_artist} 관련 추천" if primary_artist else "관심 아티스트 기반 추천"),
        ("FANDOM_TREND", max(0.0, s_fandom) * 0.15, "비슷한 취향의 팬들이 요즘 많이 찾는 항목"),
        ("FRESH_RELEASE", s_fresh * 0.10, "새로 등록·발매된 따끈한 소식"),
        ("GLOBAL_HOT", s_global * 0.10, "지금 INDIG에서 급상승 중인 인기 항목"),
        ("TASTE_MATCH", max(0.0, s_taste) * 0.45, "취향에 잘 맞는 추천"),
    ]
    best = max(components, key=lambda x: x[1])
    return best[0], best[2]  # type: ignore[return-value]


def _score_personalized_candidates(
    idx: ItemIndex,
    category: RecommendCategory,
    state: UserTasteState,
    fandom_trends: dict[str, float],
    global_trends: dict[str, float],
) -> list[dict[str, Any]]:
    """CPU-bound 후보군 추출 + 리랭킹 (asyncio.to_thread로 호출)"""
    target_type: TargetType = CATEGORY_TO_TARGET_TYPE[category]  # type: ignore[assignment]
    now_ts = time.time()
    disliked = set(state.disliked_items)
    purchased = set(state.purchased_items)
    seen_fresh = set(state.seen_fresh_items)

    candidate_keys: set[str] = set()
    taste_scores: dict[str, float] = {}

    # 1. 유저 벡터 기반 Top-50
    if state.vector and len(state.vector) == constant.REC_EMBED_DIM:
        for key, sim in idx.search(
            state.vector,
            target_type=target_type,
            top_k=constant.REC_CANDIDATE_POOL_SIZE,
            exclude_keys=disliked,
            active_only=True,
        ):
            candidate_keys.add(key)
            taste_scores[key] = sim

    # 2. 크로스 카테고리 전이: 상위 artist_affinity Top-5 아티스트의 해당 카테고리 아이템
    top_artists = sorted(
        ((int(aid), val) for aid, val in state.artist_affinity.items() if val > 0),
        key=lambda kv: kv[1],
        reverse=True,
    )[:5]
    for aid, _ in top_artists:
        for rec in idx.get_items_by_artist(aid, target_types={target_type}, active_only=True):
            if rec.item_key not in disliked:
                candidate_keys.add(rec.item_key)

    # 3. 팬덤 실시간 트렌드 + 글로벌 급상승 + 신규 발매 보강
    for key in list(fandom_trends.keys())[:15] + list(global_trends.keys())[:15]:
        rec = idx.get_item_meta(key)
        if rec and rec.is_active and rec.target_type == target_type and key not in disliked:
            candidate_keys.add(key)

    for rec in idx.all_records:
        if rec.is_active and rec.target_type == target_type and rec.item_key not in disliked:
            if _is_fresh(rec.released_at, now_ts) or len(candidate_keys) < constant.REC_CANDIDATE_POOL_SIZE:
                candidate_keys.add(rec.item_key)

    if not candidate_keys:
        return []

    max_artist_aff = max((v for v in state.artist_affinity.values() if v > 0), default=1.0)
    max_global = max((v for v in global_trends.values() if v > 0), default=1.0)

    u_vec = (
        l2_normalize(np.asarray(state.vector, dtype=np.float32))
        if state.vector and len(state.vector) == constant.REC_EMBED_DIM
        else None
    )

    scored: list[dict[str, Any]] = []
    for key in candidate_keys:
        rec = idx.get_item_meta(key)
        if rec is None or not rec.is_active or key in disliked:
            continue
        # 이미 예매 완료한 공연은 제외 (같은 아티스트의 다른 굿즈/LP는 유지)
        if rec.target_type == "CONCERT" and key in purchased:
            continue

        # S_taste
        if key in taste_scores:
            s_taste = taste_scores[key]
        elif u_vec is not None:
            r_vec = idx.get_rec_vector(key)
            s_taste = float(np.dot(r_vec, u_vec)) if r_vec is not None else 0.0
        else:
            s_taste = 0.0

        # S_artist_aligned = S_artist_affinity * (0.4 + 0.6 * S_item_genre_match)
        raw_aff = max((state.artist_affinity.get(str(aid), 0.0) for aid in rec.artist_ids), default=0.0)
        s_artist_aff = max(0.0, raw_aff / max_artist_aff) if max_artist_aff > 0 else 0.0
        genre_match = _genre_match_ratio(rec, state)
        s_artist_aligned = s_artist_aff * (0.4 + 0.6 * genre_match)

        # S_fandom_lift (음수 페널티는 0으로 자르지 않고 그대로 반영)
        raw_fan = fandom_trends.get(key, 0.0)
        raw_glob = global_trends.get(key, 0.0)
        if raw_fan < 0:
            s_fandom_lift = max(-1.0, raw_fan / (abs(raw_fan) + 3.0))
        elif raw_fan > 0:
            s_fandom_lift = min(1.0, raw_fan / (max(0.0, raw_glob) + 1.0))
        else:
            s_fandom_lift = 0.0

        # S_global_hot
        s_global_hot = min(1.0, max(0.0, raw_glob) / max_global) if raw_glob > 0 else 0.0

        # S_fresh (첫 노출 시 부스트, 1회 확인 후 제거)
        s_fresh = 1.0 if (_is_fresh(rec.released_at, now_ts) and key not in seen_fresh) else 0.0

        # S_context (시간대/지역)
        s_context = _context_boost(rec, state)

        # 음악 팬덤 특화 피로도(Cooldown) vs 반복 진입(Sticky Retention)
        revisits = state.revisit_counts.get(key, 0)
        impressions = state.impression_counts.get(key, 0)
        if revisits >= 1:
            s_fatigue_or_sticky = min(0.35, 0.10 * math.log1p(revisits))
        elif impressions >= 2:
            s_fatigue_or_sticky = -min(0.60, 0.18 * (impressions - 1))
        else:
            s_fatigue_or_sticky = 0.0

        total_score = (
            0.45 * s_taste
            + 0.20 * s_artist_aligned
            + 0.15 * s_fandom_lift
            + 0.10 * s_global_hot
            + 0.10 * s_fresh
            + s_context
            + s_fatigue_or_sticky
        )

        reason_type, reason_text = _pick_reason(
            rec,
            s_taste=s_taste,
            s_artist_aligned=s_artist_aligned,
            s_fandom=s_fandom_lift,
            s_global=s_global_hot,
            s_fresh=s_fresh,
        )

        scored.append(
            {
                "item_key": key,
                "item_id": rec.item_id,
                "target_type": rec.target_type,
                "artist_ids": rec.artist_ids,
                "score": total_score,
                "fandom_lift": s_fandom_lift,
                "global_hot": s_global_hot,
                "reason_type": reason_type,
                "reason_text": reason_text,
                "data": rec.payload,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[: constant.REC_CANDIDATE_POOL_SIZE]


def _inject_global_hot_slots(
    page_items: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    *,
    slots: tuple[int, ...] = (4, 11),
) -> list[dict[str, Any]]:
    """글로벌 급상승 전용 슬롯 1~2칸 보장 (음수 팬덤 페널티가 큰 아이템은 제외)"""
    hot_candidates = [
        x
        for x in pool
        if x.get("global_hot", 0.0) >= 0.4 and x.get("fandom_lift", 0.0) >= -0.1
    ]
    hot_candidates.sort(key=lambda x: x.get("global_hot", 0.0), reverse=True)
    if not hot_candidates:
        return page_items

    result = list(page_items)
    existing_keys = {x["item_key"] for x in result}
    for slot_idx in slots:
        if slot_idx >= len(result):
            break
        pick = next((h for h in hot_candidates if h["item_key"] not in existing_keys), None)
        if pick is None:
            break
        entry = dict(pick)
        entry["reason_type"] = "GLOBAL_HOT"
        entry["reason_text"] = "지금 INDIG에서 급상승 중인 인기 항목"
        result.insert(slot_idx, entry)
        existing_keys.add(entry["item_key"])
        result.pop()
    return result


def _sample_refresh_page(
    pool: list[dict[str, Any]],
    *,
    sticky_keys: set[str],
    size: int,
) -> list[dict[str, Any]]:
    """F5 새로고침 시 Sticky 아이템은 유지하고 나머지는 Softmax/가중치 샘플링"""
    sticky = [x for x in pool if x["item_key"] in sticky_keys]
    others = [x for x in pool if x["item_key"] not in sticky_keys]

    # 나머지에 가벼운 탐색 노이즈를 섞어 F5마다 신선한 조합 제공
    jittered = [
        (x["score"] + random.uniform(-0.04, 0.04), x)
        for x in others
    ]
    jittered.sort(key=lambda pair: pair[0], reverse=True)
    merged = sticky + [x for _, x in jittered]
    capped = _apply_diversity_cap(merged, sticky_keys=sticky_keys, limit=size)
    return _inject_global_hot_slots(capped, pool)


async def get_recommendations(
    *,
    category: RecommendCategory,
    user_uuid: str,
    page: int = 0,
    size: int = 20,
    filters: dict[str, Any] | None = None,
) -> RecommendationPage:
    """개인화 전체 추천 (F5 캐싱 + Cache Stampede 락 + 리랭킹 + Diversity Cap)"""
    idx = get_item_index()
    if not idx.is_ready:
        return RecommendationPage(items=[], page=page, size=size, hasNext=False, degraded=True)

    state = await _load_or_init_user_taste(user_uuid)
    sticky_keys = {k for k, cnt in state.revisit_counts.items() if cnt >= 1}

    pool = await store.get_cached_candidates(user_uuid, category)
    if pool is None:
        lock_acquired = await store.acquire_candidate_lock(user_uuid, category)
        if not lock_acquired:
            for _ in range(3):
                await asyncio.sleep(0.08)
                pool = await store.get_cached_candidates(user_uuid, category)
                if pool is not None:
                    break

        if pool is None:
            try:
                top_artists = sorted(
                    ((int(aid), val) for aid, val in state.artist_affinity.items() if val > 0),
                    key=lambda kv: kv[1],
                    reverse=True,
                )[:3]
                fandom_maps = await asyncio.gather(
                    *(store.get_fandom_trends(aid) for aid, _ in top_artists)
                )
                merged_fandom: dict[str, float] = {}
                for fmap in fandom_maps:
                    for k, v in fmap.items():
                        merged_fandom[k] = merged_fandom.get(k, 0.0) + v

                global_trends = await store.get_global_trends(category)
                pool = await asyncio.to_thread(
                    _score_personalized_candidates,
                    idx,
                    category,
                    state,
                    merged_fandom,
                    global_trends,
                )
                await store.cache_candidates(user_uuid, category, pool)
            finally:
                if lock_acquired:
                    await store.release_candidate_lock(user_uuid, category)

    if page == 0:
        selected = _sample_refresh_page(pool, sticky_keys=sticky_keys, size=size)
        has_next = len(pool) > size
    else:
        start = page * size
        end = start + size
        capped = _apply_diversity_cap(pool, sticky_keys=sticky_keys, limit=end)
        selected = capped[start:end]
        has_next = len(capped) > end

    # 노출 피로도 및 Freshness 확인 기록 업데이트 (Redis)
    if selected:
        now_ts = time.time()
        seen_fresh_set = set(state.seen_fresh_items)
        for entry in selected:
            k = entry["item_key"]
            state.impression_counts[k] = state.impression_counts.get(k, 0) + 1
            rec = idx.get_item_meta(k)
            if rec and _is_fresh(rec.released_at, now_ts):
                seen_fresh_set.add(k)
        state.seen_fresh_items = list(seen_fresh_set)[-200:]
        await store.save_user_taste(user_uuid, state, mark_dirty=False)

    return RecommendationPage(
        items=[_to_recommended_item(e) for e in selected],
        page=page,
        size=size,
        hasNext=has_next,
        degraded=False,
    )


async def get_popular(
    *,
    category: RecommendCategory,
    page: int = 0,
    size: int = 20,
    filters: dict[str, Any] | None = None,
) -> RecommendationPage:
    """비로그인 공개 전체 인기 목록 (실시간 트렌드 + 기본 인기도 + 신선도)"""
    idx = get_item_index()
    if not idx.is_ready:
        return RecommendationPage(items=[], page=page, size=size, hasNext=False, degraded=True)

    cached = await store.get_cached_popular(category)
    if cached is None:
        target_type: TargetType = CATEGORY_TO_TARGET_TYPE[category]  # type: ignore[assignment]
        global_trends = await store.get_global_trends(category, top_k=100)
        now_ts = time.time()

        recs = [
            r for r in idx.all_records
            if r.is_active and r.target_type == target_type
        ]
        max_trend = max((v for v in global_trends.values() if v > 0), default=1.0)
        max_prior = max((r.popularity_prior for r in recs if r.popularity_prior > 0), default=1.0)

        scored: list[dict[str, Any]] = []
        for r in recs:
            t_score = max(0.0, global_trends.get(r.item_key, 0.0)) / max_trend if max_trend > 0 else 0.0
            p_score = math.log1p(max(0.0, r.popularity_prior)) / math.log1p(max_prior) if max_prior > 0 else 0.0
            f_score = 1.0 if _is_fresh(r.released_at, now_ts) else 0.0
            total = 0.60 * t_score + 0.30 * p_score + 0.10 * f_score
            scored.append(
                {
                    "item_key": r.item_key,
                    "item_id": r.item_id,
                    "target_type": r.target_type,
                    "artist_ids": r.artist_ids,
                    "score": total,
                    "reason_type": "GLOBAL_HOT",
                    "reason_text": "지금 인기 있는 항목",
                    "data": r.payload,
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        cached = _apply_diversity_cap(scored, sticky_keys=set(), limit=100)
        await store.cache_popular(category, cached)

    start = page * size
    end = start + size
    return RecommendationPage(
        items=[_to_recommended_item(e) for e in cached[start:end]],
        page=page,
        size=size,
        hasNext=len(cached) > end,
        degraded=False,
    )


async def get_similar_items(
    *,
    target_type: TargetType,
    item_id: int,
    size: int = 10,
    user_uuid: str | None = None,
) -> RecommendationPage:
    """상세 페이지 하단 특수 추천 (Item-to-Item 유사도 + Co-occurrence + 미세 개인화)"""
    idx = get_item_index()
    if not idx.is_ready:
        return RecommendationPage(items=[], page=0, size=size, hasNext=False, degraded=True)

    anchor_key = f"{target_type}:{item_id}"
    anchor_meta = idx.get_item_meta(anchor_key)
    anchor_vec = idx.get_rec_vector(anchor_key)
    if anchor_meta is None or anchor_vec is None:
        return RecommendationPage(items=[], page=0, size=size, hasNext=False, degraded=False)

    # 같은 아티스트 팬덤 Co-occurrence 조회
    co_scores: dict[str, float] = {}
    for aid in anchor_meta.artist_ids[:3]:
        for k, v in (await store.get_fandom_trends(aid)).items():
            co_scores[k] = co_scores.get(k, 0.0) + v

    user_vec: np.ndarray | None = None
    if user_uuid:
        state = await store.get_user_taste(user_uuid)
        if state and state.vector and len(state.vector) == constant.REC_EMBED_DIM:
            user_vec = l2_normalize(np.asarray(state.vector, dtype=np.float32))

    raw_neighbors = await asyncio.to_thread(
        idx.search,
        anchor_vec,
        target_type=target_type,
        top_k=max(size * 3, 30),
        exclude_keys={anchor_key},
        active_only=True,
    )

    anchor_genres = set(anchor_meta.genres)
    anchor_moods = set(anchor_meta.moods)
    max_co = max((v for v in co_scores.values() if v > 0), default=1.0)

    scored: list[dict[str, Any]] = []
    for key, sim in raw_neighbors:
        rec = idx.get_item_meta(key)
        if rec is None:
            continue
        co_val = co_scores.get(key, 0.0)
        s_co = (co_val / max_co) if co_val > 0 else (max(-0.5, co_val / 5.0) if co_val < 0 else 0.0)

        overlap_g = len(anchor_genres & set(rec.genres)) / max(len(anchor_genres), 1) if anchor_genres else 0.0
        overlap_m = len(anchor_moods & set(rec.moods)) / max(len(anchor_moods), 1) if anchor_moods else 0.0
        s_overlap = 0.6 * overlap_g + 0.4 * overlap_m

        same_artist_bonus = 0.15 if (
            target_type != "ARTIST" and (set(anchor_meta.artist_ids) & set(rec.artist_ids))
        ) else 0.0

        s_user = 0.0
        if user_vec is not None:
            r_vec = idx.get_rec_vector(key)
            if r_vec is not None:
                s_user = 0.15 * float(np.dot(r_vec, user_vec))

        total = 0.55 * sim + 0.30 * s_co + 0.15 * s_overlap + same_artist_bonus + s_user
        reason_type: ReasonType = "FANDOM_TREND" if s_co > 0.4 else "TASTE_MATCH"
        reason_text = (
            "이 항목을 본 팬들이 함께 많이 찾았어요"
            if reason_type == "FANDOM_TREND"
            else "분위기와 스타일이 비슷한 추천"
        )
        scored.append(
            {
                "item_key": key,
                "item_id": rec.item_id,
                "target_type": rec.target_type,
                "artist_ids": rec.artist_ids,
                "score": total,
                "reason_type": reason_type,
                "reason_text": reason_text,
                "data": rec.payload,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return RecommendationPage(
        items=[_to_recommended_item(e) for e in scored[:size]],
        page=0,
        size=size,
        hasNext=False,
        degraded=False,
    )


async def get_onboarding_similar(
    *,
    artist_id: int | None = None,
    exclude_ids: set[int] | None = None,
    size: int = 24,
) -> list[dict[str, Any]]:
    """가입 온보딩 밴드 체인 추천 (공개)"""
    idx = get_item_index()
    if not idx.is_ready:
        return []

    excluded_keys = {f"ARTIST:{eid}" for eid in (exclude_ids or set())}
    if artist_id is None:
        # 첫 세트: 장르를 고루 섞어 인기순 반환
        artists = [
            r for r in idx.all_records
            if r.target_type == "ARTIST" and r.is_active and r.item_key not in excluded_keys
        ]
        artists.sort(key=lambda r: r.popularity_prior, reverse=True)
        return [r.payload for r in artists[:size]]

    excluded_keys.add(f"ARTIST:{artist_id}")
    anchor_vec = idx.get_rec_vector(f"ARTIST:{artist_id}")
    if anchor_vec is None:
        return []

    neighbors = await asyncio.to_thread(
        idx.search,
        anchor_vec,
        target_type="ARTIST",
        top_k=size,
        exclude_keys=excluded_keys,
        active_only=True,
    )
    items: list[dict[str, Any]] = []
    for key, _ in neighbors:
        rec = idx.get_item_meta(key)
        if rec:
            items.append(rec.payload)
    return items


async def semantic_search(
    *,
    query: str,
    categories: list[RecommendCategory] | None = None,
    size: int = 10,
    user_uuid: str | None = None,
) -> SemanticSearchResponse:
    """자연어 기반 하이브리드 시맨틱 검색 (0.75 순수 텍스트 + 0.25 행동 클러스터 보정)"""
    idx = get_item_index()
    clean_query = query.strip()
    if not clean_query or not idx.is_ready:
        return SemanticSearchResponse(query=clean_query, items=[], degraded=not idx.is_ready)

    target_types: set[TargetType] | None = None
    if categories:
        target_types = {
            CATEGORY_TO_TARGET_TYPE[c]  # type: ignore[misc]
            for c in categories
            if c in CATEGORY_TO_TARGET_TYPE
        }

    user_vec: np.ndarray | None = None
    if user_uuid:
        state = await store.get_user_taste(user_uuid)
        if state and state.vector and len(state.vector) == constant.REC_EMBED_DIM:
            user_vec = l2_normalize(np.asarray(state.vector, dtype=np.float32))

    try:
        q_vec = await embed_query(clean_query)
    except Exception as e:
        logger.warning(f"[시맨틱 검색 임베딩 실패 -> 키워드 폴백] query={clean_query}: {e}")
        return _keyword_fallback_search(idx, clean_query, target_types, size)

    # 1차: 순수 텍스트 인덱스 Top-K
    text_hits = await asyncio.to_thread(
        idx.search_text,
        q_vec,
        target_types=target_types,
        top_k=max(size * 3, 30),
        active_only=True,
    )
    if not text_hits:
        return SemanticSearchResponse(query=clean_query, items=[], degraded=False)

    # 2차: 상위 앵커(최대 5개)의 128d 행동 클러스터 중심으로 행동 보정 후보 확장
    anchor_vecs = [
        v
        for key, _ in text_hits[:5]
        if (v := idx.get_rec_vector(key)) is not None
    ]
    text_score_map = dict(text_hits)
    behavior_score_map: dict[str, float] = {}

    if anchor_vecs:
        cluster_center = l2_normalize(np.mean(np.stack(anchor_vecs, axis=0), axis=0))
        behav_hits = await asyncio.to_thread(
            idx.search,
            cluster_center,
            target_types=target_types,
            top_k=max(size * 3, 30),
            active_only=True,
        )
        behavior_score_map = dict(behav_hits)

    all_keys = set(text_score_map.keys()) | set(behavior_score_map.keys())
    scored: list[RecommendedItem] = []

    for key in all_keys:
        rec = idx.get_item_meta(key)
        if rec is None or not rec.is_active:
            continue
        s_text = text_score_map.get(key)
        if s_text is None:
            t_vec = idx.get_text_vector(key)
            s_text = float(np.dot(t_vec, q_vec)) if t_vec is not None else 0.0

        s_behav = behavior_score_map.get(key, 0.0)
        s_user = 0.0
        if user_vec is not None:
            r_vec = idx.get_rec_vector(key)
            if r_vec is not None:
                s_user = 0.10 * float(np.dot(r_vec, user_vec))

        final_score = 0.75 * s_text + 0.25 * s_behav + s_user
        scored.append(
            RecommendedItem(
                item_id=rec.item_id,
                target_type=rec.target_type,
                score=round(final_score, 4),
                reason_type="SEMANTIC_MATCH",
                reason_text=f"'{clean_query}' 분위기와 잘 어울리는 추천",
                data=rec.payload,
            )
        )

    scored.sort(key=lambda x: x.score, reverse=True)
    return SemanticSearchResponse(query=clean_query, items=scored[:size], degraded=False)


def _keyword_fallback_search(
    idx: ItemIndex,
    query: str,
    target_types: set[TargetType] | None,
    size: int,
) -> SemanticSearchResponse:
    tokens = [t for t in query.replace(",", " ").lower().split() if t]
    scored: list[RecommendedItem] = []
    for rec in idx.all_records:
        if not rec.is_active:
            continue
        if target_types and rec.target_type not in target_types:
            continue
        haystack = rec.text_doc.lower()
        hits = sum(1 for tok in tokens if tok in haystack)
        if hits > 0:
            scored.append(
                RecommendedItem(
                    item_id=rec.item_id,
                    target_type=rec.target_type,
                    score=round(hits / max(len(tokens), 1), 4),
                    reason_type="SEMANTIC_MATCH",
                    reason_text=f"'{query}' 키워드 일치",
                    data=rec.payload,
                )
            )
    scored.sort(key=lambda x: x.score, reverse=True)
    return SemanticSearchResponse(query=query, items=scored[:size], degraded=True)


async def apply_feedback(user_uuid: str, feedback: FeedbackRequest) -> None:
    """실시간 행동 피드백을 유저 취향 좌표 · 친화도 · 피로도/Sticky · 트렌드에 즉시 반영"""
    idx = get_item_index()
    item_key = f"{feedback.target_type}:{feedback.target_id}"
    rec = idx.get_item_meta(item_key)
    item_vec = idx.get_rec_vector(item_key)

    now_ts = time.time()
    state = await _load_or_init_user_taste(user_uuid)
    decay = _time_decay_factor(state.updated_at, now_ts)

    weight = METRIC_WEIGHTS.get(feedback.metric_type, 1.0)
    if feedback.metric_type == "DWELL" and feedback.metric_value >= 30:
        weight = max(weight, 3.0)

    # 1. 기존 친화도 시간 감쇠
    state.artist_affinity = {
        k: round(v * decay, 4)
        for k, v in state.artist_affinity.items()
        if abs(v * decay) > 0.05
    }
    state.genre_affinity = {
        k: round(v * decay, 4)
        for k, v in state.genre_affinity.items()
        if abs(v * decay) > 0.05
    }

    # 2. 유저 벡터 실시간 이동 (Time Decay + 행동 가중치)
    if item_vec is not None:
        lr = 0.25 if weight > 0 else 0.15
        if state.vector and len(state.vector) == constant.REC_EMBED_DIM:
            prev = np.asarray(state.vector, dtype=np.float32) * decay
            updated = l2_normalize(prev + lr * (weight / 5.0) * item_vec)
        else:
            updated = l2_normalize(item_vec if weight > 0 else -0.3 * item_vec)
        state.vector = updated.tolist()

    # 3. 크로스 카테고리 전이용 아티스트/장르/지역 친화도 갱신
    artist_ids = list(rec.artist_ids) if rec else []
    if feedback.artist_id is not None and feedback.artist_id not in artist_ids:
        artist_ids.append(feedback.artist_id)

    for aid in artist_ids:
        key = str(aid)
        state.artist_affinity[key] = round(state.artist_affinity.get(key, 0.0) + weight, 4)

    if rec:
        for g in rec.genres:
            state.genre_affinity[g] = round(state.genre_affinity.get(g, 0.0) + weight * 0.5, 4)
        if rec.region and weight > 0:
            state.preferred_regions[rec.region] = round(
                state.preferred_regions.get(rec.region, 0.0) + 1.0, 4
            )

    # 4. Sticky Retention / 소비완료 / 관심없음 기록
    if feedback.metric_type in REVISIT_METRICS:
        state.revisit_counts[item_key] = state.revisit_counts.get(item_key, 0) + 1
    if feedback.metric_type in ("BOOKING", "PURCHASE"):
        if item_key not in state.purchased_items:
            state.purchased_items = (state.purchased_items + [item_key])[-100:]
    if feedback.metric_type == "DISLIKE":
        if item_key not in state.disliked_items:
            state.disliked_items = (state.disliked_items + [item_key])[-100:]

    state.updated_at = now_ts
    await store.save_user_taste(user_uuid, state, mark_dirty=True)

    # 5. 실시간 팬덤/전체 트렌드 ZSET 반영
    category = TARGET_TYPE_TO_CATEGORY.get(feedback.target_type)
    if category is not None:
        top_fan_artists = [
            int(aid)
            for aid, val in sorted(
                state.artist_affinity.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[:3]
            if val > 0
        ]
        await store.record_trend_event(
            category=category,
            item_key=item_key,
            delta=weight,
            fan_artist_ids=top_fan_artists,
        )

    # 6. 강한 행동 시 후보군 캐시 즉시 무효화
    if feedback.metric_type in CACHE_INVALIDATING_METRICS:
        await store.invalidate_user_candidates(user_uuid)


async def sync_catalog_items() -> int:
    """백엔드 공개 카탈로그/공연 API에서 아이템 전체 동기화"""
    global _last_synced_at
    count = await sync_catalog_to_index(save_artifact=True)
    if count > 0:
        _last_synced_at = time.time()
        await clear_popular_cache()
    return count


async def upsert_item_from_backend(target_type: TargetType, item_id: int) -> bool:
    """관리자 등록/수정 직후 단일 아이템 1건만 즉시 임베딩 및 인덱스 반영"""
    rec = await upsert_single_item_from_backend(target_type, item_id, save_artifact=True)
    if rec is None:
        return False
    await clear_popular_cache()
    return True


async def remove_item_from_index(target_type: TargetType, item_id: int) -> bool:
    """단일 아이템을 인덱스에서 즉시 제거"""
    idx = get_item_index()
    key = f"{target_type}:{item_id}"
    if idx.get_item_meta(key) is None:
        return False
    idx.remove_item(key)
    await clear_popular_cache()
    return True


async def reload_artifact() -> bool:
    """디스크에 배포된 최신 학습 아티팩트(.npz + metadata.json) 무중단 핫 리로드"""
    idx = get_item_index()
    loaded = await asyncio.to_thread(idx.load, setting.REC_ARTIFACT_DIR)
    if loaded:
        await clear_popular_cache()
    return loaded


async def clear_popular_cache() -> None:
    """전 카테고리 인기 목록 Redis 캐시 초기화"""
    for cat in store.ALL_CATEGORIES:
        try:
            await store.get_client().delete(store.make_popular_cache_key(cat))
        except Exception:
            pass


async def flush_dirty_user_embeddings(limit: int = 100) -> int:
    """Redis dirty 유저 취향 상태를 백엔드 DB로 일괄 Write-Back 백업"""
    global _last_writeback_at
    if not ingest_client.is_ingest_configured():
        return 0

    dirty_users = await store.pop_dirty_users(limit=limit)
    if not dirty_users:
        return 0

    flushed = 0
    for uid in dirty_users:
        state = await store.get_user_taste(uid)
        if state is None:
            continue
        ok = await ingest_client.put_member_embedding(uid, state.model_dump())
        if ok:
            flushed += 1
    if flushed > 0:
        _last_writeback_at = time.time()
    return flushed
