import hmac
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status

from app.config import setting
from app.server.auth import get_user_uuid
from . import service
from .schemas import (
    FeedbackRequest,
    RecommendCategory,
    RecommendationPage,
    SemanticSearchRequest,
    SemanticSearchResponse,
    TargetType,
)

router = APIRouter(prefix="/api/recommender", tags=["recommender"])

UserUUID = Annotated[str, Depends(get_user_uuid)]


def _verify_admin(x_admin_secret: str = Header(default="")) -> None:
    """관리자 조작 API 접근 검증 (X-Admin-Secret 헤더)"""
    expected = setting.RECOMMENDER_ADMIN_SECRET.strip()
    if not expected:
        if setting.is_production:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="관리자 시크릿이 설정되지 않았습니다.",
            )
        return
    if not x_admin_secret or not hmac.compare_digest(x_admin_secret.strip(), expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="관리자 권한이 없습니다.",
        )


AdminGuard = Depends(_verify_admin)


async def _optional_user_uuid(request: Request) -> str | None:
    """공개 경로에서 로그인 유저일 때만 미세 개인화를 위한 선택적 인증"""
    try:
        return await get_user_uuid(request)
    except HTTPException:
        return None


OptionalUserUUID = Annotated[str | None, Depends(_optional_user_uuid)]


@router.get("/status")
async def recommender_status():
    """추천 엔진 준비 상태 조회"""
    return service.get_status()


# 관리자(Studio) 조작 API ==================================================================

@router.get("/admin/status", dependencies=[AdminGuard])
async def admin_recommender_status():
    """관리자용 추천 엔진 상세 상태 조회"""
    return service.get_admin_status()


@router.post("/admin/sync", dependencies=[AdminGuard])
async def admin_sync_catalog():
    """전체 카탈로그·공연 즉시 재동기화"""
    count = await service.sync_catalog_items()
    return {"ok": True, "synced_count": count}


@router.post("/admin/items/{target_type}/{item_id}/upsert", dependencies=[AdminGuard])
async def admin_upsert_item(target_type: TargetType, item_id: int):
    """신규 등록·수정된 단일 아이템 1건 즉시 임베딩 및 인덱스 반영"""
    ok = await service.upsert_item_from_backend(target_type, item_id)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"백엔드에서 아이템을 찾을 수 없습니다: {target_type}:{item_id}",
        )
    return {"ok": True, "item_key": f"{target_type}:{item_id}"}


@router.delete("/admin/items/{target_type}/{item_id}", dependencies=[AdminGuard])
async def admin_remove_item(target_type: TargetType, item_id: int):
    """삭제·비공개 처리된 단일 아이템 인덱스에서 즉시 제거"""
    removed = await service.remove_item_from_index(target_type, item_id)
    return {"ok": True, "removed": removed, "item_key": f"{target_type}:{item_id}"}


@router.post("/admin/reload-artifact", dependencies=[AdminGuard])
async def admin_reload_artifact():
    """새로 학습된 아티팩트(.npz + metadata.json) 무중단 핫 리로드"""
    loaded = await service.reload_artifact()
    if not loaded:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="로드할 학습 아티팩트가 없습니다.",
        )
    return {"ok": True, **service.get_status()}


@router.post("/admin/flush-writeback", dependencies=[AdminGuard])
async def admin_flush_writeback():
    """Redis dirty 유저 취향 벡터를 백엔드 DB로 즉시 Write-Back"""
    flushed = await service.flush_dirty_user_embeddings()
    return {"ok": True, "flushed": flushed}


@router.post("/admin/cache/clear", dependencies=[AdminGuard])
async def admin_clear_cache():
    """인기 목록 캐시 즉시 초기화"""
    await service.clear_popular_cache()
    return {"ok": True}


def _parse_csv_list(values: list[str] | None) -> list[str]:
    """v2 칩 파라미터 파싱: ?genre=EMO,SHOEGAZE 또는 ?genre=EMO&genre=SHOEGAZE 모두 지원"""
    if not values:
        return []
    out: list[str] = []
    for raw in values:
        for part in str(raw).split(","):
            token = part.strip().upper()
            if token and token not in out:
                out.append(token)
    return out


def _extract_v2_filters(
    genre: list[str] | None = Query(default=None),
    gender: list[str] | None = Query(default=None),
    composition: list[str] | None = Query(default=None),
    format: list[str] | None = Query(default=None),
    category: list[str] | None = Query(default=None),
    minPrice: int | None = Query(default=None, ge=0),
    maxPrice: int | None = Query(default=None, ge=0),
    excludeSoldOut: bool | None = Query(default=None),
    saleStatus: list[str] | None = Query(default=None),
    type: list[str] | None = Query(default=None),
    region: list[str] | None = Query(default=None),
) -> dict:
    if minPrice is not None and maxPrice is not None and maxPrice < 150_000 and minPrice > maxPrice:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="minPrice는 maxPrice보다 클 수 없습니다.",
        )
    return {
        "genre": _parse_csv_list(genre),
        "gender": _parse_csv_list(gender),
        "composition": _parse_csv_list(composition),
        "format": _parse_csv_list(format),
        "category": _parse_csv_list(category),
        "minPrice": minPrice,
        "maxPrice": maxPrice,
        "excludeSoldOut": excludeSoldOut,
        "saleStatus": _parse_csv_list(saleStatus),
        "type": _parse_csv_list(type),
        "region": _parse_csv_list(region),
    }


V2Filters = Annotated[dict, Depends(_extract_v2_filters)]


@router.get("/popular/{tab}", response_model=RecommendationPage)
async def get_popular_items(
    tab: RecommendCategory,
    filters: V2Filters,
    page: int = Query(default=0, ge=0),
    size: int = Query(default=20, ge=1, le=50),
):
    """카테고리별 전체 인기 목록 조회 (공개 · v2 칩 필터 지원)"""
    return await service.get_popular(category=tab, page=page, size=size, filters=filters)


@router.get("/similar/{target_type}/{item_id}", response_model=RecommendationPage)
async def get_similar_items(
    target_type: TargetType,
    item_id: int,
    user_uuid: OptionalUserUUID,
    size: int = Query(default=10, ge=1, le=30),
):
    """상세 페이지 하단 특수 추천 (공개 · 로그인 시 미세 개인화)"""
    return await service.get_similar_items(
        target_type=target_type,
        item_id=item_id,
        size=size,
        user_uuid=user_uuid,
    )


@router.get("/onboarding/similar")
async def get_onboarding_similar(
    artistId: int | None = Query(default=None),
    exclude: str = Query(default=""),
    size: int = Query(default=24, ge=1, le=50),
):
    """가입 온보딩 밴드 체인 추천 (공개)"""
    exclude_ids: set[int] = set()
    for part in exclude.split(","):
        p = part.strip()
        if p.isdigit():
            exclude_ids.add(int(p))
    items = await service.get_onboarding_similar(
        artist_id=artistId,
        exclude_ids=exclude_ids,
        size=size,
    )
    return {"items": items}


@router.post("/semantic-search", response_model=SemanticSearchResponse)
async def semantic_search(
    payload: SemanticSearchRequest,
    user_uuid: OptionalUserUUID,
):
    """자연어 기반 시맨틱 검색"""
    return await service.semantic_search(
        query=payload.query,
        categories=payload.categories,
        size=payload.size,
        user_uuid=user_uuid,
    )


@router.post("/feedback", status_code=status.HTTP_202_ACCEPTED)
async def submit_feedback(payload: FeedbackRequest, user_uuid: UserUUID):
    """실시간 행동/관심없음 피드백 반영 (로그인 필요)"""
    await service.apply_feedback(user_uuid=user_uuid, feedback=payload)
    return {"ok": True}


@router.get("/{tab}", response_model=RecommendationPage)
async def get_recommended_items(
    tab: RecommendCategory,
    user_uuid: UserUUID,
    filters: V2Filters,
    page: int = Query(default=0, ge=0),
    size: int = Query(default=20, ge=1, le=50),
):
    """카테고리별 개인화 전체 추천 목록 조회 (로그인 필요 · v2 칩 필터 지원)"""
    return await service.get_recommendations(
        category=tab,
        user_uuid=user_uuid,
        page=page,
        size=size,
        filters=filters,
    )
