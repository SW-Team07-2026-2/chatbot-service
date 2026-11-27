from typing import Any, Literal

from pydantic import BaseModel, Field

RecommendCategory = Literal["artists", "physical-albums", "goods", "concerts"]

CATEGORY_TO_TARGET_TYPE: dict[str, str] = {
    "artists": "ARTIST",
    "physical-albums": "PHYSICAL_ALBUM",
    "goods": "GOODS",
    "concerts": "CONCERT",
}

TARGET_TYPE_TO_CATEGORY: dict[str, RecommendCategory] = {
    "ARTIST": "artists",
    "PHYSICAL_ALBUM": "physical-albums",
    "GOODS": "goods",
    "CONCERT": "concerts",
}

TargetType = Literal[
    "ARTIST",
    "ALBUM",
    "TRACK",
    "PHYSICAL_ALBUM",
    "GOODS",
    "CONCERT",
]

MetricType = Literal[
    # 프론트 수집 메트릭
    "DETAIL_VIEW",
    "DWELL",
    "PREVIEW_PLAY",
    "SEARCH_CLICK",
    "RECOMMEND_CLICK",
    "SHARE",
    "EXTERNAL_LINK_CLICK",
    # 서버 기록 메트릭
    "CART_ADD",
    "CART_REMOVE",
    "SEARCH",
    "FOLLOW",
    "UNFOLLOW",
    "PURCHASE",
    "BOOKING",
    # 추천 전용 확장(이탈/스킵/관심없음)
    "BOUNCE",
    "SKIP",
    "DISLIKE",
]

ReasonType = Literal[
    "TASTE_MATCH",
    "FAVORITE_ARTIST",
    "FANDOM_TREND",
    "GLOBAL_HOT",
    "FRESH_RELEASE",
    "CONTEXT_BOOST",
    "EXPLORATION",
    "SEMANTIC_MATCH",
]


class ItemRecord(BaseModel):
    """추천 및 시맨틱 인덱스에 보관하는 아이템 메타데이터"""

    item_key: str = Field(..., description="고유 키 (예: ARTIST:1, CONCERT:42)")
    target_type: TargetType
    item_id: int
    artist_ids: list[int] = Field(default_factory=list)
    artist_names: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    moods: list[str] = Field(default_factory=list)
    region: str | None = None
    is_active: bool = True
    released_at: str | None = None
    popularity_prior: float = 0.0
    text_doc: str = ""
    embed_hash: str = ""
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="프론트 목록 카드 호환 원본 항목(ArtistItem, ConcertSummary 등)",
    )


class UserTasteState(BaseModel):
    """Redis에 캐시되고 백엔드 DB로 Write-Back되는 유저 실시간 취향 상태"""

    vector: list[float] = Field(default_factory=list)
    artist_affinity: dict[str, float] = Field(default_factory=dict)
    genre_affinity: dict[str, float] = Field(default_factory=dict)
    preferred_regions: dict[str, float] = Field(default_factory=dict)
    impression_counts: dict[str, int] = Field(default_factory=dict)
    revisit_counts: dict[str, int] = Field(default_factory=dict)
    purchased_items: list[str] = Field(default_factory=list)
    disliked_items: list[str] = Field(default_factory=list)
    seen_fresh_items: list[str] = Field(default_factory=list)
    updated_at: float = 0.0


class RecommendedItem(BaseModel):
    """추천 / 시맨틱 검색 응답 단일 항목"""

    item_id: int
    target_type: TargetType
    score: float
    reason_type: ReasonType
    reason_text: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class RecommendationPage(BaseModel):
    """v2 목록 규격(PageResult) 호환 추천 응답"""

    items: list[RecommendedItem] = Field(default_factory=list)
    page: int = 0
    size: int = 20
    hasNext: bool = False
    degraded: bool = False


class FeedbackRequest(BaseModel):
    """실시간 취향/피로도 반영용 피드백 요청"""

    target_type: TargetType
    target_id: int
    metric_type: MetricType
    metric_value: float = Field(default=1.0, ge=0.0)
    artist_id: int | None = None


class SemanticSearchRequest(BaseModel):
    """자연어 기반 시맨틱 검색 요청"""

    query: str = Field(..., min_length=1, max_length=200)
    categories: list[RecommendCategory] | None = None
    size: int = Field(default=10, ge=1, le=30)


class SemanticSearchResponse(BaseModel):
    """자연어 기반 시맨틱 검색 응답"""

    query: str
    items: list[RecommendedItem] = Field(default_factory=list)
    degraded: bool = False
