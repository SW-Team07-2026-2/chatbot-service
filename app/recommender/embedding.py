"""Gemini Embedding API 래퍼, 아이템 텍스트 문서 빌더, NumPy 프로젝터 추론."""

import hashlib
import json
from typing import Any

import numpy as np
from google.genai import types

from app.chatbot.uploader import get_genai_client
from app.config import constant, logger
from app.server.redis import get_client

__all__ = [
    "GENRE_KO_MAP",
    "MOOD_KO_MAP",
    "TEMPO_KO_MAP",
    "l2_normalize",
    "embed_texts",
    "embed_query",
    "project_text_to_rec",
    "build_artist_doc",
    "build_physical_album_doc",
    "build_goods_doc",
    "build_concert_doc",
]

GENRE_KO_MAP: dict[str, str] = {
    "INDIE_ROCK": "인디록 록 밴드",
    "ALTERNATIVE": "얼터너티브 록",
    "GARAGE": "개러지 록",
    "PUNK": "펑크",
    "HARDCORE": "하드코어",
    "EMO": "이모 감성 록",
    "METAL": "메탈",
    "POST_ROCK": "포스트록 몽환 연주",
    "SHOEGAZE": "슈게이징 몽환 노이즈",
    "PSYCHEDELIC": "사이키델릭",
    "INDIE_POP": "인디팝",
    "DREAM_POP": "드림팝 몽환 감성",
    "SYNTH_POP": "신스팝 일렉트로팝",
    "CITY_POP": "시티팝 레트로 밤 도시",
    "FOLK": "포크 어쿠스틱 잔잔한",
    "ELECTRONIC": "일렉트로닉 전자음악",
    "RNB_SOUL": "알앤비 소울 그루브",
    "HIPHOP": "힙합 랩",
    "JAZZ": "재즈 스윙 차분한 감성",
    "BLUES": "블루스",
    "FUNK": "펑크 그루브 신나는",
    "SKA_REGGAE": "스카 레게",
    "ETC": "인디 기타",
}

MOOD_KO_MAP: dict[str, str] = {
    "DREAMY": "몽환적인 꿈같은 새벽",
    "DARK": "어두운 우울한 쓸쓸한 비",
    "BRIGHT": "밝은 경쾌한 청량한",
    "WARM": "따뜻한 위로 회복 안정 포근한",
    "COLD": "차가운 서늘한 겨울 고독",
    "CALM": "잔잔한 차분한 안정 휴식 수면",
    "ENERGETIC": "에너지 넘치는 신나는 페스티벌",
    "INTENSE": "강렬한 폭발적인",
    "HEAVY": "묵직한 헤비한",
    "BOUNCY": "통통 튀는 리드미컬한",
    "GROOVY": "그루비한 세련된",
    "URBAN": "도시적인 밤 드라이브",
}

TEMPO_KO_MAP: dict[str, str] = {
    "SLOW": "느린 잔잔한 발라드",
    "MID": "중간 템포 미드템포",
    "FAST": "빠른 경쾌한 업템포",
    "VERY_FAST": "매우 빠른 질주감",
}


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """1D 또는 2D 배열의 마지막 축 L2 정규화"""
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 1e-12 else arr
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return np.divide(arr, np.where(norms > 1e-12, norms, 1.0))


def _expand_tags(values: list[str] | None, mapping: dict[str, str]) -> str:
    if not values:
        return ""
    tokens: list[str] = []
    for v in values:
        key = str(v).strip().upper()
        if not key:
            continue
        ko = mapping.get(key, "")
        tokens.append(f"{key}({ko})" if ko else key)
    return ", ".join(tokens)


def build_artist_doc(
    artist: dict[str, Any],
    recent_albums: list[dict[str, Any]] | None = None,
) -> tuple[str, list[str]]:
    """아티스트 텍스트 문서 생성 (최신 발매 앨범/트랙의 장르·무드에 높은 가중치 부여)"""
    name = str(artist.get("name") or "").strip()
    genres = list(artist.get("genres") or [])
    bio = str(artist.get("bio") or "").strip()
    label = str(artist.get("label") or "").strip()
    debut = artist.get("debutYear")

    collected_moods: list[str] = []
    recent_lines: list[str] = []

    if recent_albums:
        sorted_albums = sorted(
            recent_albums,
            key=lambda a: str(a.get("releasedAt") or ""),
            reverse=True,
        )
        for idx, alb in enumerate(sorted_albums[:4]):
            alb_title = str(alb.get("title") or "").strip()
            alb_genres = list(alb.get("genres") or [])
            alb_desc = str(alb.get("description") or "").strip()
            track_moods: list[str] = []
            track_emotions: list[str] = []
            track_styles: list[str] = []
            track_tempos: list[str] = []
            for tr in alb.get("tracks") or []:
                track_moods.extend(tr.get("moods") or [])
                track_emotions.extend(tr.get("emotions") or [])
                track_styles.extend(tr.get("styles") or [])
                if tr.get("tempo"):
                    track_tempos.append(str(tr["tempo"]))
                for tg in tr.get("genres") or []:
                    if tg not in alb_genres:
                        alb_genres.append(tg)

            collected_moods.extend(track_moods)
            tag_str = " ".join(
                filter(
                    None,
                    [
                        _expand_tags(alb_genres, GENRE_KO_MAP),
                        _expand_tags(list(dict.fromkeys(track_moods)), MOOD_KO_MAP),
                        _expand_tags(list(dict.fromkeys(track_tempos)), TEMPO_KO_MAP),
                        ", ".join(dict.fromkeys(track_emotions)),
                        ", ".join(dict.fromkeys(track_styles)),
                    ],
                )
            )
            line = f"앨범 '{alb_title}' ({tag_str}) {alb_desc}".strip()
            recent_lines.append(line)
            # 가장 최신 발매 앨범은 장르 전환 반영을 위해 1회 더 강조
            if idx == 0 and tag_str:
                recent_lines.insert(0, f"최근 주력 스타일: {tag_str}")

    parts = [
        f"아티스트: {name}",
        f"기본 장르: {_expand_tags(genres, GENRE_KO_MAP)}" if genres else "",
        f"레이블: {label}" if label else "",
        f"데뷔: {debut}" if debut else "",
        f"소개: {bio}" if bio else "",
        *recent_lines,
    ]
    doc = "\n".join(p for p in parts if p).strip()
    return doc, list(dict.fromkeys(collected_moods))


def build_physical_album_doc(
    item: dict[str, Any],
    album_detail: dict[str, Any] | None = None,
) -> tuple[str, list[str], list[str]]:
    """피지컬 앨범(LP/CD/CASSETTE) 텍스트 문서 생성"""
    name = str(item.get("name") or item.get("albumTitle") or "").strip()
    artist_name = str(item.get("artistName") or "").strip()
    fmt = str(item.get("format") or "").strip()
    desc = str(item.get("description") or "").strip()

    genres: list[str] = []
    moods: list[str] = []
    extra = ""
    if album_detail:
        genres = list(album_detail.get("genres") or [])
        alb_desc = str(album_detail.get("description") or "").strip()
        for tr in album_detail.get("tracks") or []:
            moods.extend(tr.get("moods") or [])
            for tg in tr.get("genres") or []:
                if tg not in genres:
                    genres.append(tg)
        extra = alb_desc

    unique_moods = list(dict.fromkeys(moods))
    parts = [
        f"피지컬 음반({fmt}): {name}",
        f"아티스트: {artist_name}" if artist_name else "",
        f"장르: {_expand_tags(genres, GENRE_KO_MAP)}" if genres else "",
        f"분위기: {_expand_tags(unique_moods, MOOD_KO_MAP)}" if unique_moods else "",
        f"설명: {desc or extra}" if (desc or extra) else "",
    ]
    return "\n".join(p for p in parts if p).strip(), genres, unique_moods


def build_goods_doc(
    item: dict[str, Any],
    artist_genres: list[str] | None = None,
) -> str:
    """굿즈 텍스트 문서 생성"""
    name = str(item.get("name") or "").strip()
    artist_name = str(item.get("artistName") or "").strip()
    category = str(item.get("category") or "").strip()
    desc = str(item.get("description") or "").strip()
    parts = [
        f"굿즈({category}): {name}",
        f"아티스트: {artist_name}" if artist_name else "",
        f"장르: {_expand_tags(artist_genres, GENRE_KO_MAP)}" if artist_genres else "",
        f"설명: {desc}" if desc else "",
    ]
    return "\n".join(p for p in parts if p).strip()


def build_concert_doc(concert: dict[str, Any]) -> tuple[str, list[str]]:
    """공연 텍스트 문서 생성"""
    title = str(concert.get("title") or "").strip()
    ctype = str(concert.get("type") or "").strip()
    genre = str(concert.get("genre") or "").strip()
    venue = concert.get("venue") or {}
    venue_name = str(venue.get("name") or "").strip()
    region = str(venue.get("region") or "").strip()
    desc = str(concert.get("description") or "").strip()

    artists = concert.get("artists") or []
    artist_names = [str(a.get("name") or "").strip() for a in artists if a.get("name")]
    genres = [genre] if genre else []
    for a in artists:
        ag = a.get("genre")
        if ag and ag not in genres:
            genres.append(str(ag))

    parts = [
        f"공연({ctype}): {title}",
        f"출연진: {', '.join(artist_names)}" if artist_names else "",
        f"장르: {_expand_tags(genres, GENRE_KO_MAP)}" if genres else "",
        f"공연장: {venue_name} ({region})" if (venue_name or region) else "",
        f"소개: {desc}" if desc else "",
    ]
    return "\n".join(p for p in parts if p).strip(), genres


async def embed_texts(
    texts: list[str],
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = 32,
) -> np.ndarray:
    """Gemini Embedding API로 여러 텍스트를 배치 임베딩 (L2 정규화 반환)"""
    if not texts:
        return np.zeros((0, constant.REC_TEXT_EMBED_DIM), dtype=np.float32)

    client = get_genai_client()
    vectors: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = [t.strip() or "인디 음악" for t in texts[i : i + batch_size]]
        res = await client.aio.models.embed_content(
            model=constant.REC_EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=constant.REC_TEXT_EMBED_DIM,
            ),
        )
        for emb in res.embeddings or []:
            vectors.append(list(emb.values or []))

    arr = np.asarray(vectors, dtype=np.float32)
    return l2_normalize(arr)


async def embed_query(query: str) -> np.ndarray:
    """자연어 검색 쿼리 1건을 임베딩 (Redis 30분 캐시 적용)"""
    normalized = " ".join(query.strip().lower().split())
    cache_key = f"rec:qemb:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]}"

    try:
        cached = await get_client().get(cache_key)
        if cached:
            return l2_normalize(np.asarray(json.loads(cached), dtype=np.float32))
    except Exception as e:
        logger.debug(f"[쿼리 임베딩 캐시 조회 생략]: {e}")

    arr = await embed_texts([query], task_type="RETRIEVAL_QUERY", batch_size=1)
    vec = arr[0]

    try:
        await get_client().set(
            cache_key,
            json.dumps(vec.tolist()),
            ex=constant.REC_QUERY_CACHE_TTL,
        )
    except Exception as e:
        logger.debug(f"[쿼리 임베딩 캐시 저장 생략]: {e}")

    return vec


def project_text_to_rec(
    text_vectors: np.ndarray,
    projector: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """학습된 IndieRecNet의 text_projector 가중치로 텍스트 벡터 -> 128d 추천 벡터 투영 (NumPy 순전파).

    projector 가중치가 아직 없을 때는 앞 128차원 슬라이스 + L2 정규화로 폴백합니다.
    """
    arr = np.asarray(text_vectors, dtype=np.float32)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]

    if projector and all(k in projector for k in ("w1", "b1", "w2", "b2")):
        h = arr @ projector["w1"].T + projector["b1"]
        if "ln_w" in projector and "ln_b" in projector:
            mean = h.mean(axis=-1, keepdims=True)
            var = h.var(axis=-1, keepdims=True)
            h = (h - mean) / np.sqrt(var + 1e-5) * projector["ln_w"] + projector["ln_b"]
        h = np.maximum(h, 0.0)
        out = h @ projector["w2"].T + projector["b2"]
        out = l2_normalize(out)
    else:
        dim = constant.REC_EMBED_DIM
        if arr.shape[1] >= dim:
            out = l2_normalize(arr[:, :dim])
        else:
            padded = np.zeros((arr.shape[0], dim), dtype=np.float32)
            padded[:, : arr.shape[1]] = arr
            out = l2_normalize(padded)

    return out[0] if single else out
