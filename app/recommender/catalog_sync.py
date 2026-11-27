"""백엔드 공개 v2 API에서 아티스트·앨범/트랙·피지컬·굿즈·공연을 수집해 이중 인덱스를 동기화."""

import asyncio
import hashlib
from typing import Any

import numpy as np

from app.config import constant, logger, setting
from app.server.http import get_backend_client
from . import store
from .embedding import (
    build_artist_doc,
    build_concert_doc,
    build_goods_doc,
    build_physical_album_doc,
    embed_texts,
    project_text_to_rec,
)
from .index import get_item_index
from .schemas import ItemRecord, TargetType


def compute_embed_hash(text_doc: str) -> str:
    return hashlib.sha256(text_doc.strip().encode("utf-8")).hexdigest()[:16]

__all__ = [
    "fetch_all_catalog_records",
    "sync_catalog_to_index",
    "upsert_single_item_from_backend",
]


def _backend_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if setting.BACKEND_HOST.strip():
        headers["Host"] = setting.BACKEND_HOST.strip()
    return headers


async def _fetch_paginated(path: str, *, page_size: int = 50, max_pages: int = 20) -> list[dict[str, Any]]:
    """v2 PageResult({items, page, size, hasNext}) 엔드포인트 전체 순회"""
    base = setting.BACKEND_URL.rstrip("/")
    client = get_backend_client()
    headers = _backend_headers()
    collected: list[dict[str, Any]] = []

    for page in range(max_pages):
        res = await client.get(
            f"{base}{path}",
            params={"page": page, "size": page_size},
            headers=headers,
            timeout=constant.BACKEND_HTTP_TIMEOUT_SECONDS,
        )
        if res.status_code != 200:
            logger.warning(f"[카탈로그 동기화] {path} page={page} status={res.status_code}")
            break
        body = res.json()
        if not isinstance(body, dict) or body.get("isSuccess") is False:
            break
        result = body.get("result") or {}
        items = result.get("items") or []
        collected.extend(items)
        if not result.get("hasNext"):
            break
    return collected


async def _fetch_detail(path: str) -> dict[str, Any] | None:
    base = setting.BACKEND_URL.rstrip("/")
    try:
        res = await get_backend_client().get(
            f"{base}{path}",
            headers=_backend_headers(),
            timeout=constant.BACKEND_HTTP_TIMEOUT_SECONDS,
        )
        if res.status_code != 200:
            return None
        body = res.json()
        if isinstance(body, dict) and body.get("isSuccess") is not False:
            result = body.get("result")
            return result if isinstance(result, dict) else None
    except Exception as e:
        logger.debug(f"[상세 조회 스킵] {path}: {e}")
    return None


async def fetch_all_catalog_records() -> list[ItemRecord]:
    """백엔드 공개 API에서 4개 카테고리(아티스트·피지컬·굿즈·공연) 메타데이터 수집"""
    artists, albums, physicals, goods, concerts = await asyncio.gather(
        _fetch_paginated("/api/catalog/artists"),
        _fetch_paginated("/api/catalog/albums"),
        _fetch_paginated("/api/catalog/physical-albums"),
        _fetch_paginated("/api/catalog/goods"),
        _fetch_paginated("/api/concerts"),
    )

    # 앨범 상세(트랙별 tempo/moods/emotions/styles 포함) 병렬 조회
    sem = asyncio.Semaphore(8)

    async def _bounded_detail(path: str) -> dict[str, Any] | None:
        async with sem:
            return await _fetch_detail(path)

    album_details_list = await asyncio.gather(
        *(_bounded_detail(f"/api/catalog/albums/{a['id']}") for a in albums if a.get("id"))
    )
    album_by_id: dict[int, dict[str, Any]] = {}
    albums_by_artist: dict[int, list[dict[str, Any]]] = {}
    for alb, detail in zip(albums, album_details_list):
        merged = {**alb, **(detail or {})}
        aid = merged.get("id")
        if aid is not None:
            album_by_id[int(aid)] = merged
        artist_id = merged.get("artistId")
        if artist_id is not None:
            albums_by_artist.setdefault(int(artist_id), []).append(merged)

    artist_details_list = await asyncio.gather(
        *(_bounded_detail(f"/api/catalog/artists/{a['id']}") for a in artists if a.get("id"))
    )

    records: list[ItemRecord] = []
    artist_genres_map: dict[int, list[str]] = {}

    # 1. 아티스트
    for art, detail in zip(artists, artist_details_list):
        merged = {**art, **(detail or {})}
        aid = int(merged["id"])
        genres = [str(g) for g in (merged.get("genres") or [])]
        artist_genres_map[aid] = genres
        doc, moods = build_artist_doc(merged, albums_by_artist.get(aid))
        records.append(
            ItemRecord(
                item_key=f"ARTIST:{aid}",
                target_type="ARTIST",
                item_id=aid,
                artist_ids=[aid],
                artist_names=[str(merged.get("name") or "")],
                genres=genres,
                moods=moods,
                is_active=True,
                popularity_prior=float(merged.get("followerCount") or 0.0),
                text_doc=doc,
                payload=art,
            )
        )

    # 2. 피지컬 앨범 (LP/CD/CASSETTE)
    for item in physicals:
        pid = int(item["id"])
        aid = int(item["artistId"]) if item.get("artistId") is not None else None
        alb_id = int(item["albumId"]) if item.get("albumId") is not None else None
        doc, genres, moods = build_physical_album_doc(item, album_by_id.get(alb_id or -1))
        if not genres and aid is not None:
            genres = list(artist_genres_map.get(aid, []))
        records.append(
            ItemRecord(
                item_key=f"PHYSICAL_ALBUM:{pid}",
                target_type="PHYSICAL_ALBUM",
                item_id=pid,
                artist_ids=[aid] if aid is not None else [],
                artist_names=[str(item.get("artistName") or "")] if item.get("artistName") else [],
                genres=genres,
                moods=moods,
                is_active=not bool(item.get("soldOut", False)),
                released_at=item.get("releasedAt"),
                text_doc=doc,
                payload=item,
            )
        )

    # 3. 굿즈
    for item in goods:
        gid = int(item["id"])
        aid = int(item["artistId"]) if item.get("artistId") is not None else None
        genres = list(artist_genres_map.get(aid, [])) if aid is not None else []
        doc = build_goods_doc(item, genres)
        records.append(
            ItemRecord(
                item_key=f"GOODS:{gid}",
                target_type="GOODS",
                item_id=gid,
                artist_ids=[aid] if aid is not None else [],
                artist_names=[str(item.get("artistName") or "")] if item.get("artistName") else [],
                genres=genres,
                is_active=not bool(item.get("soldOut", False)),
                text_doc=doc,
                payload=item,
            )
        )

    # 4. 공연
    for concert in concerts:
        cid = int(concert["id"])
        lineup = concert.get("artists") or []
        aids = [int(a["id"]) for a in lineup if a.get("id") is not None]
        anames = [str(a["name"]) for a in lineup if a.get("name")]
        doc, genres = build_concert_doc(concert)
        venue = concert.get("venue") or {}
        sale_status = str(concert.get("saleStatus") or "ON_SALE").upper()
        records.append(
            ItemRecord(
                item_key=f"CONCERT:{cid}",
                target_type="CONCERT",
                item_id=cid,
                artist_ids=aids,
                artist_names=anames,
                genres=genres,
                region=venue.get("region"),
                is_active=sale_status != "CLOSED",
                released_at=concert.get("startsAt"),
                text_doc=doc,
                payload=concert,
            )
        )

    return records


async def sync_catalog_to_index(*, save_artifact: bool = True) -> int:
    """백엔드 카탈로그를 수집해 embed_hash 증분 임베딩 후 ItemIndex 및 Redis 스냅샷 갱신"""
    records = await fetch_all_catalog_records()
    if not records:
        logger.warning("[카탈로그 동기화] 수집된 아이템이 0건입니다.")
        return 0

    for r in records:
        if not r.embed_hash:
            r.embed_hash = compute_embed_hash(r.text_doc)

    idx = get_item_index()
    reused_text: dict[int, np.ndarray] = {}
    reused_rec: dict[int, np.ndarray] = {}
    to_embed_indices: list[int] = []
    to_embed_docs: list[str] = []

    for i, r in enumerate(records):
        old_meta = idx.get_item_meta(r.item_key)
        old_t = idx.get_text_vector(r.item_key)
        old_r = idx.get_rec_vector(r.item_key)
        if old_meta is not None and old_meta.embed_hash == r.embed_hash and old_t is not None:
            reused_text[i] = old_t
            if old_r is not None:
                reused_rec[i] = old_r
        else:
            to_embed_indices.append(i)
            to_embed_docs.append(r.text_doc)

    new_text_vecs = (
        await embed_texts(to_embed_docs, task_type="RETRIEVAL_DOCUMENT")
        if to_embed_docs
        else np.zeros((0, constant.REC_TEXT_EMBED_DIM), dtype=np.float32)
    )

    text_dim = (
        new_text_vecs.shape[1]
        if len(new_text_vecs) > 0
        else (next(iter(reused_text.values())).shape[0] if reused_text else constant.REC_TEXT_EMBED_DIM)
    )
    final_text = np.zeros((len(records), text_dim), dtype=np.float32)
    final_rec = np.zeros((len(records), constant.REC_EMBED_DIM), dtype=np.float32)

    for i, vec in reused_text.items():
        final_text[i] = vec
        final_rec[i] = (
            reused_rec[i]
            if i in reused_rec
            else project_text_to_rec(vec, idx._projector)
        )
    for pos, orig_i in enumerate(to_embed_indices):
        t_vec = new_text_vecs[pos]
        final_text[orig_i] = t_vec
        final_rec[orig_i] = project_text_to_rec(t_vec, idx._projector)

    idx.build(records, final_text, final_rec, projector=idx._projector)
    if save_artifact:
        idx.save(setting.REC_ARTIFACT_DIR)
        new_ver = await store.save_index_snapshot(idx.to_snapshot_b64())
        if new_ver > 0:
            idx.set_version(new_ver)

    logger.info(
        f"[카탈로그 증분 동기화 완료] 총 {len(records)}건 (신규/변경 임베딩 {len(to_embed_indices)}건, 재사용 {len(reused_text)}건, v{idx.version})"
    )
    return len(records)


async def upsert_single_item_from_backend(
    target_type: TargetType,
    item_id: int,
    *,
    save_artifact: bool = True,
) -> ItemRecord | None:
    """신규 등록·수정된 단일 아이템 1건만 백엔드에서 조회해 즉시 임베딩 및 인덱스 반영"""
    record: ItemRecord | None = None

    if target_type == "ARTIST":
        detail = await _fetch_detail(f"/api/catalog/artists/{item_id}")
        if not detail:
            return None
        genres = [str(g) for g in (detail.get("genres") or [])]
        doc, moods = build_artist_doc(detail, detail.get("albums"))
        record = ItemRecord(
            item_key=f"ARTIST:{item_id}",
            target_type="ARTIST",
            item_id=item_id,
            artist_ids=[item_id],
            artist_names=[str(detail.get("name") or "")],
            genres=genres,
            moods=moods,
            is_active=True,
            popularity_prior=float(detail.get("followerCount") or 0.0),
            text_doc=doc,
            payload={
                "id": item_id,
                "name": detail.get("name"),
                "genres": genres,
                "debutYear": detail.get("debutYear"),
                "profileUrl": detail.get("profileUrl"),
            },
        )

    elif target_type in ("PHYSICAL_ALBUM", "GOODS"):
        detail = await _fetch_detail(f"/api/catalog/products/{item_id}")
        if not detail:
            return None
        aid = int(detail["artistId"]) if detail.get("artistId") is not None else None
        options = detail.get("options") or []
        sold_out = all(bool(o.get("soldOut", False)) for o in options) if options else False
        if target_type == "PHYSICAL_ALBUM":
            alb_id = detail.get("albumId")
            alb_detail = await _fetch_detail(f"/api/catalog/albums/{alb_id}") if alb_id else None
            doc, genres, moods = build_physical_album_doc(detail, alb_detail)
        else:
            doc = build_goods_doc(detail)
            genres, moods = [], []
        record = ItemRecord(
            item_key=f"{target_type}:{item_id}",
            target_type=target_type,
            item_id=item_id,
            artist_ids=[aid] if aid is not None else [],
            artist_names=[str(detail.get("artistName") or "")] if detail.get("artistName") else [],
            genres=genres,
            moods=moods,
            is_active=not sold_out,
            released_at=detail.get("releasedAt"),
            text_doc=doc,
            payload=detail,
        )

    elif target_type == "CONCERT":
        detail = await _fetch_detail(f"/api/concerts/{item_id}")
        if not detail:
            return None
        lineup = detail.get("artists") or []
        aids = [int(a["id"]) for a in lineup if a.get("id") is not None]
        anames = [str(a["name"]) for a in lineup if a.get("name")]
        doc, genres = build_concert_doc(detail)
        venue = detail.get("venue") or {}
        sale_status = str(detail.get("saleStatus") or "ON_SALE").upper()
        record = ItemRecord(
            item_key=f"CONCERT:{item_id}",
            target_type="CONCERT",
            item_id=item_id,
            artist_ids=aids,
            artist_names=anames,
            genres=genres,
            region=venue.get("region"),
            is_active=sale_status != "CLOSED",
            released_at=detail.get("startsAt"),
            text_doc=doc,
            payload=detail,
        )

    if record is None:
        return None

    record.embed_hash = compute_embed_hash(record.text_doc)
    text_vecs = await embed_texts([record.text_doc], task_type="RETRIEVAL_DOCUMENT", batch_size=1)
    idx = get_item_index()
    idx.upsert_item(record, text_vecs[0])
    if save_artifact:
        idx.save(setting.REC_ARTIFACT_DIR)
        new_ver = await store.save_index_snapshot(idx.to_snapshot_b64())
        if new_ver > 0:
            idx.set_version(new_ver)
    logger.info(f"[단일 아이템 즉시 반영 완료] {record.item_key} (v{idx.version})")
    return record

