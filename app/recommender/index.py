"""NumPy 기반 이중 벡터 인덱스(추천 128d + 시맨틱 텍스트 768d) 및 메타데이터 레지스트리."""

import base64
import io
import json
from pathlib import Path
import zlib

import numpy as np

from app.config import constant, logger
from .embedding import l2_normalize, project_text_to_rec
from .schemas import ItemRecord, TargetType

__all__ = ["ItemIndex", "get_item_index"]


class ItemIndex:
    """아이템 임베딩 이중 벡터 검색 및 메타데이터 레지스트리 (NumPy 경량 서빙)."""

    def __init__(self) -> None:
        self._ready: bool = False
        self._version: int = 0
        self._keys: list[str] = []
        self._key_to_pos: dict[str, int] = {}
        self._metadata: dict[str, ItemRecord] = {}
        self._text_matrix: np.ndarray = np.zeros((0, constant.REC_TEXT_EMBED_DIM), dtype=np.float32)
        self._rec_matrix: np.ndarray = np.zeros((0, constant.REC_EMBED_DIM), dtype=np.float32)
        self._projector: dict[str, np.ndarray] | None = None

    @property
    def is_ready(self) -> bool:
        return self._ready and len(self._keys) > 0

    @property
    def version(self) -> int:
        return self._version

    def set_version(self, version: int) -> None:
        self._version = int(version)

    @property
    def item_count(self) -> int:
        return len(self._keys)

    @property
    def all_records(self) -> list[ItemRecord]:
        return [self._metadata[k] for k in self._keys]

    def get_item_meta(self, item_key: str) -> ItemRecord | None:
        return self._metadata.get(item_key)

    def get_rec_vector(self, item_key: str) -> np.ndarray | None:
        pos = self._key_to_pos.get(item_key)
        if pos is None or pos >= len(self._rec_matrix):
            return None
        return self._rec_matrix[pos]

    def get_text_vector(self, item_key: str) -> np.ndarray | None:
        pos = self._key_to_pos.get(item_key)
        if pos is None or pos >= len(self._text_matrix):
            return None
        return self._text_matrix[pos]

    def get_items_by_artist(
        self,
        artist_id: int,
        *,
        target_types: set[TargetType] | None = None,
        active_only: bool = True,
    ) -> list[ItemRecord]:
        results: list[ItemRecord] = []
        for k in self._keys:
            rec = self._metadata[k]
            if active_only and not rec.is_active:
                continue
            if target_types and rec.target_type not in target_types:
                continue
            if artist_id in rec.artist_ids:
                results.append(rec)
        return results

    def build(
        self,
        records: list[ItemRecord],
        text_vectors: np.ndarray,
        rec_vectors: np.ndarray | None = None,
        projector: dict[str, np.ndarray] | None = None,
    ) -> None:
        """아이템 목록과 벡터 행렬로 이중 인덱스를 원자적으로 구축"""
        if len(records) != len(text_vectors):
            raise ValueError("records와 text_vectors의 길이가 일치해야 합니다.")

        self._projector = projector
        text_mat = (
            l2_normalize(np.asarray(text_vectors, dtype=np.float32))
            if len(records) > 0
            else np.zeros((0, constant.REC_TEXT_EMBED_DIM), dtype=np.float32)
        )
        if rec_vectors is not None and len(rec_vectors) == len(records):
            rec_mat = (
                l2_normalize(np.asarray(rec_vectors, dtype=np.float32))
                if len(records) > 0
                else np.zeros((0, constant.REC_EMBED_DIM), dtype=np.float32)
            )
        else:
            rec_mat = (
                project_text_to_rec(text_mat, projector)
                if len(records) > 0
                else np.zeros((0, constant.REC_EMBED_DIM), dtype=np.float32)
            )

        keys = [r.item_key for r in records]
        self._keys = keys
        self._key_to_pos = {k: i for i, k in enumerate(keys)}
        self._metadata = {r.item_key: r for r in records}
        self._text_matrix = text_mat
        self._rec_matrix = rec_mat
        self._ready = len(keys) > 0

    def save(self, artifact_dir: str) -> None:
        """학습/동기화된 이중 인덱스를 .npz + metadata.json으로 저장"""
        base = Path(artifact_dir)
        base.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray] = {
            "text_matrix": self._text_matrix,
            "rec_matrix": self._rec_matrix,
        }
        if self._projector:
            for k, v in self._projector.items():
                arrays[f"proj_{k}"] = np.asarray(v, dtype=np.float32)

        np.savez_compressed(base / "vectors.npz", **arrays)
        meta_payload = {
            "keys": self._keys,
            "items": [self._metadata[k].model_dump() for k in self._keys],
        }
        (base / "metadata.json").write_text(
            json.dumps(meta_payload, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self, artifact_dir: str) -> bool:
        """디스크 아티팩트(.npz + metadata.json) 로드. 파일 없으면 False 반환"""
        base = Path(artifact_dir)
        vec_path = base / "vectors.npz"
        meta_path = base / "metadata.json"
        if not vec_path.exists() or not meta_path.exists():
            return False

        try:
            meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
            records = [ItemRecord(**item) for item in meta_payload.get("items", [])]
            npz = np.load(vec_path)
            text_mat = npz["text_matrix"]
            rec_mat = npz["rec_matrix"]
            projector = {
                k.removeprefix("proj_"): npz[k]
                for k in npz.files
                if k.startswith("proj_")
            } or None
            self.build(records, text_mat, rec_mat, projector=projector)
            logger.info(f"Recommender 인덱스 로드 완료: {self.item_count}건 ({artifact_dir})")
            return True
        except Exception as e:
            logger.error(f"Recommender 인덱스 로드 실패 ({artifact_dir}): {e}")
            return False

    def to_snapshot_b64(self) -> str:
        """K8s 멀티 Pod 공유용 Redis 스냅샷 직렬화 (zlib 압축 + base64)"""
        buf = io.BytesIO()
        arrays: dict[str, np.ndarray] = {
            "text_matrix": self._text_matrix,
            "rec_matrix": self._rec_matrix,
        }
        if self._projector:
            for k, v in self._projector.items():
                arrays[f"proj_{k}"] = np.asarray(v, dtype=np.float32)
        np.savez_compressed(buf, **arrays)

        envelope = {
            "items": [self._metadata[k].model_dump() for k in self._keys],
            "npz_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        }
        compressed = zlib.compress(json.dumps(envelope, ensure_ascii=False).encode("utf-8"), level=6)
        return base64.b64encode(compressed).decode("ascii")

    def load_from_snapshot_b64(self, snapshot_b64: str, *, version: int = 0) -> bool:
        """Redis 공유 스냅샷에서 이중 인덱스 복원"""
        try:
            raw = zlib.decompress(base64.b64decode(snapshot_b64.encode("ascii")))
            envelope = json.loads(raw.decode("utf-8"))
            records = [ItemRecord(**item) for item in envelope.get("items", [])]
            npz_bytes = base64.b64decode(envelope["npz_b64"].encode("ascii"))
            npz = np.load(io.BytesIO(npz_bytes))
            text_mat = npz["text_matrix"]
            rec_mat = npz["rec_matrix"]
            projector = {
                k.removeprefix("proj_"): npz[k]
                for k in npz.files
                if k.startswith("proj_")
            } or None
            self.build(records, text_mat, rec_mat, projector=projector)
            if version > 0:
                self._version = int(version)
            logger.info(
                f"Recommender Redis 스냅샷 로드 완료: {self.item_count}건 (v{self._version})"
            )
            return True
        except Exception as e:
            logger.error(f"Recommender Redis 스냅샷 로드 실패: {e}")
            return False

    def _search_matrix(
        self,
        matrix: np.ndarray,
        query_vector: list[float] | np.ndarray,
        *,
        target_types: set[TargetType] | None = None,
        top_k: int = 50,
        exclude_keys: set[str] | None = None,
        active_only: bool = True,
    ) -> list[tuple[str, float]]:
        if not self.is_ready or len(matrix) == 0:
            return []

        q = l2_normalize(np.asarray(query_vector, dtype=np.float32))
        if q.ndim != 1 or q.shape[0] != matrix.shape[1]:
            return []

        scores = matrix @ q
        order = np.argsort(-scores)
        results: list[tuple[str, float]] = []
        excluded = exclude_keys or set()

        for pos in order:
            key = self._keys[int(pos)]
            if key in excluded:
                continue
            rec = self._metadata[key]
            if active_only and not rec.is_active:
                continue
            if target_types and rec.target_type not in target_types:
                continue
            results.append((key, float(scores[int(pos)])))
            if len(results) >= top_k:
                break
        return results

    def search(
        self,
        query_vector: list[float] | np.ndarray,
        *,
        target_type: TargetType | None = None,
        target_types: set[TargetType] | None = None,
        top_k: int = 50,
        exclude_keys: set[str] | None = None,
        active_only: bool = True,
    ) -> list[tuple[str, float]]:
        """128d 추천 벡터 공간 코사인 유사도 검색"""
        types_filter = target_types or ({target_type} if target_type else None)
        return self._search_matrix(
            self._rec_matrix,
            query_vector,
            target_types=types_filter,
            top_k=top_k,
            exclude_keys=exclude_keys,
            active_only=active_only,
        )

    def search_text(
        self,
        query_vector: list[float] | np.ndarray,
        *,
        target_types: set[TargetType] | None = None,
        top_k: int = 50,
        exclude_keys: set[str] | None = None,
        active_only: bool = True,
    ) -> list[tuple[str, float]]:
        """순수 텍스트 시맨틱 벡터 공간 코사인 유사도 검색"""
        return self._search_matrix(
            self._text_matrix,
            query_vector,
            target_types=target_types,
            top_k=top_k,
            exclude_keys=exclude_keys,
            active_only=active_only,
        )

    def upsert_item(
        self,
        record: ItemRecord,
        text_vector: list[float] | np.ndarray,
        rec_vector: list[float] | np.ndarray | None = None,
    ) -> None:
        """단일 아이템 벡터/메타데이터 즉시 갱신 (신규 등록·장르 전환 즉시 반영)"""
        t_vec = l2_normalize(np.asarray(text_vector, dtype=np.float32))
        r_vec = (
            l2_normalize(np.asarray(rec_vector, dtype=np.float32))
            if rec_vector is not None
            else project_text_to_rec(t_vec, self._projector)
        )

        pos = self._key_to_pos.get(record.item_key)
        if pos is not None:
            self._metadata[record.item_key] = record
            self._text_matrix[pos] = t_vec
            self._rec_matrix[pos] = r_vec
            return

        new_pos = len(self._keys)
        self._keys.append(record.item_key)
        self._key_to_pos[record.item_key] = new_pos
        self._metadata[record.item_key] = record
        self._text_matrix = (
            np.vstack([self._text_matrix, t_vec[None, :]])
            if len(self._text_matrix) > 0
            else t_vec[None, :]
        )
        self._rec_matrix = (
            np.vstack([self._rec_matrix, r_vec[None, :]])
            if len(self._rec_matrix) > 0
            else r_vec[None, :]
        )
        self._ready = True

    def remove_item(self, item_key: str) -> None:
        """단일 아이템 제거"""
        pos = self._key_to_pos.get(item_key)
        if pos is None:
            return
        self._keys.pop(pos)
        self._metadata.pop(item_key, None)
        self._text_matrix = np.delete(self._text_matrix, pos, axis=0)
        self._rec_matrix = np.delete(self._rec_matrix, pos, axis=0)
        self._key_to_pos = {k: i for i, k in enumerate(self._keys)}
        self._ready = len(self._keys) > 0


_item_index: ItemIndex | None = None


def get_item_index() -> ItemIndex:
    """공유 아이템 인덱스 인스턴스 반환"""
    global _item_index
    if _item_index is None:
        _item_index = ItemIndex()
    return _item_index
