"""오프라인 PyTorch IndieRecNet 학습 · 하이퍼파라미터 튜닝 · 평가 · 아티팩트 내보내기.

운영 서빙 컨테이너(ai_server)에는 PyTorch를 설치하지 않고,
로컬 또는 Google Colab에서 이 스크립트로 학습한 뒤 추출된 아티팩트(vectors.npz + metadata.json)를
서빙 서버의 REC_ARTIFACT_DIR에 배포합니다.

실행 예:
    python -m train.indie_rec_net --artifact-dir ./artifacts --epochs 30 --lr 0.001 --embed-dim 128
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from app.recommender.index import ItemIndex
from app.recommender.schemas import ItemRecord


def evaluate_ranking(
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    ground_truth: dict[int, set[int]],
    *,
    k: int = 10,
    exclude_seen: dict[int, set[int]] | None = None,
) -> dict[str, float]:
    """Hit Rate@K, NDCG@K, Recall@K 오프라인 평가 지표 계산 (NumPy, torch 불필요)"""
    hits = 0
    ndcg_sum = 0.0
    recall_sum = 0.0
    evaluated = 0

    for uid, positives in ground_truth.items():
        if not positives or uid >= len(user_vectors):
            continue
        scores = item_vectors @ user_vectors[uid]
        if exclude_seen and uid in exclude_seen:
            for seen_idx in exclude_seen[uid]:
                if 0 <= seen_idx < len(scores):
                    scores[seen_idx] = -1e9

        top_k = np.argsort(-scores)[:k]
        matched_ranks = [rank for rank, idx in enumerate(top_k, start=1) if int(idx) in positives]

        evaluated += 1
        if matched_ranks:
            hits += 1
            dcg = sum(1.0 / math.log2(r + 1) for r in matched_ranks)
            ideal_hits = min(len(positives), k)
            idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
            ndcg_sum += dcg / idcg if idcg > 0 else 0.0
            recall_sum += len(matched_ranks) / len(positives)

    if evaluated == 0:
        return {f"hit_rate@{k}": 0.0, f"ndcg@{k}": 0.0, f"recall@{k}": 0.0}

    return {
        f"hit_rate@{k}": round(hits / evaluated, 4),
        f"ndcg@{k}": round(ndcg_sum / evaluated, 4),
        f"recall@{k}": round(recall_sum / evaluated, 4),
    }


def generate_synthetic_interactions(
    num_items: int,
    item_clusters: list[int],
    *,
    num_users: int = 200,
    seed: int = 42,
) -> list[tuple[int, int, int, bool]]:
    """페르소나 기반 합성 (user_id, pos_item, neg_item, is_hard_negative) 학습 트리플 생성"""
    rng = np.random.default_rng(seed)
    clusters = np.asarray(item_clusters)
    unique_clusters = np.unique(clusters)
    triples: list[tuple[int, int, int, bool]] = []

    for u in range(num_users):
        fav_cluster = int(rng.choice(unique_clusters))
        disliked_cluster = int(
            rng.choice([c for c in unique_clusters if c != fav_cluster] or [fav_cluster])
        )
        pos_pool = np.where(clusters == fav_cluster)[0]
        hard_neg_pool = np.where(clusters == disliked_cluster)[0]
        easy_neg_pool = np.where(clusters != fav_cluster)[0]
        if len(pos_pool) == 0 or len(easy_neg_pool) == 0:
            continue

        for _ in range(12):
            pos_item = int(rng.choice(pos_pool))
            is_hard = bool(len(hard_neg_pool) > 0 and rng.random() < 0.35)
            neg_item = int(rng.choice(hard_neg_pool if is_hard else easy_neg_pool))
            triples.append((u, pos_item, neg_item, is_hard))

    return triples


def train_indie_rec_net(
    records: list[ItemRecord],
    text_vectors: np.ndarray,
    interactions: list[tuple[int, int, int, bool]],
    *,
    num_users: int,
    embed_dim: int = 128,
    epochs: int = 25,
    lr: float = 1e-3,
    dropout: float = 0.2,
    hard_neg_weight: float = 2.5,
    batch_size: int = 128,
    artifact_dir: str | None = None,
) -> dict[str, Any]:
    """PyTorch IndieRecNet 학습 및 NumPy 서빙 아티팩트 추출."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as e:
        raise RuntimeError(
            "오프라인 학습에는 PyTorch가 필요합니다. 로컬/Colab 환경에서 `pip install torch` 후 실행하세요."
        ) from e

    num_items, text_dim = text_vectors.shape

    class IndieRecNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(text_dim, 256)
            self.ln1 = nn.LayerNorm(256)
            self.drop = nn.Dropout(dropout)
            self.fc2 = nn.Linear(256, embed_dim)
            self.item_behavior = nn.Embedding(num_items, embed_dim)
            self.user_embedding = nn.Embedding(num_users, embed_dim)
            self.alpha_logit = nn.Parameter(torch.tensor(0.0))

            nn.init.normal_(self.item_behavior.weight, std=0.05)
            nn.init.normal_(self.user_embedding.weight, std=0.05)

        def project_text(self, x: torch.Tensor) -> torch.Tensor:
            h = self.drop(F.relu(self.ln1(self.fc1(x))))
            return F.normalize(self.fc2(h), p=2, dim=-1)

        def item_vectors(self, item_ids: torch.Tensor, text_mat: torch.Tensor) -> torch.Tensor:
            t_feat = self.project_text(text_mat[item_ids])
            b_feat = F.normalize(self.item_behavior(item_ids), p=2, dim=-1)
            alpha = torch.sigmoid(self.alpha_logit)
            return F.normalize(alpha * t_feat + (1.0 - alpha) * b_feat, p=2, dim=-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IndieRecNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    text_tensor = torch.tensor(text_vectors, dtype=torch.float32, device=device)

    u_arr = np.array([x[0] for x in interactions], dtype=np.int64)
    p_arr = np.array([x[1] for x in interactions], dtype=np.int64)
    n_arr = np.array([x[2] for x in interactions], dtype=np.int64)
    w_arr = np.array(
        [hard_neg_weight if x[3] else 1.0 for x in interactions],
        dtype=np.float32,
    )

    last_loss = 0.0
    model.train()
    for _ in range(epochs):
        perm = np.random.permutation(len(interactions))
        epoch_loss = 0.0
        steps = 0
        for start in range(0, len(interactions), batch_size):
            sl = perm[start : start + batch_size]
            u = torch.tensor(u_arr[sl], device=device)
            p = torch.tensor(p_arr[sl], device=device)
            n = torch.tensor(n_arr[sl], device=device)
            w = torch.tensor(w_arr[sl], device=device)

            u_vec = F.normalize(model.user_embedding(u), p=2, dim=-1)
            p_vec = model.item_vectors(p, text_tensor)
            n_vec = model.item_vectors(n, text_tensor)

            pos_score = (u_vec * p_vec).sum(dim=-1)
            neg_score = (u_vec * n_vec).sum(dim=-1)
            loss = (-F.logsigmoid(pos_score - neg_score) * w).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            steps += 1
        last_loss = epoch_loss / max(steps, 1)

    model.eval()
    with torch.no_grad():
        all_ids = torch.arange(num_items, device=device)
        final_item_vecs = model.item_vectors(all_ids, text_tensor).cpu().numpy()
        learned_alpha = float(torch.sigmoid(model.alpha_logit).item())
        projector = {
            "w1": model.fc1.weight.cpu().numpy(),
            "b1": model.fc1.bias.cpu().numpy(),
            "ln_w": model.ln1.weight.cpu().numpy(),
            "ln_b": model.ln1.bias.cpu().numpy(),
            "w2": model.fc2.weight.cpu().numpy(),
            "b2": model.fc2.bias.cpu().numpy(),
        }

    if artifact_dir:
        idx = ItemIndex()
        idx.build(records, text_vectors, final_item_vecs, projector=projector)
        idx.save(artifact_dir)

    return {
        "final_loss": round(last_loss, 4),
        "learned_alpha": round(learned_alpha, 4),
        "num_items": num_items,
        "embed_dim": embed_dim,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="INDIG IndieRecNet 오프라인 학습")
    parser.add_argument("--artifact-dir", default="/tmp/indig_recommender")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=128)
    args = parser.parse_args()

    idx = ItemIndex()
    if not idx.load(args.artifact_dir):
        raise SystemExit(
            f"아티팩트를 찾을 수 없습니다: {args.artifact_dir} (먼저 카탈로그 동기화를 실행하세요)"
        )

    records = idx.all_records
    # 장르 기반 합성 클러스터 생성 (실제 행동 로그가 부족한 초기 학습용)
    genre_to_id: dict[str, int] = {}
    clusters: list[int] = []
    for r in records:
        g = r.genres[0] if r.genres else "ETC"
        clusters.append(genre_to_id.setdefault(g, len(genre_to_id)))

    triples = generate_synthetic_interactions(len(records), clusters, num_users=200)
    summary = train_indie_rec_net(
        records,
        idx._text_matrix,
        triples,
        num_users=200,
        embed_dim=args.embed_dim,
        epochs=args.epochs,
        lr=args.lr,
        artifact_dir=args.artifact_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
