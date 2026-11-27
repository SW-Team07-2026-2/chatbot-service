import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.chatbot import skill as chatbot_skill
from app.config import constant
from app.main import app
from app.recommender import embedding, index, ingest_client, service, store
from app.recommender.schemas import FeedbackRequest, ItemRecord
from app.server import redis as server_redis
from tests.conftest import FakeRedis
from train.indie_rec_net import evaluate_ranking, generate_synthetic_interactions


def _unit(dim: int, hot_idx: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[hot_idx] = 1.0
    return v


def _build_demo_index() -> index.ItemIndex:
    """테스트용 소규모 이중 인덱스:
    - 아티스트 1 (한로로, 락/감성, 우울/회복 무드)
    - 아티스트 2 (재즈밴드, 재즈)
    - 아티스트 3 (락/재즈 전환 아티스트 A: 텍스트는 락, 128d 행동벡터는 재즈쪽)
    - 피지컬 10 (아티스트 3의 락 앨범) / 피지컬 11 (아티스트 3의 재즈 앨범)
    - 굿즈 101 (한로로 티셔츠)
    - 공연 201 (한로로 단독공연) / 공연 202 (재즈밴드 공연)
    """
    t_dim = constant.REC_TEXT_EMBED_DIM
    r_dim = constant.REC_EMBED_DIM

    records = [
        ItemRecord(
            item_key="ARTIST:1",
            target_type="ARTIST",
            item_id=1,
            artist_ids=[1],
            artist_names=["한로로"],
            genres=["INDIE_ROCK", "EMO"],
            moods=["DARK", "WARM", "CALM"],
            popularity_prior=100.0,
            text_doc="아티스트: 한로로 우울 회복 안정 비 잔잔한 인디록",
            payload={"id": 1, "name": "한로로", "genres": ["INDIE_ROCK", "EMO"]},
        ),
        ItemRecord(
            item_key="ARTIST:2",
            target_type="ARTIST",
            item_id=2,
            artist_ids=[2],
            artist_names=["재즈밴드"],
            genres=["JAZZ"],
            moods=["CALM", "GROOVY"],
            popularity_prior=80.0,
            text_doc="아티스트: 재즈밴드 재즈 스윙 차분한",
            payload={"id": 2, "name": "재즈밴드", "genres": ["JAZZ"]},
        ),
        ItemRecord(
            item_key="ARTIST:3",
            target_type="ARTIST",
            item_id=3,
            artist_ids=[3],
            artist_names=["전환밴드"],
            genres=["INDIE_ROCK"],
            moods=["ENERGETIC"],
            popularity_prior=60.0,
            text_doc="아티스트: 전환밴드 락 밴드",
            payload={"id": 3, "name": "전환밴드", "genres": ["INDIE_ROCK"]},
        ),
        ItemRecord(
            item_key="PHYSICAL_ALBUM:10",
            target_type="PHYSICAL_ALBUM",
            item_id=10,
            artist_ids=[3],
            artist_names=["전환밴드"],
            genres=["INDIE_ROCK"],
            text_doc="피지컬 음반: 전환밴드 락 앨범",
            payload={"id": 10, "name": "전환밴드 1집(락)", "artistId": 3},
        ),
        ItemRecord(
            item_key="PHYSICAL_ALBUM:11",
            target_type="PHYSICAL_ALBUM",
            item_id=11,
            artist_ids=[3],
            artist_names=["전환밴드"],
            genres=["JAZZ"],
            text_doc="피지컬 음반: 전환밴드 재즈 앨범",
            payload={"id": 11, "name": "전환밴드 2집(재즈)", "artistId": 3},
        ),
        ItemRecord(
            item_key="GOODS:101",
            target_type="GOODS",
            item_id=101,
            artist_ids=[1],
            artist_names=["한로로"],
            genres=["INDIE_ROCK", "EMO"],
            text_doc="굿즈: 한로로 티셔츠",
            payload={"id": 101, "name": "한로로 티셔츠", "artistId": 1},
        ),
        ItemRecord(
            item_key="CONCERT:201",
            target_type="CONCERT",
            item_id=201,
            artist_ids=[1],
            artist_names=["한로로"],
            genres=["INDIE_ROCK"],
            region="SEOUL",
            text_doc="공연: 한로로 서울 단독공연",
            payload={"id": 201, "title": "한로로 단독공연"},
        ),
        ItemRecord(
            item_key="CONCERT:202",
            target_type="CONCERT",
            item_id=202,
            artist_ids=[2],
            artist_names=["재즈밴드"],
            genres=["JAZZ"],
            region="BUSAN",
            text_doc="공연: 재즈밴드 부산 공연",
            payload={"id": 202, "title": "재즈밴드 부산 공연"},
        ),
    ]

    # 축 0: 한로로/감성록, 축 1: 재즈, 축 2: 일반 락
    text_vecs = np.stack(
        [
            _unit(t_dim, 0),  # ARTIST:1 (한로로)
            _unit(t_dim, 1),  # ARTIST:2 (재즈밴드)
            _unit(t_dim, 2),  # ARTIST:3 (텍스트는 락)
            _unit(t_dim, 2),  # PHYSICAL:10 (락 앨범)
            _unit(t_dim, 1),  # PHYSICAL:11 (재즈 앨범)
            _unit(t_dim, 0),  # GOODS:101 (한로로 티셔츠)
            _unit(t_dim, 0),  # CONCERT:201 (한로로 공연)
            _unit(t_dim, 1),  # CONCERT:202 (재즈 공연)
        ]
    )
    rec_vecs = np.stack(
        [
            _unit(r_dim, 0),  # ARTIST:1
            _unit(r_dim, 1),  # ARTIST:2
            embedding.l2_normalize(0.2 * _unit(r_dim, 2) + 0.8 * _unit(r_dim, 1)),  # ARTIST:3 (행동은 재즈쪽!)
            _unit(r_dim, 2),  # PHYSICAL:10 (락 앨범)
            _unit(r_dim, 1),  # PHYSICAL:11 (재즈 앨범)
            _unit(r_dim, 0),  # GOODS:101
            _unit(r_dim, 0),  # CONCERT:201
            _unit(r_dim, 1),  # CONCERT:202
        ]
    )

    idx = index.ItemIndex()
    idx.build(records, text_vecs, rec_vecs)
    return idx


@pytest.fixture(autouse=True)
def _setup_recommender_env(monkeypatch):
    fake_redis = FakeRedis()
    demo_idx = _build_demo_index()

    async def noop_ping():
        return True

    async def noop_close():
        return None

    monkeypatch.setattr(server_redis, "ping", noop_ping)
    monkeypatch.setattr(server_redis, "close", noop_close)
    monkeypatch.setattr(store, "get_client", lambda: fake_redis)
    monkeypatch.setattr(embedding, "get_client", lambda: fake_redis)
    monkeypatch.setattr(service, "get_item_index", lambda: demo_idx)
    monkeypatch.setattr(index, "_item_index", demo_idx)
    return fake_redis, demo_idx


async def test_cross_category_transfer_from_goods_to_concerts():
    """한로로 굿즈를 집중 조회하면 공연 탭에서도 한로로 공연이 최상단에 추천되어야 함"""
    for _ in range(2):
        await service.apply_feedback(
            "u-hanroro",
            FeedbackRequest(target_type="GOODS", target_id=101, metric_type="DWELL", metric_value=45),
        )

    page = await service.get_recommendations(category="concerts", user_uuid="u-hanroro", size=5)
    assert len(page.items) >= 2
    assert page.items[0].item_id == 201  # 한로로 공연이 1위


async def test_genre_aligned_album_separation_within_same_artist():
    """전환밴드의 락/재즈 앨범이 재즈 팬에게는 재즈 앨범(11) 우선으로 추천되어야 함"""
    # 재즈밴드 팔로우 + 전환밴드 상세 조회
    await service.apply_feedback(
        "u-jazz",
        FeedbackRequest(target_type="ARTIST", target_id=2, metric_type="FOLLOW"),
    )
    await service.apply_feedback(
        "u-jazz",
        FeedbackRequest(target_type="ARTIST", target_id=3, metric_type="DETAIL_VIEW"),
    )

    page = await service.get_recommendations(category="physical-albums", user_uuid="u-jazz", size=5)
    assert page.items[0].item_id == 11  # 재즈 앨범이 락 앨범(10)보다 위


async def test_sticky_revisit_vs_impression_fatigue():
    """반복 진입(revisit)한 공연은 새로고침해도 유지되고, 예매 완료 공연은 제외됨"""
    await service.apply_feedback(
        "u-sticky",
        FeedbackRequest(target_type="CONCERT", target_id=201, metric_type="DETAIL_VIEW"),
    )
    await service.apply_feedback(
        "u-sticky",
        FeedbackRequest(target_type="CONCERT", target_id=201, metric_type="DWELL", metric_value=60),
    )

    p1 = await service.get_recommendations(category="concerts", user_uuid="u-sticky", size=5)
    p2 = await service.get_recommendations(category="concerts", user_uuid="u-sticky", size=5)
    assert p1.items[0].item_id == 201
    assert p2.items[0].item_id == 201

    # 예매 완료 시 공연 201은 제외됨
    await service.apply_feedback(
        "u-sticky",
        FeedbackRequest(target_type="CONCERT", target_id=201, metric_type="BOOKING"),
    )
    p3 = await service.get_recommendations(category="concerts", user_uuid="u-sticky", size=5)
    assert all(it.item_id != 201 for it in p3.items)


async def test_hybrid_semantic_search_finds_behavior_shifted_artist(monkeypatch):
    """텍스트에 '재즈'가 없어도 행동 클러스터가 재즈쪽인 전환밴드(3)가 자연어 '재즈' 검색에 포함"""
    async def fake_embed_query(query: str) -> np.ndarray:
        return _unit(constant.REC_TEXT_EMBED_DIM, 1)  # 재즈 축

    monkeypatch.setattr(service, "embed_query", fake_embed_query)

    res = await service.semantic_search(query="재즈 차분한", categories=["artists"], size=5)
    ids = [it.item_id for it in res.items]
    assert ids[0] == 2  # 순수 재즈밴드 1위
    assert 3 in ids  # 행동 보정으로 전환밴드도 노출


async def test_chatbot_semantic_search_skill_integration(monkeypatch):
    async def fake_embed_query(query: str) -> np.ndarray:
        return _unit(constant.REC_TEXT_EMBED_DIM, 0)  # 한로로/우울/회복 축

    monkeypatch.setattr(service, "embed_query", fake_embed_query)

    res = await chatbot_skill.semantic_search_catalog(
        query="우울, 회복, 안정, 비",
        categories=["artists"],
        size=3,
    )
    assert res["items"][0]["item_id"] == 1
    assert res["items"][0]["data"]["name"] == "한로로"


def test_recommender_http_endpoints():
    with TestClient(app) as client:
        st = client.get("/api/recommender/status")
        assert st.status_code == 200
        assert st.json()["status"] == "READY"
        assert st.json()["item_count"] == 8

        pop = client.get("/api/recommender/popular/artists")
        assert pop.status_code == 200
        assert len(pop.json()["items"]) == 3

        sim = client.get("/api/recommender/similar/ARTIST/2")
        assert sim.status_code == 200
        assert len(sim.json()["items"]) >= 1

        onb = client.get("/api/recommender/onboarding/similar?artistId=2&exclude=1&size=2")
        assert onb.status_code == 200
        assert all(it["id"] not in (1, 2) for it in onb.json()["items"])

        # 개인화 추천 비로그인 401 / 로그인 200
        assert client.get("/api/recommender/concerts").status_code == 401
        rec = client.get("/api/recommender/concerts", headers={"X-User-Id": "u-http"})
        assert rec.status_code == 200


def test_offline_ranking_evaluation_and_ingest_hmac(monkeypatch):
    # 오프라인 평가 지표(Hit Rate@K, NDCG@K, Recall@K) 검증
    u_vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    i_vecs = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]], dtype=np.float32)
    metrics = evaluate_ranking(u_vecs, i_vecs, {0: {0}, 1: {1}}, k=2)
    assert metrics["hit_rate@2"] == 1.0
    assert metrics["ndcg@2"] == 1.0

    triples = generate_synthetic_interactions(6, [0, 0, 1, 1, 2, 2], num_users=10)
    assert len(triples) > 0

    # HMAC 서명 헤더 생성 검증
    monkeypatch.setattr(ingest_client.setting, "INGEST_MEMBER_KEY_ID", "k1")
    monkeypatch.setattr(ingest_client.setting, "INGEST_MEMBER_SECRET", "secret123")
    headers = ingest_client.build_ingest_headers("GET", "/ingest/members/1/metrics")
    assert headers["X-Ingest-Key-Id"] == "k1"
    assert len(headers["X-Ingest-Signature"]) == 64


def test_recommender_admin_endpoints_and_secret_guard(monkeypatch):
    monkeypatch.setattr(constant, "REC_AUTO_SYNC_ON_STARTUP", False)
    monkeypatch.setattr(service.setting, "RECOMMENDER_ADMIN_SECRET", "admin-secret-xyz")

    with TestClient(app) as client:
        # 시크릿 없이 호출 -> 403
        assert client.get("/api/recommender/admin/status").status_code == 403

        headers = {"X-Admin-Secret": "admin-secret-xyz"}
        st = client.get("/api/recommender/admin/status", headers=headers)
        assert st.status_code == 200
        body = st.json()
        assert body["counts_by_type"]["ARTIST"] == 3
        assert body["scheduler_running"] is True

        # 단일 아이템 제거
        rm = client.delete("/api/recommender/admin/items/GOODS/101", headers=headers)
        assert rm.status_code == 200
        assert rm.json()["removed"] is True

        # 캐시 초기화 & Write-Back 트리거
        assert client.post("/api/recommender/admin/cache/clear", headers=headers).status_code == 200
        wb = client.post("/api/recommender/admin/flush-writeback", headers=headers)
        assert wb.status_code == 200
        assert wb.json()["ok"] is True


async def test_redis_snapshot_roundtrip_and_multi_pod_version_sync(_setup_recommender_env):
    """Pod A가 Redis에 스냅샷을 올리면 Pod B가 버전 변화 감지로 즉시 핫 리로드해야 함"""
    _, demo_idx = _setup_recommender_env
    b64 = demo_idx.to_snapshot_b64()
    new_ver = await store.save_index_snapshot(b64)
    assert new_ver == 1

    # 새로운 빈 Pod B의 ItemIndex 시뮬레이션
    pod_b_idx = index.ItemIndex()
    assert pod_b_idx.is_ready is False

    snap = await store.load_index_snapshot()
    assert snap is not None
    ver, loaded_b64 = snap
    assert pod_b_idx.load_from_snapshot_b64(loaded_b64, version=ver) is True
    assert pod_b_idx.item_count == demo_idx.item_count
    assert pod_b_idx.version == 1


def test_v2_chip_filters_on_popular_and_personalized():
    with TestClient(app) as client:
        # 장르 칩 필터 (JAZZ만)
        res = client.get("/api/recommender/popular/artists?genre=JAZZ")
        assert res.status_code == 200
        ids = [it["item_id"] for it in res.json()["items"]]
        assert ids == [2]

        # 공연 지역 칩 필터 (SEOUL만)
        res_c = client.get(
            "/api/recommender/concerts?region=SEOUL",
            headers={"X-User-Id": "u-filter"},
        )
        assert res_c.status_code == 200
        assert [it["item_id"] for it in res_c.json()["items"]] == [201]

        # 잘못된 가격 범위 -> 400
        bad = client.get("/api/recommender/popular/goods?minPrice=50000&maxPrice=10000")
        assert bad.status_code == 400


async def test_rebuild_user_taste_from_backend_metrics(monkeypatch):
    """Redis·임베딩 백업이 모두 비었을 때 백엔드 행동 기록(metrics)으로 취향 좌표를 자동 재구성"""
    async def no_embedding(user_id: str):
        return None

    async def fake_metrics(user_id: str, *, limit: int = 100):
        return [
            {"targetType": "GOODS", "targetId": 101, "metricType": "DWELL", "metricValue": 45},
            {"targetType": "ARTIST", "targetId": 1, "metricType": "FOLLOW", "metricValue": 1},
        ]

    monkeypatch.setattr(ingest_client, "fetch_member_embedding", no_embedding)
    monkeypatch.setattr(ingest_client, "fetch_member_metrics", fake_metrics)

    page = await service.get_recommendations(category="concerts", user_uuid="u-from-metrics", size=5)
    assert page.items[0].item_id == 201  # 한로로 행동 기록 기반으로 한로로 공연 1위 추천



