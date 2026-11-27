from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import mongomock
import numpy as np
import pytest
from bson import ObjectId

from recommendation.rl.dataset import builder


NOW = datetime(2026, 11, 27, 12)
OLD = NOW - timedelta(hours=3)


@pytest.fixture
def loader():
    db = mongomock.MongoClient().db
    paper_calls, profile_calls = [], []

    def paper(paper_id):
        paper_calls.append(paper_id)
        return SimpleNamespace(arxiv_id=paper_id)

    def profile(user_id):
        profile_calls.append(user_id)
        return SimpleNamespace(user_id=user_id)

    return SimpleNamespace(
        db=db,
        get_paper_by_arxiv_id=paper,
        build_user_profile=profile,
        paper_calls=paper_calls,
        profile_calls=profile_calls,
    )


def ai(loader, rec, paper, *, ready=True, at=OLD, features=None):
    doc = {
        "_id": rec,
        "recommendation_id": rec,
        "user_id": 7,
        "created_at": at,
        "items": [
            {
                "paper_id": paper,
                "features": features
                or {"keyword": 0.2, "category": 0.3, "popularity": 0.4, "recency": 0.5},
            }
        ],
    }
    if ready is not None:
        doc["attribution_ready"] = ready
    loader.db.recommendation_events.insert_one(doc)


def spring(loader, record, paper, *, session="session-1", user=7, at=OLD, flag=False):
    loader.db.paper_recommendations.insert_one(
        {
            "_id": record,
            "session_id": session,
            "user_id": user,
            "paper_id": paper,
            "recommendation_type": "rule_based",
            "recommended_at": at,
            "was_clicked": flag,
        }
    )


def interaction(
    loader,
    record,
    *,
    rec=None,
    user=7,
    paper="p1",
    at=OLD + timedelta(minutes=1),
    action="click",
    session=None,
):
    doc = {
        "_id": record,
        "user_id": user,
        "paper_id": paper,
        "action_type": action,
        "created_at": at,
    }
    if rec is not None:
        doc["recommendation_id"] = rec
    if session is not None:
        doc["sessionId"] = session
    loader.db.recommendation_interactions.insert_one(doc)


def test_ai_clicks_negatives_and_feature_snapshot(loader):
    ai(loader, "r1", "p1")
    ai(
        loader,
        "r2",
        "p2",
        features={
            "keyword": 0.1,
            "category": 0.2,
            "popularity": 0.3,
            "recency": 0.4,
            "rule_total_score": 0.99,
        },
    )
    interaction(loader, "c1", rec="r1")
    result = builder.build_bandit_dataset_from_mongo(loader=loader, now=NOW)
    assert result.paper_ids == ["p1", "p2"]
    assert result.user_ids == [7, 7]
    np.testing.assert_array_equal(result.y, [1, 0])
    np.testing.assert_allclose(
        result.X[0],
        [
            0.2,
            0.3,
            0.4,
            0.5,
            0.2 + 0.3 + builder.W_POPULARITY * 0.4 + builder.W_RECENCY * 0.5,
        ],
    )
    np.testing.assert_allclose(result.X[1], [0.1, 0.2, 0.3, 0.4, 0.99])
    assert loader.paper_calls == []


def test_ai_legacy_clicks_only_and_validated_join(loader):
    ai(loader, "legacy-clicked", "p1", ready=None)
    ai(loader, "legacy-unclicked", "p2", ready=None)
    ai(loader, "new-negative", "p3")
    ai(loader, "duplicate", "p4")
    ai(loader, "before-only", "p5")
    loader.db.recommendation_events.update_one(
        {"_id": "duplicate"},
        {
            "$push": {
                "items": {
                    "paper_id": "p4",
                    "features": {
                        "keyword": 1,
                        "category": 1,
                        "popularity": 1,
                        "recency": 1,
                    },
                }
            }
        },
    )
    interaction(loader, "good", rec="legacy-clicked")
    interaction(loader, "repeat", rec="legacy-clicked")
    interaction(loader, "wrong-user", rec="new-negative", user=8, paper="p3")
    interaction(loader, "wrong-paper", rec="new-negative", paper="other")
    interaction(
        loader, "before", rec="before-only", paper="p5", at=OLD - timedelta(seconds=1)
    )
    interaction(loader, "equal", rec="new-negative", paper="p3", at=OLD)
    interaction(
        loader, "future", rec="new-negative", paper="p3", at=NOW + timedelta(seconds=1)
    )
    interaction(loader, "bookmark", rec="new-negative", paper="p3", action="bookmark")
    result = builder.build_bandit_dataset_from_mongo(loader=loader, now=NOW)
    assert result.paper_ids == ["p5", "p4", "p1", "p3"]
    assert dict(zip(result.paper_ids, result.y)) == {"p1": 1, "p3": 1, "p4": 0, "p5": 0}
    assert len(result.paper_ids) == 4


def test_spring_session_event_per_paper_interaction_and_true_flag(loader, monkeypatch):
    calls = []

    def features(profile, papers, now):
        calls.append(now)
        return np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]), [papers[0].arxiv_id], []

    monkeypatch.setattr(builder, "build_candidate_features", features)
    spring(loader, "paper-row-1", "p1")
    spring_id = ObjectId()
    spring(loader, spring_id, "p2")
    spring(loader, "paper-row-3", "p3")
    spring(loader, "paper-row-4", "p4", flag=True)
    loader.db.recommendation_events.insert_one(
        {
            "_id": "spring-click",
            "sessionId": "session-1",
            "userId": 7,
            "paperId": "p1",
            "activityType": "CLICK",
            "timestamp": (OLD + timedelta(minutes=2))
            .replace(tzinfo=timezone.utc)
            .isoformat(),
        }
    )
    loader.db.recommendation_events.insert_one(
        {
            "_id": "spring-wrong-user",
            "sessionId": "session-1",
            "userId": 9,
            "paperId": "p3",
            "activityType": "CLICK",
            "timestamp": OLD + timedelta(minutes=2),
        }
    )
    loader.db.recommendation_events.insert_one(
        {
            "_id": "spring-before",
            "sessionId": "session-1",
            "userId": 7,
            "paperId": "p3",
            "activityType": "CLICK",
            "timestamp": OLD - timedelta(seconds=1),
        }
    )
    interaction(loader, "ai-click", rec=str(spring_id), paper="p2", at=OLD)
    interaction(loader, "wrong-session", paper="p3", session="other")
    interaction(loader, "wrong-user", paper="p3", user=9, session="session-1")
    interaction(loader, "wrong-paper", paper="other", session="session-1")
    result = builder.build_bandit_dataset_from_mongo(loader=loader, now=NOW)
    assert dict(zip(result.paper_ids, result.y)) == {"p1": 1, "p2": 1, "p3": 0, "p4": 1}
    assert calls == [OLD] * 4
    assert loader.profile_calls == [7]
    assert set(loader.paper_calls) == {"p1", "p2", "p3", "p4"}
    assert len(loader.paper_calls) == 4


def test_maturity_limit_order_and_missing_identity(loader, monkeypatch):
    monkeypatch.setattr(
        builder,
        "build_candidate_features",
        lambda profile, papers, now: (np.ones((1, 5)), [papers[0].arxiv_id], []),
    )
    ai(loader, "late", "p-late", at=NOW - timedelta(minutes=30))
    ai(loader, "old", "p-old", at=NOW - timedelta(days=40))
    ai(loader, "valid", "p-valid")
    spring(loader, 123, "p-missing", session=None)
    spring(loader, "s1", "p-spring")
    result = builder.build_bandit_dataset_from_mongo(loader=loader, now=NOW, limit=1)
    assert result.paper_ids == ["p-valid", "p-spring"]
    assert builder.build_bandit_dataset_from_mongo(
        loader=loader, now=NOW, limit=2
    ).paper_ids == ["p-valid", "p-spring"]
    assert builder.build_bandit_dataset_from_mongo(
        loader=loader, now=NOW, limit=0
    ).X.shape == (0, 5)


def test_newest_per_source_selected_before_cap(loader, monkeypatch):
    monkeypatch.setattr(
        builder,
        "build_candidate_features",
        lambda profile, papers, now: (np.ones((1, 5)), [papers[0].arxiv_id], []),
    )
    ai(loader, "older-ai", "p-old-ai", at=OLD - timedelta(hours=1))
    ai(loader, "newer-ai", "p-new-ai", at=OLD)
    spring(loader, "older-spring", "p-old-spring", at=OLD - timedelta(hours=1))
    spring(loader, "newer-spring", "p-new-spring", at=OLD)
    loader.db.paper_recommendations.insert_one(
        {
            "_id": "other-type",
            "session_id": "session-1",
            "user_id": 7,
            "paper_id": "p-other",
            "recommendation_type": "hybrid",
            "recommended_at": OLD + timedelta(minutes=1),
        }
    )
    result = builder.build_bandit_dataset_from_mongo(loader=loader, now=NOW, limit=1)
    assert result.paper_ids == ["p-new-ai", "p-new-spring"]


def test_training_indexes_are_explicit_and_idempotent(loader):
    builder.ensure_training_indexes(loader)
    builder.ensure_training_indexes(loader)
    assert "created_at_-1__id_-1" in loader.db.recommendation_events.index_information()
    assert (
        "sessionId_1_activityType_1_timestamp_1"
        in loader.db.recommendation_events.index_information()
    )
    assert (
        "recommendation_type_1_recommended_at_-1__id_-1"
        in loader.db.paper_recommendations.index_information()
    )
    assert (
        "recommendation_id_1_action_type_1_created_at_1"
        in loader.db.recommendation_interactions.index_information()
    )
