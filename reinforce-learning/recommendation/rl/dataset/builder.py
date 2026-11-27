from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite
from typing import List, Optional

import numpy as np
from bson import ObjectId

from ...data.data_loader import MongoDataLoader
from ...rule_based.scoring import W_POPULARITY, W_RECENCY
from ..state_builder import build_candidate_features


@dataclass
class BanditDataset:
    X: np.ndarray
    y: np.ndarray
    user_ids: List[int]
    paper_ids: List[str]


# A bounded, indexed scan for each exposure source. `limit` overrides this per source.
DEFAULT_SOURCE_LIMIT = 10_000
JOIN_BATCH_SIZE = 200


def ensure_training_indexes(loader):
    """Create the lookup and sort indexes used by the scheduled training scan."""
    loader.db["recommendation_events"].create_index([("created_at", -1), ("_id", -1)])
    loader.db["recommendation_events"].create_index(
        [("sessionId", 1), ("activityType", 1), ("timestamp", 1)]
    )
    loader.db["paper_recommendations"].create_index(
        [("recommendation_type", 1), ("recommended_at", -1), ("_id", -1)]
    )
    loader.db["recommendation_interactions"].create_index(
        [("recommendation_id", 1), ("action_type", 1), ("created_at", 1)]
    )


def _utc_naive(value):
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _id(value):
    return (
        value
        if isinstance(value, (int, str)) and not isinstance(value, bool) and value != ""
        else None
    )


def _feature_row(features):
    if not isinstance(features, dict):
        return None
    try:
        keyword, category, popularity, recency = (
            float(features[key])
            for key in ("keyword", "category", "popularity", "recency")
        )
        total = (
            float(features["rule_total_score"])
            if "rule_total_score" in features
            else (keyword + category + W_POPULARITY * popularity + W_RECENCY * recency)
        )
    except (KeyError, TypeError, ValueError):
        return None
    row = (keyword, category, popularity, recency, total)
    return row if all(isfinite(value) for value in row) else None


def _click_time(doc):
    return _utc_naive(doc.get("created_at") or doc.get("timestamp"))


def _is_click(doc):
    action = doc.get("action_type") or doc.get("activityType")
    return isinstance(action, str) and action.lower() == "click"


def _matched_clicks(collection, exposures, now):
    """Look up only identifiers in this batch; then validate every join key and time."""
    ai_ids = {e["rec_id"] for e in exposures if e["source"] == "ai"}
    spring_ids = {
        e["record_id"]
        for e in exposures
        if e["source"] == "spring" and e["record_id"] is not None
    }
    if not (ai_ids or spring_ids):
        return set()

    by_ai = {}
    by_record = {}
    for index, exposure in enumerate(exposures):
        key = (exposure["user_id"], exposure["paper_id"])
        if exposure["source"] == "ai":
            by_ai.setdefault((exposure["rec_id"], *key), []).append(index)
        else:
            if exposure["record_id"] is not None:
                by_record.setdefault((exposure["record_id"], *key), []).append(index)

    clicked = set()
    query = {
        "recommendation_id": {"$in": list(ai_ids | spring_ids)},
        "action_type": {"$in": ["click", "CLICK"]},
        "created_at": {"$gte": min(e["at"] for e in exposures), "$lte": now},
    }
    projection = {
        "recommendation_id": 1,
        "user_id": 1,
        "paper_id": 1,
        "action_type": 1,
        "created_at": 1,
    }
    for doc in collection.find(query, projection):
        if not _is_click(doc):
            continue
        when = _click_time(doc)
        if when is None or when > now:
            continue
        user, paper = (
            _id(doc.get("user_id", doc.get("userId"))),
            _id(doc.get("paper_id", doc.get("paperId"))),
        )
        if user is None or paper is None:
            continue
        matches = []
        rec_id = _id(doc.get("recommendation_id"))
        if rec_id is not None:
            matches.extend(by_ai.get((rec_id, user, paper), ()))
            matches.extend(by_record.get((rec_id, user, paper), ()))
        for index in matches:
            if exposures[index]["at"] <= when:
                clicked.add(index)
    return clicked


def build_bandit_dataset_from_mongo(
    limit: Optional[int] = None,
    *,
    loader=None,
    now=None,
    lookback=timedelta(days=30),
    wait_for_clicks=timedelta(hours=1),
) -> BanditDataset:
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    if lookback < timedelta(0) or wait_for_clicks < timedelta(0):
        raise ValueError("time windows must be nonnegative")
    now = _utc_naive(now) if now is not None else datetime.utcnow()
    if now is None:
        raise ValueError("invalid now timestamp")
    start, mature_until = now - lookback, now - wait_for_clicks
    source_limit = DEFAULT_SOURCE_LIMIT if limit is None else limit
    if source_limit == 0:
        return BanditDataset(
            X=np.zeros((0, 5)), y=np.zeros((0,)), user_ids=[], paper_ids=[]
        )
    loader = loader or MongoDataLoader()
    exposures = []
    seen = set()

    ai_cursor = (
        loader.db["recommendation_events"]
        .find(
            {
                "created_at": {"$gte": start, "$lte": mature_until},
                "items": {"$exists": True},
            },
            {
                "recommendation_id": 1,
                "user_id": 1,
                "created_at": 1,
                "items": 1,
                "attribution_ready": 1,
            },
        )
        .sort([("created_at", -1), ("_id", -1)])
        .limit(source_limit)
    )
    for doc in ai_cursor:
        rec_id, user = _id(doc.get("recommendation_id")), _id(doc.get("user_id"))
        at = _utc_naive(doc.get("created_at"))
        if (
            rec_id is None
            or user is None
            or at is None
            or not start <= at <= mature_until
        ):
            continue
        for item in doc.get("items") or ():
            if not isinstance(item, dict):
                continue
            paper = _id(item.get("paper_id"))
            row = _feature_row(item.get("features"))
            key = ("ai", rec_id, user, paper)
            if paper is None or row is None or key in seen:
                continue
            seen.add(key)
            exposures.append(
                {
                    "source": "ai",
                    "rec_id": rec_id,
                    "user_id": user,
                    "paper_id": paper,
                    "at": at,
                    "row": row,
                    "attribution_ready": doc.get("attribution_ready") is True,
                }
            )

    spring_cursor = (
        loader.db["paper_recommendations"]
        .find(
            {
                "recommendation_type": "rule_based",
                "recommended_at": {"$gte": start, "$lte": mature_until},
            },
            {
                "session_id": 1,
                "user_id": 1,
                "paper_id": 1,
                "recommended_at": 1,
                "was_clicked": 1,
            },
        )
        .sort([("recommended_at", -1), ("_id", -1)])
        .limit(source_limit)
    )
    profiles, papers = {}, {}
    for doc in spring_cursor:
        session = _id(doc.get("session_id"))
        raw_record = doc.get("_id")
        record = (
            str(raw_record) if isinstance(raw_record, ObjectId) else _id(raw_record)
        )
        user, paper_id = _id(doc.get("user_id")), _id(doc.get("paper_id"))
        at = _utc_naive(doc.get("recommended_at"))
        if (
            user is None
            or paper_id is None
            or (session is None and not isinstance(record, str))
            or at is None
            or not start <= at <= mature_until
        ):
            continue
        key = ("spring", session, record, user, paper_id)
        if key in seen:
            continue
        seen.add(key)
        try:
            profile_id = int(user)
        except (TypeError, ValueError):
            continue
        if paper_id not in papers:
            papers[paper_id] = loader.get_paper_by_arxiv_id(paper_id)
        if profile_id not in profiles:
            profiles[profile_id] = loader.build_user_profile(profile_id)
        paper = papers[paper_id]
        if paper is None or profiles[profile_id] is None:
            continue
        X, ids, _ = build_candidate_features(profiles[profile_id], [paper], now=at)
        if X.shape != (1, 5) or not ids or not np.all(np.isfinite(X)):
            continue
        exposures.append(
            {
                "source": "spring",
                "session_id": session,
                "record_id": record if isinstance(record, str) else None,
                "user_id": user,
                "paper_id": paper_id,
                "at": at,
                "row": X[0],
                "flag": doc.get("was_clicked") is True,
            }
        )

    exposures.sort(
        key=lambda e: (
            e["at"],
            e["source"],
            str(e.get("rec_id", e.get("session_id"))),
            str(e["user_id"]),
            str(e["paper_id"]),
        )
    )
    rows, labels, users, paper_ids = [], [], [], []
    interactions = loader.db["recommendation_interactions"]
    spring_events = loader.db["recommendation_events"]
    for offset in range(0, len(exposures), JOIN_BATCH_SIZE):
        batch = exposures[offset : offset + JOIN_BATCH_SIZE]
        clicked = _matched_clicks(interactions, batch, now)
        # Spring's authenticated /api/events writes to recommendation_events, separate from AI items.
        spring_batch = [
            (i, e)
            for i, e in enumerate(batch)
            if e["source"] == "spring" and e["session_id"] is not None
        ]
        if spring_batch:
            sessions = {e["session_id"] for _, e in spring_batch}
            by_key = {}
            for i, e in spring_batch:
                by_key.setdefault(
                    (e["session_id"], e["user_id"], e["paper_id"]), []
                ).append(i)
            event_query = {
                "sessionId": {"$in": list(sessions)},
                "activityType": {"$in": ["click", "CLICK"]},
            }
            event_projection = {
                "sessionId": 1,
                "userId": 1,
                "paperId": 1,
                "activityType": 1,
                "timestamp": 1,
            }
            for event in spring_events.find(event_query, event_projection):
                if not _is_click(event):
                    continue
                when = _click_time(event)
                if when is None or when > now:
                    continue
                key = (
                    _id(event.get("sessionId")),
                    _id(event.get("userId")),
                    _id(event.get("paperId")),
                )
                for i in by_key.get(key, ()):
                    if batch[i]["at"] <= when:
                        clicked.add(i)
        for i, exposure in enumerate(batch):
            if (
                exposure["source"] == "ai"
                and i not in clicked
                and not exposure["attribution_ready"]
            ):
                continue
            rows.append(exposure["row"])
            labels.append(float(i in clicked or exposure.get("flag", False)))
            users.append(exposure["user_id"])
            paper_ids.append(exposure["paper_id"])

    return BanditDataset(
        X=np.asarray(rows, dtype=float).reshape(-1, 5),
        y=np.asarray(labels, dtype=float),
        user_ids=users,
        paper_ids=paper_ids,
    )
