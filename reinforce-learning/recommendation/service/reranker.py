from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import List, Optional

import numpy as np

from ..data.data_loader import MongoDataLoader
from ..models.data_models import RecommendationResult, UserProfile
from ..rl.state_builder import build_candidate_features
from ..rl.bandit_policy import (
    BUNDLED_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    SimpleBanditModel,
    read_model_checkpoint,
)

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # torch 미설치 환경에서는 RL 기능 비활성화


@dataclass
class RerankConfig:
    model_path: Path = Path(DEFAULT_MODEL_PATH)
    fallback_model_path: Optional[Path] = None


class BanditPolicyWrapper:
    """모델 파일이 갱신되면 새 모델을 읽고, 읽기에 실패하면 직전 모델을 사용한다."""

    def __init__(self, config: Optional[RerankConfig] = None) -> None:
        self.config = config or RerankConfig(fallback_model_path=BUNDLED_MODEL_PATH)
        self._model: Optional[SimpleBanditModel] = None
        self._attempted_versions: dict[Path, tuple[int, ...]] = {}
        self._load_lock = Lock()

    def _get_model(self, input_dim: int) -> Optional[SimpleBanditModel]:
        if torch is None:
            return None

        with self._load_lock:
            paths = [self.config.model_path]
            if self._model is None and self.config.fallback_model_path is not None:
                paths.append(self.config.fallback_model_path)

            for path in paths:
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                except OSError:
                    logger.exception("[RL Reranker] 모델 파일에 접근할 수 없습니다: %s", path)
                    continue
                version = (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)
                if self._attempted_versions.get(path) == version:
                    continue

                self._attempted_versions[path] = version
                try:
                    model = SimpleBanditModel(input_dim=input_dim)
                    state, _ = read_model_checkpoint(path)
                    model.load_state_dict(state)
                    if not all(torch.isfinite(param).all() for param in model.parameters()):
                        raise ValueError("모델 가중치에 유한하지 않은 값이 있습니다.")
                    self._model = model.eval()
                except Exception:
                    logger.exception("[RL Reranker] 모델을 읽지 못했습니다: %s", path)
                    continue

                logger.info("[RL Reranker] 모델 로딩 완료: %s", path)
                break

            return self._model

    def predict_scores(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        X: (N, D) feature matrix
        return: (N,) 예측 점수 or None (모델 사용 불가 시)
        """
        if X.size == 0:
            return None

        model = self._get_model(input_dim=X.shape[1])
        if model is None:
            return None

        with torch.no_grad():
            t = torch.from_numpy(X).float()
            y = model(t).cpu().numpy()
        
        logger.info(f"[RL Reranker] 🎲 RL 점수 예측 완료: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}")
        return y


class RLBanditReranker:
    """
    Rule-based 후보군(List[RecommendationResult])을 입력으로 받아
    Contextual Bandit 정책으로 rerank 후 top_k개를 반환.
    """
    def __init__(self, loader: Optional[MongoDataLoader] = None):
        self.loader = loader or MongoDataLoader()
        self.policy = BanditPolicyWrapper()

    def rerank(
        self,
        user_id: int,
        candidates: List[RecommendationResult],
        top_k: int = 6,
    ) -> List[RecommendationResult]:
        """
        candidates: RuleBasedRecommender가 만든 상위 N개 (예: 100개)
        return: RL 점수를 기준으로 다시 정렬한 상위 top_k
        """
        if not candidates:
            return []

        logger.info(f"[RL Reranker] 📥 Reranking 시작: {len(candidates)}개 후보 → top {top_k}")

        # 1) UserProfile 로딩
        profile: UserProfile = self.loader.build_user_profile(user_id)

        # 2) 후보 논문들 feature matrix 생성
        papers = [c.paper for c in candidates]
        X, _, feature_dicts = build_candidate_features(profile, papers)
        for candidate, features in zip(candidates, feature_dicts):
            candidate.features = {**(candidate.features or {}), **features}
        logger.info(f"[RL Reranker] 📊 Feature matrix 생성: shape={X.shape}")

        # 3) RL 정책으로 점수 예측
        rl_scores = self.policy.predict_scores(X)

        # RL 사용 불가(troch 미설치, 모델 없음 등) → rule-based 순서 그대로 top_k
        if rl_scores is None:
            logger.warning("[RL Reranker] ⚠️ RL 모델 사용 불가 → rule-based 결과 그대로 사용")
            return candidates[:top_k]

        logger.info("[RL Reranker] ✅ RL 모델 활성화 → RL 점수로 reranking")

        # 4) 후보들에 RL 점수 적용
        reranked: List[RecommendationResult] = []
        for c, rl_s in zip(candidates, rl_scores):
            # 기존 rule-based score를 보존하고, score를 RL 점수로 덮어씌움
            rule_score = c.score
            c.features = dict(c.features or {})
            c.features["rule_score"] = float(rule_score)
            c.features["rl_score"] = float(rl_s)
            final_score = 0.6 * rl_s + 0.4 * rule_score
            c.score = final_score
            reranked.append(c)

        # 5) RL 점수 기준 내림차순 정렬 후 top_k 
        reranked.sort(key=lambda r: r.score, reverse=True)

        # 다양성이 너무 없는 관계로 수정!
        final: List[RecommendationResult] = []
        used_categories = set()

        """
        이미 선택된 것들과 카테고리가 겹치면 패널티 (또는 스킵)
        너무 빡세게 스킵하면 추천이 비어버릴 수 있으니까 가능하면
        다른 카테고리 우선으로 구성. 왜냐면 후보군에서 추천된걸 rerank하는거라
        """
        for cand in reranked:
            cats = set(getattr(cand.paper, "categories", []) or [])
            if used_categories and cats & used_categories and len(final) >= 3:#앞쪽 3개는 그냥 두고 이후부터는 겹치는 건 한 번 건너뛰는 식
                continue

            final.append(cand)
            used_categories.update(cats)
            if len(final) >= top_k:
                break

        if len(final) < top_k:
            for cand in reranked:
                if cand in final:
                    continue
                final.append(cand)
                if len(final) >= top_k:
                    break  

        return final[:top_k]
