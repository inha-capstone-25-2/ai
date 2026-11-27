"""추천 서버와 별도로 실행하는 주기적 재학습 프로세스."""

import fcntl
import hashlib
import logging
import os
import time
from pathlib import Path

import numpy as np

from ...data.data_loader import MongoDataLoader
from ..bandit_policy import DEFAULT_MODEL_PATH, read_model_checkpoint
from ..dataset.builder import build_bandit_dataset_from_mongo, ensure_training_indexes
from .train_offline import train_offline_bandit

logger = logging.getLogger(__name__)


def retrain_if_ready(
    model_path: Path = DEFAULT_MODEL_PATH,
    *,
    min_samples: int = 200,
    min_per_label: int = 20,
    exposure_limit: int = 10_000,
    loader: MongoDataLoader | None = None,
) -> bool:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info("다른 프로세스가 추천 모델을 학습 중입니다.")
            return False

        dataset = build_bandit_dataset_from_mongo(limit=exposure_limit, loader=loader)
        clicks = np.count_nonzero(dataset.y == 1)
        non_clicks = np.count_nonzero(dataset.y == 0)
        if dataset.y.size < min_samples or min(clicks, non_clicks) < min_per_label:
            logger.info(
                "추천 학습 표본 부족: 클릭 %s건, 미클릭 %s건", clicks, non_clicks
            )
            return False

        digest = hashlib.sha256(dataset.X.astype("<f4").tobytes())
        digest.update(dataset.y.astype("<f4").tobytes())
        training_data_digest = digest.hexdigest()
        if model_path.exists():
            try:
                _, previous_digest = read_model_checkpoint(model_path)
            except Exception:
                logger.exception(
                    "기존 추천 모델 파일을 읽지 못했습니다: %s", model_path
                )
            else:
                if previous_digest == training_data_digest:
                    logger.info("이전과 동일한 추천 학습 데이터입니다.")
                    return False

        train_offline_bandit(
            model_path=model_path,
            dataset=dataset,
            training_data_digest=training_data_digest,
        )
        logger.info("추천 모델 갱신 완료: 클릭 %s건, 미클릭 %s건", clicks, non_clicks)
        return True


def main() -> None:
    interval = int(os.getenv("RECOMMENDATION_TRAIN_INTERVAL_SECONDS", "86400"))
    min_samples = int(os.getenv("RECOMMENDATION_TRAIN_MIN_SAMPLES", "200"))
    min_per_label = int(os.getenv("RECOMMENDATION_TRAIN_MIN_PER_LABEL", "20"))
    if min(interval, min_samples, min_per_label) < 1:
        raise ValueError("학습 주기와 최소 표본 수는 양수여야 합니다.")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    indexes_ready = False
    while True:
        loader = None
        delay = interval
        try:
            loader = MongoDataLoader()
            if not indexes_ready:
                ensure_training_indexes(loader)
                indexes_ready = True
            retrain_if_ready(
                min_samples=min_samples, min_per_label=min_per_label, loader=loader
            )
        except Exception:
            logger.exception("추천 모델 재학습에 실패했습니다.")
            delay = min(interval, 300)
        finally:
            if loader is not None:
                loader.client.close()
        time.sleep(delay)


if __name__ == "__main__":
    main()
