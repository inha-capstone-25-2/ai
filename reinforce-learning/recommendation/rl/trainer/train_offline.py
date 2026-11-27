from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..bandit_policy import DEFAULT_MODEL_PATH, SimpleBanditModel
from ..dataset.builder import BanditDataset, build_bandit_dataset_from_mongo


def train_offline_bandit(
    model_path: Optional[Path] = None,
    batch_size: int = 256,
    num_epochs: int = 30,
    lr: float = 1e-2,
    limit: Optional[int] = None,
    *,
    dataset: Optional[BanditDataset] = None,
    training_data_digest: Optional[str] = None,
) -> Path:
    """저장된 추천과 클릭으로 선형 모델을 학습하고 모델 파일을 교체한다."""
    model_path = Path(model_path or DEFAULT_MODEL_PATH)

    # 1) Dataset 구축
    if dataset is None:
        dataset = build_bandit_dataset_from_mongo(limit=limit)
    X, y = dataset.X, dataset.y

    if X.size == 0 or y.size == 0:
        raise RuntimeError(
            "offline 학습용 데이터가 없습니다. "
            "MongoDB에 추천 노출 로그가 쌓였는지 확인하세요."
        )
    if not np.isfinite(X).all() or not np.isin(y, (0, 1)).all():
        raise ValueError(
            "추천 특징은 유한한 값이어야 하고 클릭 라벨은 0 또는 1이어야 합니다."
        )

    input_dim = X.shape[1]

    # 2) TensorDataset 준비
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()

    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
    )

    # 3) 모델 초기화
    model = SimpleBanditModel(input_dim=input_dim)
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.fill_(y_t.mean().item())
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 회귀 문제로 간주하여 MSE 사용 (reward ~ expected value)
    criterion = nn.MSELoss()

    # 4) 학습 루프
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for xb, yb in dl:
            optimizer.zero_grad()
            pred = model(xb)  # (B,)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * xb.size(0)

        epoch_loss /= len(ds)
        print(f"[epoch {epoch + 1}/{num_epochs}] loss={epoch_loss:.4f}")

    if not all(torch.isfinite(param).all() for param in model.parameters()):
        raise ValueError("학습한 모델의 가중치에 유한하지 않은 값이 있습니다.")

    state = model.state_dict()
    checkpoint = (
        state
        if training_data_digest is None
        else {
            "model_state_dict": state,
            "training_data_digest": training_data_digest,
        }
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=model_path.parent, suffix=".tmp", delete=False
    ) as tmp:
        temp_path = Path(tmp.name)
        try:
            torch.save(checkpoint, tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            temp_path.replace(model_path)
        finally:
            temp_path.unlink(missing_ok=True)
    print(f"Saved bandit policy model to: {model_path}")

    return model_path


if __name__ == "__main__":
    train_offline_bandit()
