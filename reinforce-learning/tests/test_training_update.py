import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from recommendation.rl.bandit_policy import SimpleBanditModel, read_model_checkpoint
from recommendation.rl.dataset.builder import BanditDataset
from recommendation.rl.trainer.scheduled import retrain_if_ready
from recommendation.rl.trainer.train_offline import train_offline_bandit
from recommendation.service.reranker import BanditPolicyWrapper, RerankConfig
from server import get_recommendations, get_recommendations_rl


class RecommendationResponseTest(unittest.IsolatedAsyncioTestCase):
    async def test_responses_return_the_logged_recommendation_id(self):
        raw_response = {"recommendation_id": "exposure-id", "results": []}
        with patch("server.get_user_recommendations", return_value=raw_response):
            rule_result = await get_recommendations(user_id=1, limit=6, session_id=None)
        with patch("server.get_user_recommendations_rl", return_value=raw_response):
            hybrid_result = await get_recommendations_rl(
                user_id=1,
                limit=6,
                candidate_k=100,
                base_paper_id=None,
                session_id="caller-session",
            )

        self.assertEqual(rule_result.recommendation_id, "exposure-id")
        self.assertEqual(rule_result.session_id, "exposure-id")
        self.assertEqual(hybrid_result.recommendation_id, "exposure-id")
        self.assertEqual(hybrid_result.session_id, "caller-session")


def labeled_samples(labels):
    return BanditDataset(
        X=np.arange(len(labels) * 5).reshape(-1, 5) / 20,
        y=np.array(labels, dtype=np.float32),
        user_ids=[1] * len(labels),
        paper_ids=list(map(str, range(len(labels)))),
    )


class TrainingUpdateTest(unittest.TestCase):
    def test_training_learns_to_score_clicked_examples_higher(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            dataset = labeled_samples([0, 0, 1, 1])
            train_offline_bandit(path, dataset=dataset)

            policy = BanditPolicyWrapper(RerankConfig(path))
            scores = policy.predict_scores(dataset.X)
            self.assertGreater(
                scores[dataset.y == 1].min(), scores[dataset.y == 0].max()
            )

    def test_training_waits_for_both_labels_then_publishes_only_changed_data(self):
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pt"
            with patch(
                "recommendation.rl.trainer.scheduled.build_bandit_dataset_from_mongo"
            ) as build:
                build.return_value = labeled_samples([0, 0])
                self.assertFalse(
                    retrain_if_ready(model_path, min_samples=2, min_per_label=1)
                )
                self.assertFalse(model_path.exists())

                build.return_value = labeled_samples([0, 1])
                self.assertTrue(
                    retrain_if_ready(model_path, min_samples=2, min_per_label=1)
                )
                stored_model = model_path.read_bytes()
                _, first_digest = read_model_checkpoint(model_path)
                self.assertIsNotNone(first_digest)
                self.assertFalse(
                    retrain_if_ready(model_path, min_samples=2, min_per_label=1)
                )
                self.assertEqual(stored_model, model_path.read_bytes())

                build.return_value = labeled_samples([1, 0])
                self.assertTrue(
                    retrain_if_ready(model_path, min_samples=2, min_per_label=1)
                )
                self.assertNotEqual(first_digest, read_model_checkpoint(model_path)[1])

    def test_failed_save_preserves_the_previous_checkpoint(self):
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pt"
            model_path.write_bytes(b"previous model")
            with patch(
                "recommendation.rl.trainer.train_offline.torch.save",
                side_effect=OSError,
            ):
                with self.assertRaises(OSError):
                    train_offline_bandit(
                        model_path, num_epochs=1, dataset=labeled_samples([0, 1])
                    )

            self.assertEqual(model_path.read_bytes(), b"previous model")
            self.assertEqual(list(Path(directory).iterdir()), [model_path])

    def test_server_reloads_updated_model_and_keeps_last_good_version(self):
        with TemporaryDirectory() as directory:
            fallback_path = Path(directory) / "seed.pt"
            model_path = Path(directory) / "model.pt"
            policy = BanditPolicyWrapper(RerankConfig(model_path, fallback_path))
            features = np.ones((1, 5))

            self.assertIsNone(policy.predict_scores(features))
            model = SimpleBanditModel(input_dim=5)
            with torch.no_grad():
                model.linear.weight.zero_()
                model.linear.bias.fill_(1)
            torch.save(model.state_dict(), fallback_path)
            np.testing.assert_array_equal(policy.predict_scores(features), [1])

            with torch.no_grad():
                model.linear.bias.fill_(2)
            staging = Path(directory) / "staging.pt"
            torch.save(
                {"model_state_dict": model.state_dict(), "training_data_digest": "new"},
                staging,
            )
            staging.replace(model_path)
            np.testing.assert_array_equal(policy.predict_scores(features), [2])

            staging.write_bytes(b"invalid model")
            staging.replace(model_path)
            np.testing.assert_array_equal(policy.predict_scores(features), [2])
