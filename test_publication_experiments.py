"""Tests for controlled baselines and evaluation."""

import numpy as np

from publication_experiments import evaluate_against_fixed_opponent


def test_fixed_opponent_evaluation_is_exploration_free_and_reproducible() -> None:
    params = {"epsilon_start": 0.2, "epsilon_decay": 0.99, "epsilon_min": 0.05}
    first = evaluate_against_fixed_opponent("always_defect", 10, 5, 4, params)
    second = evaluate_against_fixed_opponent("always_defect", 10, 5, 4, params)
    assert first["mean_reward"] == second["mean_reward"]
    assert first["mean_cooperation_rate"] == second["mean_cooperation_rate"]
    assert 0.0 <= first["mean_cooperation_rate"] <= 1.0


def test_all_controlled_opponents_return_complete_episode_counts() -> None:
    for opponent in ("always_cooperate", "always_defect", "tit_for_tat", "pavlov"):
        result = evaluate_against_fixed_opponent(opponent, 2, 3, 10)
        assert len(result["evaluation"]) == 3
        for episode in result["evaluation"]:
            assert sum(episode["outcome_counts"].values()) == 100
            assert np.isfinite(episode["reward"])
