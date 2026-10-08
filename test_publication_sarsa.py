"""Tests for the publication-oriented SARSA implementation."""

import numpy as np

from publication_sarsa import TwoStepSARSAAgent, _apply_n_step_update, run_self_play


def test_exact_two_step_target() -> None:
    agent = TwoStepSARSAAgent(alpha=0.5, gamma=0.9)
    agent.set_q("start", "C", 1.0)
    agent.set_q("C", "D", 2.0)
    agent.update_two_step(
        state="start",
        action="C",
        reward_1=3.0,
        reward_2=0.0,
        next_state="C",
        next_action="D",
    )
    assert np.isclose(agent.q("start", "C"), 2.81)


def test_forward_view_uses_two_rewards_and_bootstrap_action() -> None:
    agent = TwoStepSARSAAgent(alpha=1.0, gamma=0.9)
    agent.set_q("C", "D", 2.0)
    _apply_n_step_update(
        agent,
        states=["start", "C", "D"],
        actions=["C", "D", "D"],
        rewards=[3.0, 4.0, 99.0],
        tau=0,
    )
    # R_1 + gamma*R_2 + gamma^2*Q(S_2,A_2) = 3 + 3.6 + 1.62.
    assert np.isclose(agent.q("start", "C"), 8.22)


def test_terminal_two_step_target_does_not_bootstrap() -> None:
    agent = TwoStepSARSAAgent(alpha=1.0, gamma=0.9)
    agent.set_q("C", "D", 99.0)
    agent.update_two_step(
        state="start",
        action="C",
        reward_1=3.0,
        reward_2=4.0,
        next_state="C",
        next_action="D",
        terminal=True,
    )
    assert np.isclose(agent.q("start", "C"), 3.0)


def test_reproducibility_with_seed() -> None:
    params = {"epsilon_start": 0.3, "epsilon_decay": 0.99, "epsilon_min": 0.05}
    first = run_self_play(8, 12, params, params, seed=7)
    second = run_self_play(8, 12, params, params, seed=7)
    assert first["episode_rewards_1"] == second["episode_rewards_1"]
    assert first["cooperation_rates_2"] == second["cooperation_rates_2"]


def test_episode_metrics_are_well_formed() -> None:
    result = run_self_play(4, 10, seed=11)
    assert len(result["episode_rewards_1"]) == 4
    assert all(0.0 <= rate <= 1.0 for rate in result["cooperation_rates_1"])
    assert all(sum(counts.values()) == 10 for counts in result["outcome_counts"])
