"""Controlled baselines and evaluation helpers for the PD study.

These helpers keep training and evaluation separate: exploration is disabled
for evaluation, and every result carries the random seed and opponent label.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from publication_sarsa import (
    Action,
    PrisonersDilemmaEnvironment,
    State,
    TwoStepSARSAAgent,
    _apply_n_step_update,
)

OpponentPolicy = Callable[[Optional[Action], Optional[Action], np.random.Generator], Action]


def always_cooperate(_: Optional[Action], __: Optional[Action], ___: np.random.Generator) -> Action:
    return "C"


def always_defect(_: Optional[Action], __: Optional[Action], ___: np.random.Generator) -> Action:
    return "D"


def tit_for_tat(opponent_last_action: Optional[Action], _: Optional[Action], __: np.random.Generator) -> Action:
    return opponent_last_action if opponent_last_action in ("C", "D") else "C"


def pavlov(opponent_last_action: Optional[Action], agent_last_action: Optional[Action], _: np.random.Generator) -> Action:
    # Win-stay/lose-shift requires the previous joint outcome. This baseline
    # uses the standard heuristic: repeat C after CC and D after DC; otherwise shift.
    if opponent_last_action == "C" and agent_last_action == "C":
        return "C"
    if opponent_last_action == "D" and agent_last_action == "C":
        return "D"
    if opponent_last_action == "C" and agent_last_action == "D":
        return "D"
    return "C"


OPPONENTS: Dict[str, OpponentPolicy] = {
    "always_cooperate": always_cooperate,
    "always_defect": always_defect,
    "tit_for_tat": tit_for_tat,
    "pavlov": pavlov,
}


def _play_episode(
    agent: TwoStepSARSAAgent,
    opponent: OpponentPolicy,
    rng: np.random.Generator,
    train: bool,
) -> Dict[str, float | Dict[str, int]]:
    env = PrisonersDilemmaEnvironment()
    states: List[State] = []
    actions: List[Action] = []
    rewards: List[float] = []
    opponent_last: Optional[Action] = None
    agent_last: Optional[Action] = None
    counts = {"CC": 0, "CD": 0, "DC": 0, "DD": 0}
    steps = 100

    for step in range(steps):
        state = agent.state_from_opponent_action(opponent_last)
        action = agent.choose_action(state, rng)
        opponent_action = opponent(opponent_last, agent_last, rng)
        reward, _ = env.step(action, opponent_action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        counts[action + opponent_action] += 1
        if train and step >= 2:
            _apply_n_step_update(agent, states, actions, rewards, step - 2)
        opponent_last, agent_last = action, opponent_action

    if train:
        for tau in range(max(0, steps - 2), steps):
            _apply_n_step_update(agent, states, actions, rewards, tau)

    return {
        "reward": float(sum(rewards)),
        "cooperation_rate": actions.count("C") / steps,
        "outcome_counts": counts,
    }


def evaluate_against_fixed_opponent(
    opponent_name: str,
    training_episodes: int = 1_000,
    evaluation_episodes: int = 100,
    seed: Optional[int] = None,
    agent_params: Optional[dict] = None,
) -> Dict[str, object]:
    """Train a two-step learner and evaluate it with epsilon forced to zero."""
    if opponent_name not in OPPONENTS:
        raise ValueError(f"Unknown opponent: {opponent_name}")
    if training_episodes <= 0 or evaluation_episodes <= 0:
        raise ValueError("Episode counts must be positive.")

    rng = np.random.default_rng(seed)
    agent = TwoStepSARSAAgent(**(agent_params or {}))
    opponent = OPPONENTS[opponent_name]
    for _ in range(training_episodes):
        _play_episode(agent, opponent, rng, train=True)
        agent.decay_exploration()

    agent.epsilon = 0.0
    evaluation = [_play_episode(agent, opponent, rng, train=False) for _ in range(evaluation_episodes)]
    return {
        "opponent": opponent_name,
        "seed": seed,
        "training_episodes": training_episodes,
        "evaluation_episodes": evaluation_episodes,
        "evaluation": evaluation,
        "mean_reward": float(np.mean([item["reward"] for item in evaluation])),
        "mean_cooperation_rate": float(np.mean([item["cooperation_rate"] for item in evaluation])),
    }
