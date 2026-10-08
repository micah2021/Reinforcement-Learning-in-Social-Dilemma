"""Publication-oriented n-step SARSA experiments for a repeated Prisoner's Dilemma.

The canonical implementation in this file uses a memory-one state (the opponent's
previous action) and explicit finite-episode return truncation. It is intended for
reproducible experiments, not as a claim that independent multi-agent learning has
general convergence guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Action = str
State = str
Payoff = Tuple[float, float]


@dataclass(frozen=True)
class PrisonersDilemmaEnvironment:
    """Repeated two-player Prisoner's Dilemma with the canonical payoff matrix."""

    reward_matrix: Dict[Tuple[Action, Action], Payoff] | None = None

    def __post_init__(self) -> None:
        matrix = self.reward_matrix or {
            ("C", "C"): (3.0, 3.0),
            ("C", "D"): (0.0, 5.0),
            ("D", "C"): (5.0, 0.0),
            ("D", "D"): (1.0, 1.0),
        }
        object.__setattr__(self, "reward_matrix", matrix)
        t = matrix[("D", "C")][0]
        r = matrix[("C", "C")][0]
        p = matrix[("D", "D")][0]
        s = matrix[("C", "D")][0]
        if not (t > r > p > s):
            raise ValueError("Payoffs must satisfy T > R > P > S.")

    def step(self, action_1: Action, action_2: Action) -> Payoff:
        if (action_1, action_2) not in self.reward_matrix:
            raise ValueError(f"Unknown action pair: {(action_1, action_2)}")
        return self.reward_matrix[(action_1, action_2)]


class TwoStepSARSAAgent:
    """An epsilon-greedy two-step SARSA agent with a tabular Q function."""

    ACTIONS: Tuple[Action, Action] = ("C", "D")

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon_start: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1].")
        if not 0 <= epsilon_min <= epsilon_start <= 1:
            raise ValueError("Require 0 <= epsilon_min <= epsilon_start <= 1.")
        if not 0 < epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in (0, 1].")
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table: Dict[Tuple[State, Action], float] = {}
        self.q_history: Dict[State, Dict[Action, List[float]]] = {
            state: {action: [] for action in self.ACTIONS}
            for state in ("start", "C", "D")
        }

    @staticmethod
    def state_from_opponent_action(action: Optional[Action]) -> State:
        return action if action in ("C", "D") else "start"

    def q(self, state: State, action: Action) -> float:
        return self.q_table.get((state, action), 0.0)

    def set_q(self, state: State, action: Action, value: float) -> None:
        self.q_table[(state, action)] = float(value)

    def choose_action(self, state: State, rng: np.random.Generator) -> Action:
        if rng.random() < self.epsilon:
            return str(rng.choice(self.ACTIONS))
        values = np.array([self.q(state, action) for action in self.ACTIONS])
        best = np.flatnonzero(values == values.max())
        return self.ACTIONS[int(rng.choice(best))]

    def update_two_step(
        self,
        state: State,
        action: Action,
        reward_1: float,
        reward_2: Optional[float] = None,
        next_state: Optional[State] = None,
        next_action: Optional[Action] = None,
        terminal: bool = False,
    ) -> float:
        """Apply a two-step SARSA update, truncating the return at terminal time."""
        target = reward_1
        if reward_2 is not None and not terminal:
            target += self.gamma * reward_2
            if next_state is not None and next_action is not None:
                target += self.gamma**2 * self.q(next_state, next_action)
        elif not terminal and next_state is not None and next_action is not None:
            target += self.gamma * self.q(next_state, next_action)
        updated = self.q(state, action) + self.alpha * (target - self.q(state, action))
        self.set_q(state, action, updated)
        return updated

    def decay_exploration(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def record_q_values(self) -> None:
        for state in self.q_history:
            for action in self.ACTIONS:
                self.q_history[state][action].append(self.q(state, action))


def _apply_n_step_update(
    agent: TwoStepSARSAAgent,
    states: Sequence[State],
    actions: Sequence[Action],
    rewards: Sequence[float],
    tau: int,
    n: int = 2,
) -> None:
    """Apply the forward-view n-step target for transition ``tau``.

    ``rewards[k]`` is the reward produced by ``(states[k], actions[k])``;
    therefore it is R_{tau+1} when k == tau. The caller must invoke this only
    after the bootstrap action at index ``tau + n`` is available, or during the
    terminal flush when the return is intentionally truncated.
    """
    if not 0 <= tau < len(rewards):
        raise IndexError("tau must index an observed transition")
    horizon = min(tau + n, len(rewards))
    target = sum(
        agent.gamma ** (k - tau) * rewards[k]
        for k in range(tau, horizon)
    )
    if tau + n < len(rewards):
        target += agent.gamma**n * agent.q(states[tau + n], actions[tau + n])
    current = agent.q(states[tau], actions[tau])
    agent.set_q(states[tau], actions[tau], current + agent.alpha * (target - current))


def run_self_play(
    num_episodes: int = 1_000,
    steps_per_episode: int = 100,
    agent_1_params: Optional[dict] = None,
    agent_2_params: Optional[dict] = None,
    seed: Optional[int] = None,
) -> dict:
    """Train two independent learners and return raw episode-level metrics.

    Each episode is finite. A transition at index ``tau`` is updated only after
    the two-step bootstrap action at ``tau + 2`` has been observed. The final two
    transitions are flushed with truncated returns, so no terminal transition
    bootstraps from a nonexistent future.
    """
    if num_episodes <= 0 or steps_per_episode <= 0:
        raise ValueError("num_episodes and steps_per_episode must be positive.")
    rng = np.random.default_rng(seed)
    env = PrisonersDilemmaEnvironment()
    agent_1 = TwoStepSARSAAgent(**(agent_1_params or {}))
    agent_2 = TwoStepSARSAAgent(**(agent_2_params or {}))
    rewards_1: List[float] = []
    rewards_2: List[float] = []
    cooperation_1: List[float] = []
    cooperation_2: List[float] = []
    outcomes: List[Dict[str, int]] = []

    for _ in range(num_episodes):
        states_1: List[State] = []
        states_2: List[State] = []
        actions_1: List[Action] = []
        actions_2: List[Action] = []
        returns_1: List[float] = []
        returns_2: List[float] = []
        opponent_1: Optional[Action] = None
        opponent_2: Optional[Action] = None
        counts = {"CC": 0, "CD": 0, "DC": 0, "DD": 0}

        for step in range(steps_per_episode):
            state_1 = agent_1.state_from_opponent_action(opponent_1)
            state_2 = agent_2.state_from_opponent_action(opponent_2)
            action_1 = agent_1.choose_action(state_1, rng)
            action_2 = agent_2.choose_action(state_2, rng)
            reward_1, reward_2 = env.step(action_1, action_2)
            states_1.append(state_1)
            states_2.append(state_2)
            actions_1.append(action_1)
            actions_2.append(action_2)
            returns_1.append(reward_1)
            returns_2.append(reward_2)
            counts[action_1 + action_2] += 1

            # At step k, the action at k is the bootstrap action for tau=k-2.
            tau = step - 2
            if tau >= 0:
                _apply_n_step_update(agent_1, states_1, actions_1, returns_1, tau)
                _apply_n_step_update(agent_2, states_2, actions_2, returns_2, tau)
            opponent_1, opponent_2 = action_2, action_1

        # The final two transitions have no future bootstrap state/action.
        for tau in range(max(0, steps_per_episode - 2), steps_per_episode):
            _apply_n_step_update(agent_1, states_1, actions_1, returns_1, tau)
            _apply_n_step_update(agent_2, states_2, actions_2, returns_2, tau)

        agent_1.record_q_values()
        agent_2.record_q_values()
        agent_1.decay_exploration()
        agent_2.decay_exploration()
        rewards_1.append(float(sum(returns_1)))
        rewards_2.append(float(sum(returns_2)))
        cooperation_1.append(actions_1.count("C") / steps_per_episode)
        cooperation_2.append(actions_2.count("C") / steps_per_episode)
        outcomes.append(counts)

    return {
        "agent1": agent_1,
        "agent2": agent_2,
        "episode_rewards_1": rewards_1,
        "episode_rewards_2": rewards_2,
        "cooperation_rates_1": cooperation_1,
        "cooperation_rates_2": cooperation_2,
        "outcome_counts": outcomes,
        "seed": seed,
    }
