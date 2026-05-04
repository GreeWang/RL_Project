from __future__ import annotations

from pathlib import Path

import numpy as np


class TabularTDAgent:
    algorithm = "td"

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        rng: np.random.Generator,
    ) -> None:
        if n_states <= 0 or n_actions <= 0:
            raise ValueError("n_states and n_actions must be positive.")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1].")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1].")
        if not (0.0 <= epsilon_end <= epsilon_start <= 1.0):
            raise ValueError("epsilon must satisfy 0 <= epsilon_end <= epsilon_start <= 1.")
        if not (0.0 < epsilon_decay <= 1.0):
            raise ValueError("epsilon_decay must be in (0, 1].")

        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.rng = rng
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    def select_action(self, state_id: int, greedy: bool = False) -> int:
        if greedy:
            return self._greedy_action(state_id)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self._greedy_action(state_id)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def greedy_policy(self) -> np.ndarray:
        return np.argmax(self.q_table, axis=1)

    def save(self, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            q_table=self.q_table,
            alpha=np.array(self.alpha),
            gamma=np.array(self.gamma),
            epsilon=np.array(self.epsilon),
            epsilon_start=np.array(self.epsilon_start),
            epsilon_end=np.array(self.epsilon_end),
            epsilon_decay=np.array(self.epsilon_decay),
            n_states=np.array(self.n_states),
            n_actions=np.array(self.n_actions),
            algorithm=np.array(self.algorithm),
        )

    @classmethod
    def load(cls, path: str):
        data = np.load(path)
        rng = np.random.default_rng(0)
        agent = cls(
            n_states=int(data["n_states"]),
            n_actions=int(data["n_actions"]),
            alpha=float(data["alpha"]),
            gamma=float(data["gamma"]),
            epsilon_start=float(data["epsilon_start"]),
            epsilon_end=float(data["epsilon_end"]),
            epsilon_decay=float(data["epsilon_decay"]),
            rng=rng,
        )
        agent.epsilon = float(data["epsilon"])
        agent.q_table = data["q_table"].astype(np.float64)
        return agent

    def _greedy_action(self, state_id: int) -> int:
        values = self.q_table[state_id]
        best_value = np.max(values)
        best_actions = np.flatnonzero(np.isclose(values, best_value))
        return int(self.rng.choice(best_actions))


class QLearningAgent(TabularTDAgent):
    algorithm = "q_learning"

    def update(
        self,
        state_id: int,
        action: int,
        reward: float,
        next_state_id: int,
        terminated: bool,
    ) -> None:
        current_value = self.q_table[state_id, action]
        bootstrap = 0.0 if terminated else np.max(self.q_table[next_state_id])
        target = reward + self.gamma * bootstrap
        self.q_table[state_id, action] = current_value + self.alpha * (target - current_value)


class SarsaAgent(TabularTDAgent):
    algorithm = "sarsa"

    def update(
        self,
        state_id: int,
        action: int,
        reward: float,
        next_state_id: int,
        next_action: int | None,
        terminated: bool,
    ) -> None:
        current_value = self.q_table[state_id, action]
        bootstrap = 0.0 if terminated or next_action is None else self.q_table[next_state_id, next_action]
        target = reward + self.gamma * bootstrap
        self.q_table[state_id, action] = current_value + self.alpha * (target - current_value)
