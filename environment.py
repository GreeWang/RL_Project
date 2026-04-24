from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import ACTION_DELTAS, ACTION_NAMES, MapSpec, build_map_spec, resolve_rewards


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        rows: int = 5,
        cols: int = 5,
        map_name: str = "easy",
        reward_setting: str = "default",
        max_steps: int = 50,
        render_mode: str | None = None,
        goal_reward: float | None = None,
        trap_penalty: float | None = None,
        step_penalty: float | None = None,
        wall_penalty: float | None = None,
    ) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render mode '{render_mode}'. Choices: {self.metadata['render_modes']}."
            )
        self.map_spec: MapSpec = build_map_spec(map_name, rows, cols)
        self.rows = rows
        self.cols = cols
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.reward_setting = reward_setting
        self.rewards = resolve_rewards(
            reward_setting,
            goal_reward=goal_reward,
            trap_penalty=trap_penalty,
            step_penalty=step_penalty,
            wall_penalty=wall_penalty,
        )
        self.obstacles = set(self.map_spec.obstacles)
        self.traps = set(self.map_spec.traps)
        self.goal = self.map_spec.goal
        self.start = self.map_spec.start
        self.position = self.start
        self.steps_taken = 0
        self.last_action: int | None = None

        high = np.array([rows - 1, cols - 1], dtype=np.int32)
        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.int32),
            high=high,
            shape=(2,),
            dtype=np.int32,
        )
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.action_space.seed(0)

    @property
    def n_states(self) -> int:
        return self.rows * self.cols

    @property
    def n_actions(self) -> int:
        return int(self.action_space.n)

    def _get_obs(self) -> np.ndarray:
        return np.array(self.position, dtype=np.int32)

    def _get_info(
        self,
        *,
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        hit_trap: bool = False,
        invalid_move: bool = False,
    ) -> dict[str, Any]:
        return {
            "position": tuple(self.position),
            "state_id": self.obs_to_state_id(self._get_obs()),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "hit_trap": hit_trap,
            "invalid_move": invalid_move,
            "steps_taken": self.steps_taken,
            "goal": tuple(self.goal),
            "shortest_path_len": self.shortest_path_len(),
            "action_name": None if self.last_action is None else ACTION_NAMES[self.last_action],
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)

        start_position = self.start
        if options and "start_position" in options:
            candidate = tuple(options["start_position"])
            if candidate in self.obstacles:
                raise ValueError("start_position cannot be an obstacle.")
            if candidate in self.traps:
                raise ValueError("start_position cannot be a trap.")
            if candidate == self.goal:
                raise ValueError("start_position cannot be the goal.")
            row, col = candidate
            if not (0 <= row < self.rows and 0 <= col < self.cols):
                raise ValueError("start_position must stay inside the grid.")
            start_position = candidate

        self.position = start_position
        self.steps_taken = 0
        self.last_action = None
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}.")

        self.last_action = int(action)
        self.steps_taken += 1
        delta_row, delta_col = ACTION_DELTAS[self.last_action]
        next_position = (self.position[0] + delta_row, self.position[1] + delta_col)

        invalid_move = False
        hit_trap = False
        terminated = False
        truncated = False

        if not self._in_bounds(next_position) or next_position in self.obstacles:
            reward = self.rewards["wall_penalty"]
            invalid_move = True
            next_position = self.position
        elif next_position == self.goal:
            self.position = next_position
            reward = self.rewards["goal_reward"]
            terminated = True
        elif next_position in self.traps:
            self.position = next_position
            reward = self.rewards["trap_penalty"]
            hit_trap = True
            terminated = True
        else:
            self.position = next_position
            reward = self.rewards["step_penalty"]

        if not terminated and self.steps_taken >= self.max_steps:
            truncated = True

        observation = self._get_obs()
        info = self._get_info(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            hit_trap=hit_trap,
            invalid_move=invalid_move,
        )
        return observation, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "human":
            frame = self._render_ansi()
            print(frame)
            return None
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        return self._render_ansi()

    def close(self) -> None:
        return None

    def obs_to_state_id(self, obs: np.ndarray) -> int:
        row, col = int(obs[0]), int(obs[1])
        if not self._in_bounds((row, col)):
            raise ValueError(f"Observation {obs.tolist()} is out of bounds.")
        return row * self.cols + col

    def shortest_path_len(self) -> int:
        queue: deque[tuple[tuple[int, int], int]] = deque([(self.start, 0)])
        visited = {self.start}
        blocked = self.obstacles | self.traps

        while queue:
            position, distance = queue.popleft()
            if position == self.goal:
                return distance
            for delta_row, delta_col in ACTION_DELTAS.values():
                neighbor = (position[0] + delta_row, position[1] + delta_col)
                if not self._in_bounds(neighbor):
                    continue
                if neighbor in blocked or neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
        return -1

    def _in_bounds(self, position: tuple[int, int]) -> bool:
        row, col = position
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _render_ansi(self) -> str:
        cells: list[str] = []
        for row in range(self.rows):
            parts: list[str] = []
            for col in range(self.cols):
                cell = (row, col)
                if cell == self.position:
                    parts.append("A")
                elif cell == self.goal:
                    parts.append("G")
                elif cell == self.start:
                    parts.append("S")
                elif cell in self.obstacles:
                    parts.append("#")
                elif cell in self.traps:
                    parts.append("T")
                else:
                    parts.append(".")
            cells.append(" ".join(parts))
        return "\n".join(cells)

    def _render_rgb_array(self) -> np.ndarray:
        cell_size = 20
        frame = np.zeros((self.rows * cell_size, self.cols * cell_size, 3), dtype=np.uint8)
        colors = {
            "empty": np.array([245, 245, 245], dtype=np.uint8),
            "start": np.array([66, 165, 245], dtype=np.uint8),
            "goal": np.array([67, 160, 71], dtype=np.uint8),
            "obstacle": np.array([66, 66, 66], dtype=np.uint8),
            "trap": np.array([229, 57, 53], dtype=np.uint8),
            "agent": np.array([255, 179, 0], dtype=np.uint8),
        }
        for row in range(self.rows):
            for col in range(self.cols):
                cell = (row, col)
                color = colors["empty"]
                if cell == self.start:
                    color = colors["start"]
                if cell == self.goal:
                    color = colors["goal"]
                if cell in self.obstacles:
                    color = colors["obstacle"]
                if cell in self.traps:
                    color = colors["trap"]
                if cell == self.position:
                    color = colors["agent"]
                row_slice = slice(row * cell_size, (row + 1) * cell_size)
                col_slice = slice(col * cell_size, (col + 1) * cell_size)
                frame[row_slice, col_slice] = color
        return frame
