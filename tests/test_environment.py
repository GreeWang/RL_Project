from __future__ import annotations

import numpy as np

from environment import GridWorldEnv


def test_reset_is_reproducible_with_seed() -> None:
    env = GridWorldEnv(rows=5, cols=5, map_name="easy", max_steps=20)
    obs1, info1 = env.reset(seed=123)
    obs2, info2 = env.reset(seed=123)
    np.testing.assert_array_equal(obs1, obs2)
    assert info1["state_id"] == info2["state_id"]


def test_wall_or_obstacle_keeps_agent_in_place() -> None:
    env = GridWorldEnv(rows=5, cols=5, map_name="easy", max_steps=20)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(3)
    np.testing.assert_array_equal(obs, np.array([0, 0], dtype=np.int32))
    assert reward == env.rewards["wall_penalty"]
    assert not terminated
    assert not truncated
    assert info["invalid_move"] is True


def test_goal_terminates_episode() -> None:
    env = GridWorldEnv(rows=5, cols=5, map_name="easy", max_steps=20)
    obs, _ = env.reset(seed=0, options={"start_position": (4, 3)})
    assert env.obs_to_state_id(obs) == 4 * env.cols + 3
    _, reward, terminated, truncated, info = env.step(1)
    assert reward == env.rewards["goal_reward"]
    assert terminated is True
    assert truncated is False
    assert info["position"] == env.goal


def test_trap_terminates_episode() -> None:
    env = GridWorldEnv(rows=5, cols=5, map_name="easy", max_steps=20)
    trap = next(iter(env.traps))
    start = (trap[0], trap[1] - 1)
    env.reset(seed=0, options={"start_position": start})
    _, reward, terminated, truncated, info = env.step(1)
    assert reward == env.rewards["trap_penalty"]
    assert terminated is True
    assert truncated is False
    assert info["hit_trap"] is True


def test_max_steps_truncates_episode() -> None:
    env = GridWorldEnv(rows=5, cols=5, map_name="easy", max_steps=1)
    env.reset(seed=0)
    _, _, terminated, truncated, _ = env.step(1)
    assert terminated is False
    assert truncated is True


def test_shortest_path_is_found() -> None:
    env = GridWorldEnv(rows=5, cols=5, map_name="easy", max_steps=20)
    shortest = env.shortest_path_len()
    assert shortest > 0


def test_risk_map_builds_with_valid_route() -> None:
    env = GridWorldEnv(rows=5, cols=5, map_name="risk", max_steps=30)
    assert env.start == (0, 0)
    assert env.goal == (0, 4)
    assert env.start not in env.obstacles
    assert env.goal not in env.obstacles
    assert env.start not in env.traps
    assert env.goal not in env.traps
    assert env.shortest_path_len() > 0
