from __future__ import annotations

import numpy as np

from q_learning import QLearningAgent


def test_update_matches_q_learning_formula() -> None:
    rng = np.random.default_rng(0)
    agent = QLearningAgent(
        n_states=4,
        n_actions=2,
        alpha=0.5,
        gamma=0.9,
        epsilon_start=0.2,
        epsilon_end=0.1,
        epsilon_decay=0.99,
        rng=rng,
    )
    agent.q_table[1, 0] = 4.0
    agent.update(state_id=0, action=1, reward=2.0, next_state_id=1, terminated=False)
    expected = 0.0 + 0.5 * ((2.0 + 0.9 * 4.0) - 0.0)
    assert np.isclose(agent.q_table[0, 1], expected)


def test_update_disables_bootstrap_when_terminated() -> None:
    rng = np.random.default_rng(0)
    agent = QLearningAgent(
        n_states=4,
        n_actions=2,
        alpha=1.0,
        gamma=0.9,
        epsilon_start=0.2,
        epsilon_end=0.1,
        epsilon_decay=0.99,
        rng=rng,
    )
    agent.q_table[1, 0] = 9.0
    agent.update(state_id=0, action=1, reward=2.0, next_state_id=1, terminated=True)
    assert agent.q_table[0, 1] == 2.0


def test_epsilon_decay_respects_floor() -> None:
    rng = np.random.default_rng(0)
    agent = QLearningAgent(
        n_states=4,
        n_actions=2,
        alpha=0.5,
        gamma=0.9,
        epsilon_start=0.2,
        epsilon_end=0.1,
        epsilon_decay=0.5,
        rng=rng,
    )
    agent.decay_epsilon()
    agent.decay_epsilon()
    assert agent.epsilon == 0.1


def test_greedy_policy_uses_argmax() -> None:
    rng = np.random.default_rng(0)
    agent = QLearningAgent(
        n_states=3,
        n_actions=2,
        alpha=0.5,
        gamma=0.9,
        epsilon_start=0.2,
        epsilon_end=0.1,
        epsilon_decay=0.9,
        rng=rng,
    )
    agent.q_table[:] = np.array([[1.0, 2.0], [5.0, 4.0], [0.0, 1.0]])
    np.testing.assert_array_equal(agent.greedy_policy(), np.array([1, 0, 1]))
