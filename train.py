from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path

import numpy as np

from config import DEFAULTS, list_reward_presets
from environment import GridWorldEnv
from q_learning import QLearningAgent
from utils_io import ensure_dir, get_git_commit, save_json, write_csv_rows
from utils_seed import make_rng, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate tabular Q-Learning on GridWorld.")
    parser.add_argument("--exp-name", type=str, default=DEFAULTS.exp_name)
    parser.add_argument("--rows", type=int, default=DEFAULTS.rows)
    parser.add_argument("--cols", type=int, default=DEFAULTS.cols)
    parser.add_argument("--map-name", type=str, default=DEFAULTS.map_name, choices=["easy", "medium", "hard"])
    parser.add_argument(
        "--reward-setting",
        type=str,
        default=DEFAULTS.reward_setting,
        choices=list_reward_presets(),
    )
    parser.add_argument("--goal-reward", type=float, default=DEFAULTS.goal_reward)
    parser.add_argument("--trap-penalty", type=float, default=DEFAULTS.trap_penalty)
    parser.add_argument("--step-penalty", type=float, default=DEFAULTS.step_penalty)
    parser.add_argument("--wall-penalty", type=float, default=DEFAULTS.wall_penalty)
    parser.add_argument("--alpha", type=float, default=DEFAULTS.alpha)
    parser.add_argument("--gamma", type=float, default=DEFAULTS.gamma)
    parser.add_argument("--epsilon-start", type=float, default=DEFAULTS.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=DEFAULTS.epsilon_end)
    parser.add_argument("--epsilon-decay", type=float, default=DEFAULTS.epsilon_decay)
    parser.add_argument("--episodes", type=int, default=DEFAULTS.episodes)
    parser.add_argument("--max-steps", type=int, default=DEFAULTS.max_steps)
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
    parser.add_argument("--eval-every", type=int, default=DEFAULTS.eval_every)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULTS.eval_episodes)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULTS.checkpoint_every)
    parser.add_argument("--render-mode", type=str, default=DEFAULTS.render_mode, choices=["ansi", "human", "rgb_array"])
    parser.add_argument("--outdir", type=str, default=DEFAULTS.outdir)
    parser.add_argument("--save-qtable", dest="save_qtable", action="store_true")
    parser.add_argument("--no-save-qtable", dest="save_qtable", action="store_false")
    parser.set_defaults(save_qtable=DEFAULTS.save_qtable)
    parser.add_argument("--no-train", action="store_true", default=DEFAULTS.no_train)
    return parser.parse_args()


def make_env(args: argparse.Namespace) -> GridWorldEnv:
    return GridWorldEnv(
        rows=args.rows,
        cols=args.cols,
        map_name=args.map_name,
        reward_setting=args.reward_setting,
        max_steps=args.max_steps,
        render_mode=args.render_mode,
        goal_reward=args.goal_reward,
        trap_penalty=args.trap_penalty,
        step_penalty=args.step_penalty,
        wall_penalty=args.wall_penalty,
    )


def save_episode_log(rows: list[dict], csv_path: str) -> None:
    write_csv_rows(rows, csv_path)


def save_summary(summary: dict, path: str) -> None:
    save_json(summary, path)


def evaluate_policy(env: GridWorldEnv, agent: QLearningAgent, episodes: int, seed: int) -> dict:
    rewards: list[float] = []
    steps: list[int] = []
    successes = 0
    terminated_count = 0
    truncated_count = 0
    trap_count = 0
    logs: list[dict] = []

    for episode_idx in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed + episode_idx)
        state_id = env.obs_to_state_id(obs)
        total_reward = 0.0
        step_count = 0
        terminated = False
        truncated = False
        hit_trap = False

        while not (terminated or truncated):
            action = agent.select_action(state_id, greedy=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            state_id = env.obs_to_state_id(next_obs)
            total_reward += reward
            step_count += 1
            hit_trap = bool(info["hit_trap"])

        successes += int(tuple(env.position) == env.goal and terminated)
        terminated_count += int(terminated)
        truncated_count += int(truncated)
        trap_count += int(hit_trap)
        rewards.append(total_reward)
        steps.append(step_count)
        logs.append(
            {
                "phase": "eval",
                "episode": episode_idx,
                "total_reward": total_reward,
                "steps": step_count,
                "success": int(tuple(env.position) == env.goal and terminated),
                "terminated": int(terminated),
                "truncated": int(truncated),
                "hit_trap": int(hit_trap),
                "epsilon": 0.0,
            }
        )

    return {
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_steps": float(np.mean(steps)) if steps else 0.0,
        "success_rate": successes / episodes if episodes else 0.0,
        "terminated_rate": terminated_count / episodes if episodes else 0.0,
        "truncated_rate": truncated_count / episodes if episodes else 0.0,
        "trap_rate": trap_count / episodes if episodes else 0.0,
        "episodes": episodes,
        "logs": logs,
    }


def train_one_run(args: argparse.Namespace) -> dict:
    set_global_seed(args.seed)
    outdir = ensure_dir(args.outdir)
    run_id = f"{args.exp_name}_{args.seed}_{uuid.uuid4().hex[:8]}"
    env = make_env(args)
    rng = make_rng(args.seed)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        rng=rng,
    )

    config_snapshot = vars(args).copy()
    config_snapshot["map_spec"] = env.map_spec.as_dict()
    config_snapshot["git_commit"] = get_git_commit(Path(__file__).resolve().parent)
    config_path = outdir / "config.json"
    save_json(config_snapshot, config_path)

    episode_logs: list[dict] = []
    eval_logs: list[dict] = []
    qtable_paths: list[str] = []
    best_eval = {"success_rate": -1.0, "avg_steps": float("inf")}

    train_episodes = 0 if args.no_train else args.episodes

    for episode_idx in range(1, train_episodes + 1):
        start_time = time.perf_counter()
        obs, _ = env.reset(seed=args.seed + episode_idx)
        state_id = env.obs_to_state_id(obs)
        total_reward = 0.0
        step_count = 0
        terminated = False
        truncated = False
        hit_trap = False
        qtable_path = ""

        while not (terminated or truncated):
            action = agent.select_action(state_id, greedy=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state_id = env.obs_to_state_id(next_obs)
            agent.update(
                state_id=state_id,
                action=action,
                reward=reward,
                next_state_id=next_state_id,
                terminated=terminated,
            )
            state_id = next_state_id
            total_reward += reward
            step_count += 1
            hit_trap = bool(info["hit_trap"])

        agent.decay_epsilon()
        elapsed_sec = time.perf_counter() - start_time

        if args.save_qtable and episode_idx % args.checkpoint_every == 0:
            checkpoint_path = outdir / f"qtable_ep{episode_idx}.npz"
            agent.save(str(checkpoint_path))
            qtable_path = str(checkpoint_path)
            qtable_paths.append(qtable_path)

        success = int(tuple(env.position) == env.goal and terminated)
        episode_logs.append(
            {
                "run_id": run_id,
                "exp_name": args.exp_name,
                "phase": "train",
                "episode": episode_idx,
                "seed": args.seed,
                "map_name": args.map_name,
                "rows": args.rows,
                "cols": args.cols,
                "alpha": args.alpha,
                "gamma": args.gamma,
                "epsilon": agent.epsilon,
                "reward_setting": args.reward_setting,
                "max_steps": args.max_steps,
                "total_reward": total_reward,
                "steps": step_count,
                "success": success,
                "terminated": int(terminated),
                "truncated": int(truncated),
                "hit_trap": int(hit_trap),
                "elapsed_sec": elapsed_sec,
                "qtable_path": qtable_path,
            }
        )

        if args.eval_every > 0 and episode_idx % args.eval_every == 0:
            eval_result = evaluate_policy(
                env=env,
                agent=agent,
                episodes=args.eval_episodes,
                seed=args.seed + 10_000 + episode_idx,
            )
            eval_logs.extend(
                {
                    "run_id": run_id,
                    "exp_name": args.exp_name,
                    "seed": args.seed,
                    "map_name": args.map_name,
                    "rows": args.rows,
                    "cols": args.cols,
                    "alpha": args.alpha,
                    "gamma": args.gamma,
                    "reward_setting": args.reward_setting,
                    "max_steps": args.max_steps,
                    **row,
                }
                for row in eval_result["logs"]
            )
            if eval_result["success_rate"] > best_eval["success_rate"]:
                best_eval = {
                    "success_rate": eval_result["success_rate"],
                    "avg_steps": eval_result["avg_steps"],
                }

    final_qtable_path = ""
    if args.save_qtable:
        final_qtable = outdir / "qtable_final.npz"
        agent.save(str(final_qtable))
        final_qtable_path = str(final_qtable)

    all_logs = episode_logs + eval_logs
    episode_log_path = outdir / "episode_log.csv"
    save_episode_log(all_logs, str(episode_log_path))

    last_window = episode_logs[-100:] if episode_logs else []
    last100_avg_reward = float(np.mean([row["total_reward"] for row in last_window])) if last_window else 0.0
    last100_success_rate = float(np.mean([row["success"] for row in last_window])) if last_window else 0.0
    last100_avg_steps = float(np.mean([row["steps"] for row in last_window])) if last_window else 0.0

    summary_row = {
        "run_id": run_id,
        "exp_name": args.exp_name,
        "seed": args.seed,
        "map_name": args.map_name,
        "episodes": train_episodes,
        "last100_avg_reward": last100_avg_reward,
        "last100_success_rate": last100_success_rate,
        "last100_avg_steps": last100_avg_steps,
        "best_eval_success_rate": best_eval["success_rate"] if best_eval["success_rate"] >= 0 else 0.0,
        "best_eval_avg_steps": best_eval["avg_steps"] if np.isfinite(best_eval["avg_steps"]) else 0.0,
        "final_epsilon": agent.epsilon,
        "qtable_path": final_qtable_path,
        "config_path": str(config_path),
        "git_commit": config_snapshot["git_commit"],
    }
    run_summary_path = outdir / "run_summary.csv"
    write_csv_rows([summary_row], run_summary_path)

    summary_json = {
        **summary_row,
        "map_spec": env.map_spec.as_dict(),
        "reward_config": env.rewards,
        "shortest_path_len": env.shortest_path_len(),
        "checkpoints": qtable_paths,
        "episode_log_path": str(episode_log_path),
        "run_summary_path": str(run_summary_path),
    }
    save_summary(summary_json, outdir / "summary.json")
    env.close()
    return summary_json


def main() -> None:
    args = parse_args()
    summary = train_one_run(args)
    print("Run complete")
    print(f"run_id={summary['run_id']}")
    print(f"episode_log={summary['episode_log_path']}")
    print(f"run_summary={summary['run_summary_path']}")
    print(f"qtable={summary['qtable_path']}")
    print(f"best_eval_success_rate={summary['best_eval_success_rate']:.3f}")


if __name__ == "__main__":
    main()
