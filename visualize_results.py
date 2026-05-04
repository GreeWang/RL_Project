from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import ACTION_NAMES


ARROW_BY_ACTION = {
    0: "^",
    1: ">",
    2: "v",
    3: "<",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a training run from an output directory.")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Directory containing summary.json, episode_log.csv, and qtable_final.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save figures. Defaults to <run-dir>/figures.",
    )
    return parser.parse_args()


def load_summary(run_dir: Path) -> dict:
    with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_episode_log(run_dir: Path) -> list[dict]:
    with (run_dir / "episode_log.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in (
            "episode",
            "seed",
            "rows",
            "cols",
            "max_steps",
            "success",
            "terminated",
            "truncated",
            "hit_trap",
        ):
            row[key] = int(row[key])
        for key in ("alpha", "gamma", "epsilon", "total_reward", "steps"):
            row[key] = float(row[key])
        elapsed = row.get("elapsed_sec", "")
        row["elapsed_sec"] = float(elapsed) if elapsed else np.nan
    return rows


def load_qtable(run_dir: Path) -> np.ndarray:
    qtable = np.load(run_dir / "qtable_final.npz")["q_table"]
    return qtable.astype(np.float64)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    result = np.empty_like(values, dtype=np.float64)
    for idx in range(len(values)):
        left = max(0, idx - window + 1)
        result[idx] = float(np.mean(values[left : idx + 1]))
    return result


def build_training_figure(
    train_rows: list[dict],
    eval_rows: list[dict],
    summary: dict,
    output_path: Path,
) -> None:
    episodes = np.array([row["episode"] for row in train_rows], dtype=np.int32)
    rewards = np.array([row["total_reward"] for row in train_rows], dtype=np.float64)
    steps = np.array([row["steps"] for row in train_rows], dtype=np.float64)
    success = np.array([row["success"] for row in train_rows], dtype=np.float64)
    epsilon = np.array([row["epsilon"] for row in train_rows], dtype=np.float64)

    reward_ma = rolling_mean(rewards, window=min(5, len(rewards)))
    steps_ma = rolling_mean(steps, window=min(5, len(steps)))
    success_ma = rolling_mean(success, window=min(5, len(success)))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(f"Smoke Test Metrics: {summary['exp_name']} ({summary['run_id']})", fontsize=16)

    axes[0, 0].plot(episodes, rewards, color="#d95f02", alpha=0.45, label="reward")
    axes[0, 0].plot(episodes, reward_ma, color="#1b9e77", linewidth=2.0, label="reward MA")
    axes[0, 0].axhline(0.0, color="#444444", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Training Reward")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].legend()

    axes[0, 1].plot(episodes, steps, color="#7570b3", alpha=0.45, label="steps")
    axes[0, 1].plot(episodes, steps_ma, color="#1f78b4", linewidth=2.0, label="steps MA")
    axes[0, 1].axhline(
        summary["shortest_path_len"],
        color="#33a02c",
        linestyle="--",
        linewidth=1.5,
        label="shortest path",
    )
    axes[0, 1].axhline(
        train_rows[0]["max_steps"],
        color="#e31a1c",
        linestyle=":",
        linewidth=1.5,
        label="max steps",
    )
    axes[0, 1].set_title("Episode Length")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Steps")
    axes[0, 1].legend()

    axes[1, 0].plot(episodes, success, color="#6a3d9a", alpha=0.35, marker="o", label="success")
    axes[1, 0].plot(episodes, success_ma, color="#b15928", linewidth=2.0, label="success MA")
    axes[1, 0].plot(episodes, epsilon, color="#1f78b4", linewidth=1.5, label="epsilon")
    axes[1, 0].set_title("Success and Exploration")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].legend()

    axes[1, 1].axis("off")
    train_success_rate = float(np.mean(success)) if len(success) else 0.0
    trap_rate = float(np.mean([row["hit_trap"] for row in train_rows])) if train_rows else 0.0
    truncated_rate = float(np.mean([row["truncated"] for row in train_rows])) if train_rows else 0.0
    eval_text = "No eval data"
    if eval_rows:
        eval_text = (
            f"Best eval success: {summary['best_eval_success_rate']:.2f}\n"
            f"Best eval avg steps: {summary['best_eval_avg_steps']:.2f}"
        )

    info_lines = [
        f"Run: {summary['run_id']}",
        f"Map: {summary['map_name']} ({summary['map_spec']['rows']}x{summary['map_spec']['cols']})",
        f"Train episodes: {summary['episodes']}",
        f"Train success rate: {train_success_rate:.2f}",
        f"Train truncation rate: {truncated_rate:.2f}",
        f"Train trap rate: {trap_rate:.2f}",
        f"Final epsilon: {summary['final_epsilon']:.3f}",
        f"Shortest path length: {summary['shortest_path_len']}",
        eval_text,
    ]
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(info_lines),
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f7f7f7", "edgecolor": "#bbbbbb"},
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_eval_figure(
    eval_rows: list[dict],
    summary: dict,
    run_dir: Path,
    output_path: Path,
) -> None:
    if not eval_rows:
        return

    config_path = Path(summary["config_path"])
    if not config_path.is_absolute():
        config_path = run_dir / config_path.name
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    eval_every = int(config["eval_every"])
    eval_per_batch = int(config["eval_episodes"])

    rewards = np.array([row["total_reward"] for row in eval_rows], dtype=np.float64)
    steps = np.array([row["steps"] for row in eval_rows], dtype=np.float64)
    success = np.array([row["success"] for row in eval_rows], dtype=np.float64)
    batch_ids = np.arange(len(eval_rows)) // eval_per_batch
    eval_points = np.array([(batch_id + 1) * eval_every for batch_id in batch_ids], dtype=np.int32)

    unique_points = np.unique(eval_points)
    mean_rewards = np.array([np.mean(rewards[eval_points == point]) for point in unique_points], dtype=np.float64)
    mean_steps = np.array([np.mean(steps[eval_points == point]) for point in unique_points], dtype=np.float64)
    success_rates = np.array([np.mean(success[eval_points == point]) for point in unique_points], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    fig.suptitle("Eval Checkpoints", fontsize=15)

    axes[0].plot(unique_points, mean_rewards, marker="o", color="#e66101")
    axes[0].set_title("Eval Reward")
    axes[0].set_xlabel("Training Episode")
    axes[0].set_ylabel("Average Reward")

    axes[1].plot(unique_points, mean_steps, marker="o", color="#5e3c99")
    axes[1].axhline(summary["shortest_path_len"], linestyle="--", color="#33a02c", linewidth=1.2)
    axes[1].set_title("Eval Steps")
    axes[1].set_xlabel("Training Episode")
    axes[1].set_ylabel("Average Steps")

    axes[2].plot(unique_points, success_rates, marker="o", color="#1b9e77")
    axes[2].set_title("Eval Success Rate")
    axes[2].set_xlabel("Training Episode")
    axes[2].set_ylabel("Success Rate")
    axes[2].set_ylim(-0.05, 1.05)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_policy_figure(
    q_table: np.ndarray,
    summary: dict,
    output_path: Path,
) -> None:
    rows = int(summary["map_spec"]["rows"])
    cols = int(summary["map_spec"]["cols"])
    obstacles = {tuple(cell) for cell in summary["map_spec"]["obstacles"]}
    traps = {tuple(cell) for cell in summary["map_spec"]["traps"]}
    start = tuple(summary["map_spec"]["start"])
    goal = tuple(summary["map_spec"]["goal"])

    state_values = np.max(q_table, axis=1).reshape(rows, cols)
    policy = np.argmax(q_table, axis=1).reshape(rows, cols)
    final_path = [tuple(position) for position in summary.get("final_path", [])]

    masked_values = state_values.copy()
    for row, col in obstacles:
        masked_values[row, col] = np.nan

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad(color="#4d4d4d")
    image = ax.imshow(masked_values, cmap=cmap)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="State Value")

    for row in range(rows):
        for col in range(cols):
            cell = (row, col)
            if cell in obstacles:
                text = "#"
                color = "white"
            elif cell in traps:
                text = "T"
                color = "#b2182b"
            elif cell == start:
                text = "S"
                color = "#2166ac"
            elif cell == goal:
                text = "G"
                color = "#1b7837"
            else:
                text = ARROW_BY_ACTION[int(policy[row, col])]
                color = "black"
            ax.text(col, row, text, ha="center", va="center", fontsize=16, fontweight="bold", color=color)

    if len(final_path) > 1:
        path_rows = [position[0] for position in final_path]
        path_cols = [position[1] for position in final_path]
        ax.plot(path_cols, path_rows, color="#fdb863", linewidth=3.0, alpha=0.9, label="greedy path")
        ax.scatter(path_cols, path_rows, color="#fdb863", edgecolor="#5e3c99", s=45, zorder=3)

    title_parts = ["Final Policy and State Values"]
    if "algorithm" in summary:
        title_parts.append(str(summary["algorithm"]))
    if "route_type" in summary:
        title_parts.append(f"route={summary['route_type']}")
    ax.set_title(" | ".join(title_parts))
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    legend_lines = [
        "Legend:",
        f"S=start  G=goal  T=trap  #=obstacle",
        "Arrows show greedy action from final Q-table",
        "Orange line shows final greedy rollout path",
        f"Action order: {', '.join(f'{idx}={name}' for idx, name in enumerate(ACTION_NAMES))}",
    ]
    fig.text(0.02, 0.02, "\n".join(legend_lines), ha="left", va="bottom", fontsize=10, family="monospace")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve() if args.output_dir else run_dir / "figures")

    summary = load_summary(run_dir)
    rows = load_episode_log(run_dir)
    q_table = load_qtable(run_dir)

    train_rows = [row for row in rows if row["phase"] == "train"]
    eval_rows = [row for row in rows if row["phase"] == "eval"]

    build_training_figure(train_rows, eval_rows, summary, output_dir / "training_metrics.png")
    build_eval_figure(eval_rows, summary, run_dir, output_dir / "eval_metrics.png")
    build_policy_figure(q_table, summary, output_dir / "policy_value_heatmap.png")

    saved = sorted(path.name for path in output_dir.glob("*.png"))
    print(f"Saved {len(saved)} figure(s) to {output_dir}")
    for name in saved:
        print(name)


if __name__ == "__main__":
    main()
