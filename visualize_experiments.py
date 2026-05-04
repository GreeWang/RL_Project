from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate and visualize report experiment groups.")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="experiment_figures")
    parser.add_argument("--window", type=int, default=50)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_episode_log(run_dir: Path) -> list[dict]:
    log_path = run_dir / "episode_log.csv"
    if not log_path.exists():
        return []
    with log_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict] = []
    for row in rows:
        if row.get("phase") != "train":
            continue
        parsed.append(
            {
                "episode": int(row["episode"]),
                "success": float(row["success"]),
                "hit_trap": float(row["hit_trap"]),
                "algorithm": row.get("algorithm", "q_learning"),
            }
        )
    return parsed


def rolling_mean(values: list[float], window: int) -> np.ndarray:
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    result = np.empty_like(arr)
    for idx in range(len(arr)):
        left = max(0, idx - window + 1)
        result[idx] = float(np.mean(arr[left : idx + 1]))
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def collect_algorithm_curves(results_dir: Path, window: int) -> dict[str, dict[int, list[tuple[float, float]]]]:
    curves: dict[str, dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for run_dir in sorted((results_dir / "algorithm_comparison").glob("*/seed_*")):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        algorithm = summary.get("algorithm", run_dir.parent.name)
        rows = load_episode_log(run_dir)
        if not rows:
            continue
        success_ma = rolling_mean([row["success"] for row in rows], window)
        trap_ma = rolling_mean([row["hit_trap"] for row in rows], window)
        for row, success_value, trap_value in zip(rows, success_ma, trap_ma):
            curves[algorithm][row["episode"]].append((float(success_value), float(trap_value)))
    return curves


def plot_algorithm_curves(curves: dict[str, dict[int, list[tuple[float, float]]]], output_dir: Path) -> None:
    if not curves:
        return
    colors = {"q_learning": "#1b9e77", "sarsa": "#d95f02"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for algorithm, by_episode in sorted(curves.items()):
        episodes = np.array(sorted(by_episode), dtype=np.int32)
        success = np.array([np.mean([values[0] for values in by_episode[episode]]) for episode in episodes])
        trap = np.array([np.mean([values[1] for values in by_episode[episode]]) for episode in episodes])
        color = colors.get(algorithm, None)
        axes[0].plot(episodes, success, label=algorithm, color=color, linewidth=2.0)
        axes[1].plot(episodes, trap, label=algorithm, color=color, linewidth=2.0)

    axes[0].set_title("Q-learning vs SARSA Success Rate")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Rolling success rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend()

    axes[1].set_title("Q-learning vs SARSA Trap Hit Rate")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Rolling trap hit rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()
    fig.savefig(output_dir / "algorithm_comparison_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def collect_trap_penalty_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for summary_path in sorted((results_dir / "trap_penalty").glob("*/trap_*/seed_*/summary.json")):
        summary = load_json(summary_path)
        rows.append(
            {
                "algorithm": summary.get("algorithm", summary_path.parents[2].name),
                "trap_penalty": float(summary["reward_config"]["trap_penalty"]),
                "seed": int(summary["seed"]),
                "success_rate": float(summary.get("final_eval_success_rate", summary.get("last100_success_rate", 0.0))),
                "avg_reward": float(summary.get("final_eval_avg_reward", summary.get("last100_avg_reward", 0.0))),
                "avg_steps": float(summary.get("final_eval_avg_steps", summary.get("last100_avg_steps", 0.0))),
                "trap_rate": float(summary.get("final_eval_trap_rate", summary.get("last100_trap_rate", 0.0))),
                "final_path_length": float(summary.get("final_path_length", 0.0)),
                "route_type": summary.get("route_type", "unknown"),
            }
        )
    return rows


def summarize_trap_penalty(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["algorithm"], row["trap_penalty"])].append(row)

    summary_rows: list[dict] = []
    for (algorithm, trap_penalty), group in sorted(grouped.items()):
        route_counts = Counter(row["route_type"] for row in group)
        summary_rows.append(
            {
                "algorithm": algorithm,
                "trap_penalty": trap_penalty,
                "success_rate_mean": float(np.mean([row["success_rate"] for row in group])),
                "avg_reward_mean": float(np.mean([row["avg_reward"] for row in group])),
                "avg_steps_mean": float(np.mean([row["avg_steps"] for row in group])),
                "trap_rate_mean": float(np.mean([row["trap_rate"] for row in group])),
                "final_path_length_mean": float(np.mean([row["final_path_length"] for row in group])),
                "route_counts": ";".join(f"{name}:{count}" for name, count in sorted(route_counts.items())),
                "n": len(group),
            }
        )
    return summary_rows


def plot_trap_penalty(summary_rows: list[dict], output_dir: Path) -> None:
    if not summary_rows:
        return
    metrics = [
        ("success_rate_mean", "Success Rate"),
        ("avg_reward_mean", "Average Reward"),
        ("avg_steps_mean", "Average Steps"),
        ("trap_rate_mean", "Trap Hit Rate"),
        ("final_path_length_mean", "Final Path Length"),
    ]
    algorithms = sorted({row["algorithm"] for row in summary_rows})
    colors = {"q_learning": "#1b9e77", "sarsa": "#d95f02"}
    markers = {"q_learning": "o", "sarsa": "D"}
    offsets = {"q_learning": -0.12, "sarsa": 0.12}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, (metric_key, title) in zip(axes_flat, metrics):
        for algorithm in algorithms:
            points = [row for row in summary_rows if row["algorithm"] == algorithm]
            points.sort(key=lambda row: row["trap_penalty"])
            penalties = np.array([row["trap_penalty"] for row in points], dtype=np.float64)
            values = np.array([row[metric_key] for row in points], dtype=np.float64)
            plot_penalties = penalties + offsets.get(algorithm, 0.0)
            ax.plot(
                plot_penalties,
                values,
                marker=markers.get(algorithm, "o"),
                linewidth=2.0,
                label=algorithm,
                color=colors.get(algorithm),
            )
        ax.set_title(title)
        ax.set_xlabel("Trap penalty")
        ax.set_xticks([-30, -10, -5])
        ax.grid(alpha=0.25)
    axes_flat[0].set_ylim(-0.05, 1.05)
    axes_flat[3].set_ylim(-0.05, 1.05)
    axes_flat[-1].axis("off")
    axes_flat[0].legend()
    fig.savefig(output_dir / "trap_penalty_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = ensure_dir(Path(args.output_dir))

    curves = collect_algorithm_curves(results_dir, args.window)
    plot_algorithm_curves(curves, output_dir)

    trap_rows = collect_trap_penalty_rows(results_dir)
    trap_summary = summarize_trap_penalty(trap_rows)
    write_csv(output_dir / "trap_penalty_runs.csv", trap_rows)
    write_csv(output_dir / "trap_penalty_summary.csv", trap_summary)
    plot_trap_penalty(trap_summary, output_dir)

    saved = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    print(f"Saved {len(saved)} file(s) to {output_dir}")
    for name in saved:
        print(name)


if __name__ == "__main__":
    main()
