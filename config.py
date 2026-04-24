from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


ACTION_NAMES = ("up", "right", "down", "left")
ACTION_DELTAS = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}


@dataclass(frozen=True)
class RewardPreset:
    name: str
    goal_reward: float
    trap_penalty: float
    step_penalty: float
    wall_penalty: float


@dataclass(frozen=True)
class MapSpec:
    name: str
    rows: int
    cols: int
    start: tuple[int, int]
    goal: tuple[int, int]
    obstacles: tuple[tuple[int, int], ...]
    traps: tuple[tuple[int, int], ...]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainingDefaults:
    exp_name: str = "baseline"
    rows: int = 5
    cols: int = 5
    map_name: str = "easy"
    reward_setting: str = "default"
    goal_reward: float = 10.0
    trap_penalty: float = -10.0
    step_penalty: float = -1.0
    wall_penalty: float = -1.0
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 0.2
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    episodes: int = 1000
    max_steps: int = 50
    seed: int = 42
    eval_every: int = 100
    eval_episodes: int = 20
    checkpoint_every: int = 200
    render_mode: str | None = None
    outdir: str = "outputs/baseline"
    save_qtable: bool = True
    no_train: bool = False


DEFAULTS = TrainingDefaults()

REWARD_PRESETS: dict[str, RewardPreset] = {
    "default": RewardPreset(
        name="default",
        goal_reward=10.0,
        trap_penalty=-10.0,
        step_penalty=-1.0,
        wall_penalty=-1.0,
    ),
    "aggressive": RewardPreset(
        name="aggressive",
        goal_reward=15.0,
        trap_penalty=-12.0,
        step_penalty=-1.5,
        wall_penalty=-2.0,
    ),
    "safe": RewardPreset(
        name="safe",
        goal_reward=8.0,
        trap_penalty=-15.0,
        step_penalty=-0.8,
        wall_penalty=-1.5,
    ),
}


def list_reward_presets() -> tuple[str, ...]:
    return tuple(REWARD_PRESETS.keys())


def get_reward_preset(name: str) -> RewardPreset:
    try:
        return REWARD_PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(list_reward_presets())
        raise ValueError(f"Unknown reward setting '{name}'. Choices: {choices}.") from exc


def resolve_rewards(
    reward_setting: str,
    *,
    goal_reward: float | None = None,
    trap_penalty: float | None = None,
    step_penalty: float | None = None,
    wall_penalty: float | None = None,
) -> dict[str, float]:
    preset = get_reward_preset(reward_setting)
    return {
        "goal_reward": preset.goal_reward if goal_reward is None else goal_reward,
        "trap_penalty": preset.trap_penalty if trap_penalty is None else trap_penalty,
        "step_penalty": preset.step_penalty if step_penalty is None else step_penalty,
        "wall_penalty": preset.wall_penalty if wall_penalty is None else wall_penalty,
    }


def _filter_cells(
    cells: Iterable[tuple[int, int]],
    *,
    rows: int,
    cols: int,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    kept: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for row, col in cells:
        if not (0 <= row < rows and 0 <= col < cols):
            continue
        if (row, col) in (start, goal):
            continue
        if (row, col) in seen:
            continue
        kept.append((row, col))
        seen.add((row, col))
    return tuple(kept)


def build_map_spec(name: str, rows: int, cols: int) -> MapSpec:
    if rows < 4 or cols < 4:
        raise ValueError("Grid size must be at least 4x4.")

    start = (0, 0)
    goal = (rows - 1, cols - 1)
    mid_row = rows // 2
    mid_col = cols // 2

    if name == "easy":
        obstacles = (
            (1, 1),
            (2, 1),
            (rows - 2, cols - 3),
        )
        traps = (
            (mid_row, cols - 2),
        )
    elif name == "medium":
        obstacles = (
            (1, 1),
            (1, 2),
            (2, 1),
            (mid_row, mid_col),
            (rows - 2, cols - 3),
        )
        traps = (
            (mid_row - 1, cols - 2),
            (rows - 2, 1),
        )
    elif name == "hard":
        obstacles = (
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 1),
            (2, mid_col),
            (mid_row, 2),
            (mid_row, 3),
            (rows - 2, cols - 3),
            (rows - 3, cols - 3),
        )
        traps = (
            (mid_row - 1, cols - 2),
            (rows - 2, 1),
            (rows - 3, cols - 2),
        )
    else:
        raise ValueError("Unknown map name '{}'. Choices: easy, medium, hard.".format(name))

    filtered_obstacles = _filter_cells(
        obstacles,
        rows=rows,
        cols=cols,
        start=start,
        goal=goal,
    )
    filtered_traps = _filter_cells(
        traps,
        rows=rows,
        cols=cols,
        start=start,
        goal=goal,
    )
    trap_set = set(filtered_traps)
    filtered_obstacles = tuple(cell for cell in filtered_obstacles if cell not in trap_set)

    return MapSpec(
        name=name,
        rows=rows,
        cols=cols,
        start=start,
        goal=goal,
        obstacles=filtered_obstacles,
        traps=filtered_traps,
    )
