from __future__ import annotations

from pathlib import Path

from train import parse_args, train_one_run


def test_train_smoke_run_creates_outputs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--exp-name",
            "smoke",
            "--episodes",
            "20",
            "--eval-every",
            "10",
            "--eval-episodes",
            "4",
            "--checkpoint-every",
            "10",
            "--outdir",
            str(tmp_path / "outputs"),
            "--seed",
            "7",
        ],
    )
    args = parse_args()
    summary = train_one_run(args)
    assert Path(summary["episode_log_path"]).exists()
    assert Path(summary["run_summary_path"]).exists()
    assert Path(summary["qtable_path"]).exists()
