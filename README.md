# RL Project

这是一个按实施计划落地的 `GridWorld + tabular Q-Learning` 最小工程实现，包含：

- Gymnasium 风格环境 `GridWorldEnv`
- 可配置的地图 preset、奖励 preset、训练超参数
- 命令行训练入口 `train.py`
- 标准化输出：`episode_log.csv`、`run_summary.csv`、`summary.json`、`qtable_final.npz`
- 基础单元测试与训练 smoke test

## 项目结构

```text
RL_Project/
├── config.py
├── environment.py
├── q_learning.py
├── train.py
├── utils_io.py
├── utils_seed.py
├── requirements.txt
├── tests/
│   ├── test_environment.py
│   ├── test_q_learning.py
│   └── test_train_integration.py
└── outputs/
```

## 安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 训练

```bash
python train.py ^
  --exp-name baseline_easy ^
  --rows 5 --cols 5 ^
  --map-name easy ^
  --reward-setting default ^
  --goal-reward 10 --trap-penalty -10 --step-penalty -1 --wall-penalty -1 ^
  --alpha 0.1 --gamma 0.95 ^
  --epsilon-start 0.2 --epsilon-end 0.05 --epsilon-decay 0.995 ^
  --episodes 1000 --max-steps 50 ^
  --seed 42 ^
  --eval-every 100 --eval-episodes 20 ^
  --checkpoint-every 200 ^
  --outdir outputs/baseline_easy
```

## 快速 smoke test

```bash
python train.py --episodes 20 --eval-every 10 --eval-episodes 4 --outdir outputs/smoke
pytest -q
```

## 关键参数

- `--map-name`: `easy` / `medium` / `hard`
- `--reward-setting`: `default` / `aggressive` / `safe`
- `--alpha`, `--gamma`: Q-Learning 核心超参数
- `--epsilon-start`, `--epsilon-end`, `--epsilon-decay`: epsilon-greedy 探索调度
- `--eval-every`, `--eval-episodes`: 周期性 greedy 评估
- `--checkpoint-every`: 保存中间 `Q-table` 的频率
- `--no-train`: 只做环境和输出链路检查

## 输出文件

- `episode_log.csv`: 逐 episode 训练和评估记录
- `run_summary.csv`: 单次运行摘要
- `summary.json`: 结构化汇总，包含地图、奖励和 checkpoint 信息
- `qtable_final.npz`: 最终 Q-table
- `qtable_ep*.npz`: 周期性 checkpoint

## 环境定义

- 动作编码：`0=up`, `1=right`, `2=down`, `3=left`
- `terminated=True`: 到达目标或踩中陷阱
- `truncated=True`: 达到 `max_steps`
- 观察值：当前位置 `[row, col]`
- 状态 ID：`row * cols + col`

## 测试内容

- 环境 reset/step、墙体、陷阱、目标、步数截断
- Q 更新公式、终止状态 bootstrap 关闭、epsilon 衰减
- 一轮小规模训练是否能生成标准输出
