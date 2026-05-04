# RL Project

这是一个按实施计划落地的 `GridWorld + tabular TD control` 最小工程实现，包含：

- Gymnasium 风格环境 `GridWorldEnv`
- 可配置的地图 preset、奖励 preset、训练超参数
- 命令行训练入口 `train.py`，支持 Q-learning 与 SARSA
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
  --algorithm q_learning ^
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

- `--algorithm`: `q_learning` / `sarsa`
- `--map-name`: `easy` / `medium` / `hard` / `risk`
- `--reward-setting`: `default` / `aggressive` / `safe`
- `--alpha`, `--gamma`: Q-Learning 核心超参数
- `--epsilon-start`, `--epsilon-end`, `--epsilon-decay`: epsilon-greedy 探索调度
- `--eval-every`, `--eval-episodes`: 周期性 greedy 评估
- `--checkpoint-every`: 保存中间 `Q-table` 的频率
- `--no-train`: 只做环境和输出链路检查

## 新增报告实验

```bash
bash run_experiments.sh
python visualize_experiments.py --results-dir results --output-dir experiment_figures
```

新增两组实验输出：

- `results/algorithm_comparison/`: hard map 上的 Q-learning vs SARSA。
- `results/trap_penalty/`: risk map 上 `trap_penalty=-5/-10/-30` 的风险敏感性实验。

聚合图表和表格会写入 `experiment_figures/`，包括 success rate、trap hit rate、trap penalty 汇总表和最终路径类型。

## 现有实验结果

当前已使用 seeds `42, 123, 456` 跑完新增两组实验。所有结果保存在 `results/algorithm_comparison/`、`results/trap_penalty/`，聚合表格保存在 `experiment_figures/trap_penalty_summary.csv` 和 `experiment_figures/trap_penalty_runs.csv`。

### Q-learning vs SARSA

设置：`hard` map，dense reward，`alpha=0.1`，`gamma=0.9`，epsilon 从 `1.0` 衰减到 `0.05`，`episodes=1000`。

| Algorithm | Final eval success rate | Final eval avg reward | Final eval avg steps | Final eval trap rate | Final route type |
| --- | ---: | ---: | ---: | ---: | --- |
| Q-learning | 1.00 | 9.30 | 8.00 | 0.00 | risky 3/3 |
| SARSA | 1.00 | 9.30 | 8.00 | 0.00 | risky 3/3 |

在当前 `hard` map 上，两种算法最终都能稳定到达目标，且 greedy evaluation 中没有踩中陷阱。这个设置下二者最终性能接近，因此更适合用 success rate 曲线和 trap hit rate 曲线观察训练过程差异，而不是只比较最终平均值。

### Trap Penalty / Risk Sensitivity

设置：`risk` map，dense reward，`trap_penalty=-5/-10/-30`，其他训练参数与上表一致。

| Algorithm | Trap penalty | Final eval success rate | Final eval avg reward | Final eval avg steps | Final path length | Final route type |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Q-learning | -5 | 1.00 | 9.70 | 4.00 | 4.00 | risky 3/3 |
| Q-learning | -10 | 1.00 | 9.70 | 4.00 | 4.00 | risky 3/3 |
| Q-learning | -30 | 1.00 | 9.70 | 4.00 | 4.00 | risky 3/3 |
| SARSA | -5 | 1.00 | 9.70 | 4.00 | 4.00 | risky 3/3 |
| SARSA | -10 | 1.00 | 9.50 | 6.00 | 6.00 | risky 2/3, safe 1/3 |
| SARSA | -30 | 1.00 | 9.10 | 10.00 | 10.00 | safe 3/3 |

这个实验呈现出更明显的 on-policy/off-policy 差异。Q-learning 在三个 trap penalty 下都保持 4-step risky path，说明它更倾向于学习理论上回报最高的短路径；SARSA 随着 trap penalty 从 `-5` 加大到 `-30`，最终路径从 risky 转向 safe，平均步数从 `4.00` 增加到 `10.00`。这支持报告中的结论：SARSA 的 on-policy 更新会把探索期间可能发生的风险反映进价值估计，因此在高陷阱惩罚环境中更容易学到保守、安全的路径。

### SARSA Hyperparameter Tuning

设置：沿用原 Q-learning 调参实验口径，`episodes=800`，`eval_every=20`，`eval_episodes=10`，`max_steps=100`。新增结果保存在 `results/sarsa_alpha_comparison/`、`results/sarsa_gamma_comparison/`、`results/sarsa_epsilon_comparison/`。

| Experiment | Setting | Seeds | Final eval success rate | Final eval avg reward | Final eval avg steps | Last100 success rate | Convergence episodes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| SARSA alpha | `alpha=0.05` | 3 | 1.00 | 3.00 | 8.00 | 0.987 | 201, 198, 205 |
| SARSA alpha | `alpha=0.1` | 3 | 1.00 | 3.00 | 8.00 | 0.980 | 154, 152, 150 |
| SARSA alpha | `alpha=0.5` | 3 | 1.00 | 3.00 | 8.00 | 1.000 | 115, 121, 120 |
| SARSA gamma | `gamma=0.5` | 5 | 1.00 | 3.00 | 8.00 | 0.986 | 194, 178, 186, 186, 194 |
| SARSA gamma | `gamma=0.8` | 5 | 1.00 | 3.00 | 8.00 | 0.992 | 157, 153, 159, 157, 160 |
| SARSA gamma | `gamma=0.95` | 5 | 1.00 | 3.00 | 8.00 | 0.972 | 154, 152, 147, 150, 149 |
| SARSA gamma | `gamma=0.99` | 5 | 1.00 | 3.00 | 8.00 | 0.974 | 146, 146, 153, 143, 144 |
| SARSA epsilon decay | `decay=0.95` | 3 | 1.00 | 3.00 | 8.00 | 0.990 | 144, 142, 143 |
| SARSA epsilon decay | `decay=0.99` | 3 | 1.00 | 3.00 | 8.00 | 1.000 | 214, 220, 211 |
| SARSA epsilon decay | `decay=0.995` | 3 | 1.00 | 3.00 | 8.00 | 1.000 | 345, 361, 352 |

从最终 greedy evaluation 看，所有 SARSA 调参设置都能达到 `1.00` success rate，说明该 easy/default 设置整体较容易。差异主要体现在收敛速度：较大的 `alpha=0.5` 收敛最快；较高的 `gamma=0.99` 略快于低折扣因子；epsilon decay 越快，越早稳定，其中 `decay=0.95` 的收敛 episode 明显早于 `0.99` 和 `0.995`。

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
- Q-learning / SARSA 更新公式、终止状态 bootstrap 关闭、epsilon 衰减
- 一轮小规模训练是否能生成标准输出
