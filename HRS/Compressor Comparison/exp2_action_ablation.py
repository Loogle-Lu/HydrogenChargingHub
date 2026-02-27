"""
实验2: 动作空间消融实验

对比「完整 8 维动作(含压缩机智能控制)」与「固定压缩机动作(仅 4 维有效)」的收益差异。
突出压缩机智能控制的边际贡献。

- Full 8D: [ele, fc, comp_load, cooling, battery, bypass, c3_pressure, chiller]
- Fixed Compressor 4D: [ele, fc, battery, chiller]，压缩机相关固定为 [0.7, 0.7, 0.5, 0.5]

输出:
- 1 张图: 左柱状图 (Full vs Fixed 平均 Profit) + 右曲线图 (训练过程)

使用方法:
    cd "Compressor Comparison"
    python exp2_action_ablation.py
"""

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import gym
from gym import spaces
from env import HydrogenEnv
from SAC import SAC, ReplayBuffer


# ======================== 配置 ========================
NUM_RUNS = 1
NUM_EPISODES = 80
WARMUP_STEPS = 400
BATCH_SIZE = 256
LR = 3e-4
MA_WINDOW = 15

# 固定压缩机动作 (comp_load, cooling, bypass, c3_pressure)
FIXED_COMPRESSOR_ACTIONS = [0.7, 0.7, 0.5, 0.5]


class FixedCompressorActionWrapper(gym.ActionWrapper):
    """
    将 8 维动作空间压缩为 4 维: [ele, fc, battery, chiller]
    压缩机相关维度固定为常量
    """
    def __init__(self, env, fixed_compressor=None):
        super().__init__(env)
        if fixed_compressor is None:
            fixed_compressor = FIXED_COMPRESSOR_ACTIONS
        self.fixed = np.array(fixed_compressor, dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

    def action(self, action):
        # action: [ele, fc, battery, chiller]
        a = np.asarray(action, dtype=np.float32).flatten()
        if len(a) < 4:
            a = np.pad(a, (0, 4 - len(a)), constant_values=0.5)
        # 映射到 8 维: [ele, fc, comp_load, cooling, battery, bypass, c3_pressure, chiller]
        full = np.array([
            a[0], a[1],           # ele, fc
            self.fixed[0],        # comp_load
            self.fixed[1],        # cooling
            a[2],                 # battery
            self.fixed[2],        # bypass
            self.fixed[3],        # c3_pressure
            a[3]                  # chiller
        ], dtype=np.float32)
        return full


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def moving_average(data, window):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def train_sac_full_8d(num_episodes, num_runs):
    """SAC 完整 8 维动作"""
    all_profits = []
    for run in range(num_runs):
        set_seed(42 + run)
        env = HydrogenEnv(enable_i2s_constraint=True)
        state_dim = env.observation_space.shape[0]
        action_dim = 8
        agent = SAC(state_dim, action_dim, lr=LR)
        replay_buffer = ReplayBuffer(capacity=100000)

        run_profits = []
        total_steps = 0

        for ep in range(num_episodes):
            state = env.reset()
            ep_profit = 0
            done = False

            while not done:
                if total_steps < WARMUP_STEPS:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state, evaluate=False)

                next_state, reward, done, info = env.step(action)
                replay_buffer.push(state, action, reward, next_state, float(done))

                if total_steps >= WARMUP_STEPS and len(replay_buffer) >= BATCH_SIZE:
                    agent.update(replay_buffer, BATCH_SIZE)

                state = next_state
                ep_profit += info.get("profit", 0.0)
                total_steps += 1

            run_profits.append(ep_profit)

        all_profits.append(run_profits)

    return np.mean(all_profits, axis=0)


def train_sac_fixed_compressor_4d(num_episodes, num_runs):
    """SAC 固定压缩机 (4 维有效动作)"""
    all_profits = []
    for run in range(num_runs):
        set_seed(42 + run)
        env_raw = HydrogenEnv(enable_i2s_constraint=True)
        env = FixedCompressorActionWrapper(env_raw)

        state_dim = env.observation_space.shape[0]
        action_dim = 4
        agent = SAC(state_dim, action_dim, lr=LR)
        replay_buffer = ReplayBuffer(capacity=100000)

        run_profits = []
        total_steps = 0

        for ep in range(num_episodes):
            state = env.reset()
            ep_profit = 0
            done = False

            while not done:
                if total_steps < WARMUP_STEPS:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state, evaluate=False)

                next_state, reward, done, info = env.step(action)
                replay_buffer.push(state, action, reward, next_state, float(done))

                if total_steps >= WARMUP_STEPS and len(replay_buffer) >= BATCH_SIZE:
                    agent.update(replay_buffer, BATCH_SIZE)

                state = next_state
                ep_profit += info.get("profit", 0.0)
                total_steps += 1

            run_profits.append(ep_profit)

        all_profits.append(run_profits)

    return np.mean(all_profits, axis=0)


def main():
    print("=" * 60)
    print("  实验2: 动作空间消融 (Full 8D vs Fixed Compressor 4D)")
    print("=" * 60)
    print(f"  Runs: {NUM_RUNS}, Episodes: {NUM_EPISODES}")
    print("=" * 60)

    print("\n[1/2] Training SAC Full 8D (compressor intelligent control)...")
    profits_full = train_sac_full_8d(NUM_EPISODES, NUM_RUNS)
    print(f"  Full 8D: Final MA Profit = ${np.mean(profits_full[-20:]):.0f}")

    print("\n[2/2] Training SAC Fixed Compressor 4D...")
    profits_fixed = train_sac_fixed_compressor_4d(NUM_EPISODES, NUM_RUNS)
    print(f"  Fixed 4D: Final MA Profit = ${np.mean(profits_fixed[-20:]):.0f}")

    # ========== 绘图 ==========
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Action Space Ablation: Full 8D vs Fixed Compressor 4D",
                 fontsize=12, fontweight="bold")

    # 左: 柱状图
    names = ["Full 8D", "Fixed Compressor 4D"]
    avgs = [np.mean(profits_full[-20:]), np.mean(profits_fixed[-20:])]
    colors = ["#1f77b4", "#ff7f0e"]
    x = np.arange(2)
    bars = ax1.bar(x, avgs, color=colors, edgecolor="gray", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Avg Profit (Last 20 Ep, $)")
    ax1.set_title("(a) Average Profit")
    ax1.grid(True, axis="y", alpha=0.3, linestyle="--")
    for b, v in zip(bars, avgs):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 20, f"{v:.0f}",
                 ha="center", va="bottom", fontsize=9)

    # 右: 曲线图
    ax2.plot(profits_full, alpha=0.2, color=colors[0], linewidth=0.6)
    ma_full = moving_average(profits_full, MA_WINDOW)
    ax2.plot(range(MA_WINDOW - 1, len(profits_full)), ma_full,
             color=colors[0], linewidth=2, label="Full 8D")

    ax2.plot(profits_fixed, alpha=0.2, color=colors[1], linewidth=0.6)
    ma_fixed = moving_average(profits_fixed, MA_WINDOW)
    ax2.plot(range(MA_WINDOW - 1, len(profits_fixed)), ma_fixed,
             color=colors[1], linewidth=2, label="Fixed Compressor 4D")

    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode Profit ($)")
    ax2.set_title("(b) Training Curves (MA{})".format(MA_WINDOW))
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3, linestyle="--")

    plt.show()
    print("\nFigure saved: CompressorComparison_exp2_action_ablation.png")


if __name__ == "__main__":
    main()
