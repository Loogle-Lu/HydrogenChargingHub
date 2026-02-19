"""
实验2: 动作空间消融实验

对比「完整 7 维动作(含压缩机智能控制)」与「固定压缩机动作(仅 3 维有效)」的收益差异。
突出压缩机智能控制的边际贡献。
(环境已移除BESS，EV充电来自 绿色能源+电网+H2转电)

- Full 7D: [ele, fc, comp_load, cooling, bypass, c3_pressure, chiller]
- Fixed Compressor 3D: [ele, fc, chiller]，压缩机相关固定为 [0.7, 0.7, 0.5, 0.5]

输出:
- 2×2 图: Reward 柱状图 | Reward 曲线 | Profit 柱状图 | Profit 曲线
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
    将 7 维动作空间压缩为 3 维: [ele, fc, chiller]
    压缩机相关维度固定为常量 (comp_load, cooling, bypass, c3_pressure)
    """
    def __init__(self, env, fixed_compressor=None):
        super().__init__(env)
        if fixed_compressor is None:
            fixed_compressor = FIXED_COMPRESSOR_ACTIONS
        self.fixed = np.array(fixed_compressor, dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

    def action(self, action):
        # action: [ele, fc, chiller]
        a = np.asarray(action, dtype=np.float32).flatten()
        if len(a) < 3:
            a = np.pad(a, (0, 3 - len(a)), constant_values=0.5)
        # 映射到 7 维: [ele, fc, comp_load, cooling, bypass, c3_pressure, chiller]
        full = np.array([
            a[0], a[1],           # ele, fc
            self.fixed[0],        # comp_load
            self.fixed[1],        # cooling
            self.fixed[2],        # bypass
            self.fixed[3],        # c3_pressure
            a[2]                  # chiller
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


def train_sac_full_7d(num_episodes, num_runs):
    """SAC 完整 7 维动作 (无BESS)，返回 (rewards, profits)"""
    all_rewards, all_profits = [], []
    for run in range(num_runs):
        set_seed(42 + run)
        env = HydrogenEnv(enable_i2s_constraint=True)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = SAC(state_dim, action_dim, lr=LR)
        replay_buffer = ReplayBuffer(capacity=100000)

        run_rewards, run_profits = [], []
        total_steps = 0

        for ep in range(num_episodes):
            state = env.reset()
            ep_reward, ep_profit = 0.0, 0.0
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
                ep_reward += reward
                ep_profit += info.get("profit", 0.0)
                total_steps += 1

            run_rewards.append(ep_reward)
            run_profits.append(ep_profit)

        all_rewards.append(run_rewards)
        all_profits.append(run_profits)

    return np.mean(all_rewards, axis=0), np.mean(all_profits, axis=0)


def train_sac_fixed_compressor_3d(num_episodes, num_runs):
    """SAC 固定压缩机 (3 维有效动作)，返回 (rewards, profits)"""
    all_rewards, all_profits = [], []
    for run in range(num_runs):
        set_seed(42 + run)
        env_raw = HydrogenEnv(enable_i2s_constraint=True)
        env = FixedCompressorActionWrapper(env_raw)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = SAC(state_dim, action_dim, lr=LR)
        replay_buffer = ReplayBuffer(capacity=100000)

        run_rewards, run_profits = [], []
        total_steps = 0

        for ep in range(num_episodes):
            state = env.reset()
            ep_reward, ep_profit = 0.0, 0.0
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
                ep_reward += reward
                ep_profit += info.get("profit", 0.0)
                total_steps += 1

            run_rewards.append(ep_reward)
            run_profits.append(ep_profit)

        all_rewards.append(run_rewards)
        all_profits.append(run_profits)

    return np.mean(all_rewards, axis=0), np.mean(all_profits, axis=0)


def main():
    print("=" * 60)
    print("  实验2: 动作空间消融 (Full 8D vs Fixed Compressor 4D)")
    print("=" * 60)
    print(f"  Runs: {NUM_RUNS}, Episodes: {NUM_EPISODES}")
    print("=" * 60)

    print("\n[1/2] Training SAC Full 7D (compressor intelligent control)...")
    rewards_full, profits_full = train_sac_full_7d(NUM_EPISODES, NUM_RUNS)
    print(f"  Full 7D: Final MA Reward = {np.mean(rewards_full[-20:]):.2f}, MA Profit = ${np.mean(profits_full[-20:]):.0f}")

    print("\n[2/2] Training SAC Fixed Compressor 3D...")
    rewards_fixed, profits_fixed = train_sac_fixed_compressor_3d(NUM_EPISODES, NUM_RUNS)
    print(f"  Fixed 3D: Final MA Reward = {np.mean(rewards_fixed[-20:]):.2f}, MA Profit = ${np.mean(profits_fixed[-20:]):.0f}")

    # ========== 绘图 2×2 ==========
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle("Exp2: Action Space Ablation (Reward + Profit)",
                 fontsize=12, fontweight="bold")

    names = ["Full 7D", "Fixed Compressor 3D"]
    colors = ["#1f77b4", "#ff7f0e"]
    r_full = np.mean(rewards_full[-20:]) if len(rewards_full) >= 20 else np.mean(rewards_full)
    r_fix = np.mean(rewards_fixed[-20:]) if len(rewards_fixed) >= 20 else np.mean(rewards_fixed)
    p_full = np.mean(profits_full[-20:]) if len(profits_full) >= 20 else np.mean(profits_full)
    p_fix = np.mean(profits_fixed[-20:]) if len(profits_fixed) >= 20 else np.mean(profits_fixed)

    # (a) Reward 柱状图
    x = np.arange(2)
    bars = axs[0, 0].bar(x, [r_full, r_fix], color=colors, edgecolor="gray", linewidth=0.5)
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(names)
    axs[0, 0].set_ylabel("Avg Reward (Last 20 Ep)")
    axs[0, 0].set_title("(a) Reward")
    axs[0, 0].grid(True, axis="y", alpha=0.3, linestyle="--")
    for b, v in zip(bars, [r_full, r_fix]):
        axs[0, 0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{v:.1f}",
                       ha="center", va="bottom", fontsize=9)

    # (b) Reward 曲线
    axs[0, 1].plot(rewards_full, alpha=0.2, color=colors[0], linewidth=0.6)
    axs[0, 1].plot(range(MA_WINDOW - 1, len(rewards_full)), moving_average(rewards_full, MA_WINDOW),
                   color=colors[0], linewidth=2, label="Full 7D")
    axs[0, 1].plot(rewards_fixed, alpha=0.2, color=colors[1], linewidth=0.6)
    axs[0, 1].plot(range(MA_WINDOW - 1, len(rewards_fixed)), moving_average(rewards_fixed, MA_WINDOW),
                   color=colors[1], linewidth=2, label="Fixed Compressor 3D")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Episode Reward")
    axs[0, 1].set_title("(b) Reward Curves (MA{})".format(MA_WINDOW))
    axs[0, 1].legend(loc="best")
    axs[0, 1].grid(True, alpha=0.3, linestyle="--")

    # (c) Profit 柱状图
    bars = axs[1, 0].bar(x, [p_full, p_fix], color=colors, edgecolor="gray", linewidth=0.5)
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(names)
    axs[1, 0].set_ylabel("Avg Profit (Last 20 Ep, $)")
    axs[1, 0].set_title("(c) Profit")
    axs[1, 0].grid(True, axis="y", alpha=0.3, linestyle="--")
    for b, v in zip(bars, [p_full, p_fix]):
        axs[1, 0].text(b.get_x() + b.get_width() / 2, b.get_height() + 20, f"{v:.0f}",
                       ha="center", va="bottom", fontsize=9)

    # (d) Profit 曲线
    axs[1, 1].plot(profits_full, alpha=0.2, color=colors[0], linewidth=0.6)
    axs[1, 1].plot(range(MA_WINDOW - 1, len(profits_full)), moving_average(profits_full, MA_WINDOW),
                   color=colors[0], linewidth=2, label="Full 7D")
    axs[1, 1].plot(profits_fixed, alpha=0.2, color=colors[1], linewidth=0.6)
    axs[1, 1].plot(range(MA_WINDOW - 1, len(profits_fixed)), moving_average(profits_fixed, MA_WINDOW),
                   color=colors[1], linewidth=2, label="Fixed Compressor 3D")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Episode Profit ($)")
    axs[1, 1].set_title("(d) Profit Curves (MA{})".format(MA_WINDOW))
    axs[1, 1].legend(loc="best")
    axs[1, 1].grid(True, alpha=0.3, linestyle="--")

    plt.savefig("CompressorComparison_exp2_action_ablation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigure saved: CompressorComparison_exp2_action_ablation.png")


if __name__ == "__main__":
    main()
