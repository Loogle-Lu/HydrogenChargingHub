"""
实验1: 压缩机技术消融实验

逐个关闭 VSD / 冷却 / 旁路 / 自适应压力，对比 Full 与 Naive 的收益差异。
突出级联压缩机各项技术的边际贡献。

输出:
- 2×2 图: Reward 柱状图 | Reward 曲线 | Profit 柱状图 | Profit 曲线

使用方法:
    cd "Compressor Comparison"
    python exp1_compressor_ablation.py
"""

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from env import HydrogenEnv
from SAC import SAC, ReplayBuffer
from config import Config


# ======================== 配置 ========================
NUM_RUNS = 3             # 每种配置重复运行次数，取平均
NUM_EPISODES = 80        # 每种配置训练 Episode 数
WARMUP_STEPS = 400
BATCH_SIZE = 256
LR = 3e-4
MA_WINDOW = 15

# 消融配置: (名称, enable_vsd, enable_dynamic_cooling, enable_bypass, enable_adaptive_pressure)
ABLATION_VARIANTS = [
    ("Full", True, True, True, True),
    ("w/o VSD", False, True, True, True),
    ("w/o Cooling", True, False, True, True),
    ("w/o Bypass", True, True, False, True),
    ("w/o Adapt.Pressure", True, True, True, False),
    ("Naive", False, False, False, False),
]

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b"
]


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_compressor_config():
    return {
        "enable_vsd": Config.enable_vsd,
        "enable_dynamic_cooling": Config.enable_dynamic_cooling,
        "enable_bypass": Config.enable_bypass,
        "enable_adaptive_pressure": Config.enable_adaptive_pressure,
    }


def _restore_compressor_config(saved):
    Config.enable_vsd = saved["enable_vsd"]
    Config.enable_dynamic_cooling = saved["enable_dynamic_cooling"]
    Config.enable_bypass = saved["enable_bypass"]
    Config.enable_adaptive_pressure = saved["enable_adaptive_pressure"]


def _set_compressor_config(vsd, cooling, bypass, adapt_p):
    Config.enable_vsd = vsd
    Config.enable_dynamic_cooling = cooling
    Config.enable_bypass = bypass
    Config.enable_adaptive_pressure = adapt_p


def train_sac_one_variant(name, vsd, cooling, bypass, adapt_p, num_episodes, num_runs):
    """在指定压缩机配置下训练 SAC，返回 (avg_rewards, avg_profits)"""
    all_rewards = []
    all_profits = []
    for run in range(num_runs):
        set_seed(42 + run)
        _set_compressor_config(vsd, cooling, bypass, adapt_p)
        env = HydrogenEnv(enable_i2s_constraint=True)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = SAC(state_dim, action_dim, lr=LR)
        replay_buffer = ReplayBuffer(capacity=100000)

        run_rewards = []
        run_profits = []
        total_steps = 0

        for ep in range(num_episodes):
            state = env.reset()
            ep_reward = 0.0
            ep_profit = 0.0
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


def moving_average(data, window):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def main():
    saved_config = _save_compressor_config()

    print("=" * 60)
    print("  实验1: 压缩机技术消融 (Compressor Technology Ablation)")
    print("=" * 60)
    print(f"  Variants: {[v[0] for v in ABLATION_VARIANTS]}")
    print(f"  Runs: {NUM_RUNS}, Episodes: {NUM_EPISODES}")
    print("=" * 60)

    results = {}  # name -> (rewards, profits)
    for i, (name, vsd, cooling, bypass, adapt_p) in enumerate(ABLATION_VARIANTS):
        print(f"\n[{i+1}/{len(ABLATION_VARIANTS)}] Training {name}...")
        avg_rewards, avg_profits = train_sac_one_variant(
            name, vsd, cooling, bypass, adapt_p, NUM_EPISODES, NUM_RUNS
        )
        results[name] = (avg_rewards, avg_profits)
        r20 = np.mean(avg_rewards[-20:]) if len(avg_rewards) >= 20 else np.mean(avg_rewards)
        p20 = np.mean(avg_profits[-20:]) if len(avg_profits) >= 20 else np.mean(avg_profits)
        print(f"  {name}: Final MA Reward = {r20:.2f}, MA Profit = ${p20:.0f}")

    _restore_compressor_config(saved_config)

    # ========== 绘图: 2×2 ==========
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle("Exp1: Compressor Technology Ablation (Reward + Profit)",
                 fontsize=12, fontweight="bold")

    names = [v[0] for v in ABLATION_VARIANTS]
    avg_reward_last20 = [np.mean(results[n][0][-20:]) if len(results[n][0]) >= 20
                         else np.mean(results[n][0]) for n in names]
    avg_profit_last20 = [np.mean(results[n][1][-20:]) if len(results[n][1]) >= 20
                         else np.mean(results[n][1]) for n in names]

    # (a) Reward 柱状图
    x = np.arange(len(names))
    bars = axs[0, 0].bar(x, avg_reward_last20, color=COLORS[:len(names)],
                         edgecolor="gray", linewidth=0.5)
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(names, rotation=25, ha="right")
    axs[0, 0].set_ylabel("Avg Reward (Last 20 Ep)")
    axs[0, 0].set_title("(a) Reward by Configuration")
    axs[0, 0].grid(True, axis="y", alpha=0.3, linestyle="--")
    for b, v in zip(bars, avg_reward_last20):
        axs[0, 0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{v:.1f}",
                       ha="center", va="bottom", fontsize=8)

    # (b) Reward 曲线
    for i, name in enumerate(names):
        rewards = results[name][0]
        axs[0, 1].plot(rewards, alpha=0.2, color=COLORS[i], linewidth=0.6)
        ma = moving_average(rewards, MA_WINDOW)
        axs[0, 1].plot(range(MA_WINDOW - 1, len(rewards)), ma,
                       color=COLORS[i], linewidth=2, label=name)
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Episode Reward")
    axs[0, 1].set_title("(b) Reward Curves (MA{})".format(MA_WINDOW))
    axs[0, 1].legend(loc="best", fontsize=7)
    axs[0, 1].grid(True, alpha=0.3, linestyle="--")

    # (c) Profit 柱状图
    bars = axs[1, 0].bar(x, avg_profit_last20, color=COLORS[:len(names)],
                         edgecolor="gray", linewidth=0.5)
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(names, rotation=25, ha="right")
    axs[1, 0].set_ylabel("Avg Profit (Last 20 Ep, $)")
    axs[1, 0].set_title("(c) Profit by Configuration")
    axs[1, 0].grid(True, axis="y", alpha=0.3, linestyle="--")
    for b, v in zip(bars, avg_profit_last20):
        axs[1, 0].text(b.get_x() + b.get_width() / 2, b.get_height() + 20, f"{v:.0f}",
                       ha="center", va="bottom", fontsize=8)

    # (d) Profit 曲线
    for i, name in enumerate(names):
        profits = results[name][1]
        axs[1, 1].plot(profits, alpha=0.2, color=COLORS[i], linewidth=0.6)
        ma = moving_average(profits, MA_WINDOW)
        axs[1, 1].plot(range(MA_WINDOW - 1, len(profits)), ma,
                       color=COLORS[i], linewidth=2, label=name)
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Episode Profit ($)")
    axs[1, 1].set_title("(d) Profit Curves (MA{})".format(MA_WINDOW))
    axs[1, 1].legend(loc="best", fontsize=7)
    axs[1, 1].grid(True, alpha=0.3, linestyle="--")

    plt.savefig("CompressorComparison_exp1_ablation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigure saved: CompressorComparison_exp1_ablation.png")


if __name__ == "__main__":
    main()
