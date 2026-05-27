"""
实验1: 压缩机技术消融实验 — 全因子递增消融 (Full Factorial Additive Ablation)

从 Naive (无技术) 出发, 按技术数量递增枚举所有 C(4,k) 组合 (k=0..4),
共 2^4 = 16 组, 展示各技术及其交互效应对加氢站收益的贡献.

4 项技术:
  VSD  — Variable Speed Drive (变速驱动)
  DI   — Dynamic Intercooling (动态级间冷却)
  BP   — Bypass (连续旁路控制)
  AP   — Adaptive Pressure (自适应压力控制)

分层:
  Level 0 (1 组): Naive
  Level 1 (4 组): +V, +DI, +BP, +AP
  Level 2 (6 组): +V+DI, +V+BP, +V+AP, +DI+BP, +DI+AP, +BP+AP
  Level 3 (4 组): +V+DI+BP, +V+DI+AP, +V+BP+AP, +DI+BP+AP
  Level 4 (1 组): Full

输出:
- 2×2 图: 按层分组柱状图 (Reward | Profit) + 代表性学习曲线

使用方法:
    python exp2_technologies_ablation.py
"""

import itertools
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from env import HydrogenEnv
from SAC import SAC, ReplayBuffer
from config import Config


# ======================== 超参数 ========================
NUM_RUNS = 1
NUM_EPISODES = 200
WARMUP_STEPS = 500
BATCH_SIZE = 256
LR = 3e-4
MA_WINDOW = 20

# ======================== 技术定义 ========================
TECH_DEFS = [
    ("VSD", "enable_vsd"),
    ("DI",  "enable_dynamic_cooling"),
    ("BP",  "enable_bypass"),
    ("AP",  "enable_adaptive_pressure"),
]

# 每层颜色 (Level 0..4)
LEVEL_PALETTES = {
    0: ["#8c8c8c"],                                                  # Naive: 灰色
    1: ["#aec7e8", "#7fb3e0", "#4a97d1", "#1f77b4"],                # 蓝色系
    2: ["#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", "#238b45", "#005a32"],  # 绿色系
    3: ["#fdbe85", "#fd8d3c", "#e6550d", "#a63603"],                # 橙色系
    4: ["#d62728"],                                                  # Full: 红色
}


def generate_variants():
    """生成全因子 2^4 = 16 组合, 按技术数量 (level) 分层."""
    variants = []
    for k in range(len(TECH_DEFS) + 1):
        for combo in itertools.combinations(range(len(TECH_DEFS)), k):
            flags = [False] * 4
            for idx in combo:
                flags[idx] = True
            if k == 0:
                name = "Naive"
            elif k == len(TECH_DEFS):
                name = "Full"
            else:
                name = "+".join(TECH_DEFS[i][0] for i in combo)
            variants.append((name, k, *flags))
    return variants


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_compressor_config():
    return {attr: getattr(Config, attr) for _, attr in TECH_DEFS}


def _restore_compressor_config(saved):
    for attr, val in saved.items():
        setattr(Config, attr, val)


def _set_compressor_config(flags):
    for (_, attr), flag in zip(TECH_DEFS, flags):
        setattr(Config, attr, flag)


def train_sac_one_variant(name, flags, num_episodes, num_runs):
    """在指定压缩机配置下训练 SAC, 返回 (avg_rewards, avg_profits)."""
    all_rewards, all_profits = [], []
    for run in range(num_runs):
        set_seed(42 + run)
        _set_compressor_config(flags)
        env = HydrogenEnv()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = SAC(state_dim, action_dim, lr=LR)
        replay_buffer = ReplayBuffer(capacity=100000)

        run_rewards, run_profits = [], []
        total_steps = 0

        for ep in range(num_episodes):
            state = env.reset()
            ep_reward, ep_profit, done = 0.0, 0.0, False

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
    variants = generate_variants()

    print("=" * 70)
    print("  实验1: 全因子递增消融 (Full Factorial Additive Ablation)")
    print("=" * 70)
    for level in range(5):
        names = [v[0] for v in variants if v[1] == level]
        print(f"  Level {level} ({len(names)}): {names}")
    print(f"  Total: {len(variants)} variants | Runs: {NUM_RUNS} | Episodes: {NUM_EPISODES}")
    print("=" * 70)

    results = {}
    for i, (name, level, *flags) in enumerate(variants):
        print(f"\n[{i+1}/{len(variants)}] Level {level} — Training '{name}'...")
        avg_rewards, avg_profits = train_sac_one_variant(name, flags, NUM_EPISODES, NUM_RUNS)
        results[name] = (avg_rewards, avg_profits, level)
        r20 = np.mean(avg_rewards[-20:]) if len(avg_rewards) >= 20 else np.mean(avg_rewards)
        p20 = np.mean(avg_profits[-20:]) if len(avg_profits) >= 20 else np.mean(avg_profits)
        print(f"  {name}: MA Reward = {r20:.2f}, MA Profit = ${p20:.0f}")

    _restore_compressor_config(saved_config)

    # ======================== 绘图 ========================
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9
    fig, axs = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    fig.suptitle("Exp 1: Full Factorial Additive Ablation — Compressor Technologies",
                 fontsize=13, fontweight="bold")

    # --- 准备数据 ---
    all_names, all_levels = [], []
    avg_reward_final, avg_profit_final = [], []
    bar_colors = []

    for name, level, *_ in variants:
        all_names.append(name)
        all_levels.append(level)
        rw = results[name][0]
        pf = results[name][1]
        avg_reward_final.append(np.mean(rw[-20:]) if len(rw) >= 20 else np.mean(rw))
        avg_profit_final.append(np.mean(pf[-20:]) if len(pf) >= 20 else np.mean(pf))
        palette = LEVEL_PALETTES[level]
        level_names = [v[0] for v in variants if v[1] == level]
        idx_in_level = level_names.index(name)
        bar_colors.append(palette[idx_in_level % len(palette)])

    x = np.arange(len(all_names))

    # --- (a) Reward 柱状图 (分层) ---
    bars = axs[0, 0].bar(x, avg_reward_final, color=bar_colors, edgecolor="gray", linewidth=0.5)
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(all_names, rotation=55, ha="right", fontsize=7)
    axs[0, 0].set_ylabel("Avg Reward (Last 20 Ep)")
    axs[0, 0].set_title("(a) Reward by Configuration")
    axs[0, 0].grid(True, axis="y", alpha=0.3, linestyle="--")
    for level in range(5):
        idxs = [i for i, l in enumerate(all_levels) if l == level]
        if idxs:
            mid = (idxs[0] + idxs[-1]) / 2
            axs[0, 0].text(mid, axs[0, 0].get_ylim()[1] * 0.98,
                           f"L{level}", ha="center", fontsize=8, fontstyle="italic", color="gray")
    for b, v in zip(bars, avg_reward_final):
        axs[0, 0].text(b.get_x() + b.get_width() / 2, b.get_height(),
                       f"{v:.1f}", ha="center", va="bottom", fontsize=6, rotation=90)

    # --- (b) Reward 学习曲线 (代表性: Naive, 各层最佳, Full) ---
    representative = _pick_representatives(variants, results, metric_idx=0)
    curve_colors = ["#8c8c8c", "#1f77b4", "#238b45", "#e6550d", "#d62728"]
    for (rname, rlevel), color in zip(representative, curve_colors):
        rw = results[rname][0]
        axs[0, 1].plot(rw, alpha=0.15, color=color, linewidth=0.5)
        ma = moving_average(rw, MA_WINDOW)
        axs[0, 1].plot(range(MA_WINDOW - 1, len(rw)), ma,
                       color=color, linewidth=2, label=f"L{rlevel}: {rname}")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Episode Reward")
    axs[0, 1].set_title("(b) Reward Curves (Best per Level)")
    axs[0, 1].legend(loc="best", fontsize=7)
    axs[0, 1].grid(True, alpha=0.3, linestyle="--")

    # --- (c) Profit 柱状图 ---
    bars = axs[1, 0].bar(x, avg_profit_final, color=bar_colors, edgecolor="gray", linewidth=0.5)
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(all_names, rotation=55, ha="right", fontsize=7)
    axs[1, 0].set_ylabel("Avg Profit (Last 20 Ep, $)")
    axs[1, 0].set_title("(c) Profit by Configuration")
    axs[1, 0].grid(True, axis="y", alpha=0.3, linestyle="--")
    for level in range(5):
        idxs = [i for i, l in enumerate(all_levels) if l == level]
        if idxs:
            mid = (idxs[0] + idxs[-1]) / 2
            axs[1, 0].text(mid, axs[1, 0].get_ylim()[1] * 0.98,
                           f"L{level}", ha="center", fontsize=8, fontstyle="italic", color="gray")
    for b, v in zip(bars, avg_profit_final):
        axs[1, 0].text(b.get_x() + b.get_width() / 2, b.get_height(),
                       f"{v:.0f}", ha="center", va="bottom", fontsize=6, rotation=90)

    # --- (d) Profit 学习曲线 ---
    representative_p = _pick_representatives(variants, results, metric_idx=1)
    for (rname, rlevel), color in zip(representative_p, curve_colors):
        pf = results[rname][1]
        axs[1, 1].plot(pf, alpha=0.15, color=color, linewidth=0.5)
        ma = moving_average(pf, MA_WINDOW)
        axs[1, 1].plot(range(MA_WINDOW - 1, len(pf)), ma,
                       color=color, linewidth=2, label=f"L{rlevel}: {rname}")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Episode Profit ($)")
    axs[1, 1].set_title("(d) Profit Curves (Best per Level)")
    axs[1, 1].legend(loc="best", fontsize=7)
    axs[1, 1].grid(True, alpha=0.3, linestyle="--")

    plt.savefig("CompressorComparison_exp1_ablation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigure saved: CompressorComparison_exp1_ablation.png")

    # ======================== 打印汇总表 ========================
    print("\n" + "=" * 70)
    print(f"{'Level':<6} {'Variant':<16} {'Avg Reward':>12} {'Avg Profit ($)':>16}")
    print("-" * 70)
    for name, level, *_ in variants:
        rw = results[name][0]
        pf = results[name][1]
        r_val = np.mean(rw[-20:]) if len(rw) >= 20 else np.mean(rw)
        p_val = np.mean(pf[-20:]) if len(pf) >= 20 else np.mean(pf)
        print(f"  {level:<4} {name:<16} {r_val:>12.2f} {p_val:>16.0f}")
    print("=" * 70)


def _pick_representatives(variants, results, metric_idx=0):
    """为每层选出表现最佳的变体, 用于学习曲线展示 (共 5 条线: L0..L4)."""
    reps = []
    for level in range(5):
        level_vars = [(v[0], v[1]) for v in variants if v[1] == level]
        if not level_vars:
            continue
        best_name, best_level = max(
            level_vars,
            key=lambda v: np.mean(results[v[0]][metric_idx][-20:])
                if len(results[v[0]][metric_idx]) >= 20
                else np.mean(results[v[0]][metric_idx])
        )
        reps.append((best_name, best_level))
    return reps


if __name__ == "__main__":
    main()
