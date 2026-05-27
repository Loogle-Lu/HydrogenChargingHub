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
from tqdm.auto import tqdm
from env import HydrogenEnv
from DDPG import DDPG, ReplayBuffer
from config import Config


# ======================== 超参数（与 compare.py 中 DDPG 公平基线对齐）====================
NUM_RUNS = 5  # 独立种子重复；柱状图误差棒 + 曲线±std 阴影
NUM_EPISODES = 1000
WARMUP_STEPS = 250
BATCH_SIZE = 256
LR = 3e-4
MA_WINDOW = 20
BUFFER_CAPACITY = 200_000

# ======================== 技术定义 ========================
TECH_DEFS = [
    ("VSD", "enable_vsd"),
    ("DI",  "enable_dynamic_cooling"),
    ("BP",  "enable_bypass"),
    ("AP",  "enable_adaptive_pressure"),
]

# 每层颜色 (Level 0..4): 灰 (Naive) → 蓝 → 绿 → 橙（色系渐变）
LEVEL_PALETTES = {
    0: ["#9e9e9e"],
    1: ["#deebf7", "#9ecae1", "#3182bd", "#08519c"],
    2: ["#edf8e9", "#c7e9c0", "#74c476", "#41ab5d", "#238b45", "#005a32"],
    3: ["#d9f0a3", "#fec44f", "#fe9929", "#ec7014"],
    4: ["#ff7f0e"],
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


def _std_across_runs(vals):
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size <= 1:
        return 0.0
    return float(np.std(vals, ddof=1))


def train_ddpg_one_variant(name, flags, num_episodes, num_runs):
    """在指定压缩机配置下训练 DDPG, 返回 rewards, profits，shape (num_runs, num_episodes)."""
    all_rewards, all_profits = [], []
    progress = tqdm(total=num_runs * num_episodes, desc=f"{name} training",
                    unit="ep", leave=False)
    for run in range(num_runs):
        set_seed(42 + run)
        _set_compressor_config(flags)
        env = HydrogenEnv()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = DDPG(state_dim, action_dim, lr=LR)
        replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)

        run_rewards, run_profits = [], []
        total_steps = 0

        for ep in range(num_episodes):
            state = env.reset(episode_index=ep)
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
            progress.update(1)
            progress.set_postfix(run=f"{run + 1}/{num_runs}",
                                 episode=f"{ep + 1}/{num_episodes}",
                                 reward=f"{ep_reward:.2f}",
                                 profit=f"{ep_profit:.2f}")

        all_rewards.append(run_rewards)
        all_profits.append(run_profits)
    progress.close()

    return np.asarray(all_rewards, dtype=np.float64), np.asarray(all_profits, dtype=np.float64)


def _mean_std_ma_curves(R, window):
    n_runs, n_ep = R.shape
    if n_ep < window:
        ma_runs = np.asarray(R, dtype=np.float64)
        x = np.arange(n_ep)
    else:
        ma_runs = np.stack([moving_average(R[i], window) for i in range(n_runs)])
        x = np.arange(window - 1, n_ep)
    mean_ma = np.mean(ma_runs, axis=0)
    if n_runs <= 1:
        std_ma = np.zeros_like(mean_ma)
    else:
        std_ma = np.std(ma_runs, axis=0, ddof=1)
    return x, mean_ma, std_ma


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
        rw, pf = train_ddpg_one_variant(name, flags, NUM_EPISODES, NUM_RUNS)
        results[name] = (rw, pf, level)
        ln = min(20, rw.shape[1])
        r_pr, p_pr = np.mean(rw[:, -ln:], axis=1), np.mean(pf[:, -ln:], axis=1)
        print(f"  {name}: Last-{ln} Reward = {np.mean(r_pr):.2f} ± {_std_across_runs(r_pr):.2f}, "
              f"Profit = ${np.mean(p_pr):,.0f} ± ${_std_across_runs(p_pr):,.0f}")

    _restore_compressor_config(saved_config)

    # ======================== 绘图 ========================
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9
    fig, axs = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    fig.suptitle(
        f"Exp 2: Full Factorial Additive Ablation",
        fontsize=13, fontweight="bold")

    # --- 准备数据 ---
    all_names, all_levels = [], []
    avg_reward_final, std_reward_final = [], []
    avg_profit_final, std_profit_final = [], []
    bar_colors = []
    _ln = min(20, NUM_EPISODES)

    for name, level, *_ in variants:
        all_names.append(name)
        all_levels.append(level)
        rw = results[name][0]
        pf = results[name][1]
        r_pr = np.mean(rw[:, -_ln:], axis=1)
        p_pr = np.mean(pf[:, -_ln:], axis=1)
        avg_reward_final.append(float(np.mean(r_pr)))
        std_reward_final.append(_std_across_runs(r_pr))
        avg_profit_final.append(float(np.mean(p_pr)))
        std_profit_final.append(_std_across_runs(p_pr))
        palette = LEVEL_PALETTES[level]
        level_names = [v[0] for v in variants if v[1] == level]
        idx_in_level = level_names.index(name)
        bar_colors.append(palette[idx_in_level % len(palette)])

    x = np.arange(len(all_names))

    def _annotate_level_bands(ax, all_levels, y_frac=0.92):
        """在柱状图顶部留白区内标注 L0..L4，避免与上边框重叠。"""
        ylo, yhi = ax.get_ylim()
        y_txt = ylo + (yhi - ylo) * y_frac
        for level in range(5):
            idxs = [i for i, l in enumerate(all_levels) if l == level]
            if idxs:
                mid = (idxs[0] + idxs[-1]) / 2
                ax.text(mid, y_txt, f"L{level}", ha="center", va="top", fontsize=9,
                        fontweight="bold", color="#222222")

    # --- (a) Reward 柱状图 (分层) ---
    bars = axs[0, 0].bar(x, avg_reward_final, yerr=std_reward_final, capsize=3,
                         color=bar_colors, edgecolor="gray", linewidth=0.5, error_kw={"elinewidth": 1.0})
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(all_names, rotation=55, ha="right", fontsize=7)
    axs[0, 0].set_ylabel("Avg Reward (Last 20 Ep)")
    axs[0, 0].set_title("(a) Reward by Configuration")
    axs[0, 0].grid(True, axis="y", alpha=0.3, linestyle="--")
    _rw_lo = min(np.asarray(avg_reward_final) - np.asarray(std_reward_final))
    _rw_hi = max(np.asarray(avg_reward_final) + np.asarray(std_reward_final))
    _rw_rng = _rw_hi - _rw_lo if _rw_hi != _rw_lo else max(abs(_rw_hi), 1.0)
    axs[0, 0].set_ylim(_rw_lo - 0.08 * _rw_rng, _rw_hi + 0.28 * _rw_rng)
    _annotate_level_bands(axs[0, 0], all_levels, y_frac=0.91)
    _bar_ylo, _bar_yhi = axs[0, 0].get_ylim()
    _val_pad_r = (_bar_yhi - _bar_ylo) * 0.012
    for b, v, e in zip(bars, avg_reward_final, std_reward_final):
        axs[0, 0].text(b.get_x() + b.get_width() / 2, b.get_height() + e + _val_pad_r,
                       f"{v:.1f}\n±{e:.2f}", ha="center", va="bottom", fontsize=5, rotation=90)

    # --- (b) Reward 学习曲线 (代表性: Naive, 各层最佳, Full) ---
    representative = _pick_representatives(variants, results, metric_idx=0)
    curve_colors = ["#9e9e9e", "#3182bd", "#31a354", "#fe9929", "#ff7f0e"]
    for (rname, rlevel), color in zip(representative, curve_colors):
        rw = results[rname][0]
        x_ma, m_ma, s_ma = _mean_std_ma_curves(rw, MA_WINDOW)
        axs[0, 1].fill_between(x_ma, m_ma - s_ma, m_ma + s_ma, color=color, alpha=0.2, linewidth=0)
        axs[0, 1].plot(x_ma, m_ma, color=color, linewidth=2, label=f"L{rlevel}: {rname}")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Episode Reward")
    axs[0, 1].set_title("(b) Reward Curves")
    axs[0, 1].legend(loc="best", fontsize=7)
    axs[0, 1].grid(True, alpha=0.3, linestyle="--")

    # --- (c) Profit 柱状图 ---
    bars = axs[1, 0].bar(x, avg_profit_final, yerr=std_profit_final, capsize=3,
                         color=bar_colors, edgecolor="gray", linewidth=0.5, error_kw={"elinewidth": 1.0})
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(all_names, rotation=55, ha="right", fontsize=7)
    axs[1, 0].set_ylabel("Avg Profit (Last 20 Ep, $)")
    axs[1, 0].set_title("(c) Profit by Configuration")
    axs[1, 0].grid(True, axis="y", alpha=0.3, linestyle="--")
    _pf_max = max(np.asarray(avg_profit_final) + np.asarray(std_profit_final))
    axs[1, 0].set_ylim(0, _pf_max * 1.22)
    _annotate_level_bands(axs[1, 0], all_levels, y_frac=0.92)
    _py_lo, _py_hi = axs[1, 0].get_ylim()
    _val_pad_p = (_py_hi - _py_lo) * 0.015
    for b, v, e in zip(bars, avg_profit_final, std_profit_final):
        axs[1, 0].text(b.get_x() + b.get_width() / 2, b.get_height() + e + _val_pad_p,
                       f"{v:.0f}\n±{e:.0f}", ha="center", va="bottom", fontsize=5, rotation=90)

    # --- (d) Profit 学习曲线 ---
    representative_p = _pick_representatives(variants, results, metric_idx=1)
    for (rname, rlevel), color in zip(representative_p, curve_colors):
        pf = results[rname][1]
        x_ma, m_ma, s_ma = _mean_std_ma_curves(pf, MA_WINDOW)
        axs[1, 1].fill_between(x_ma, m_ma - s_ma, m_ma + s_ma, color=color, alpha=0.2, linewidth=0)
        axs[1, 1].plot(x_ma, m_ma, color=color, linewidth=2, label=f"L{rlevel}: {rname}")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Episode Profit ($)")
    axs[1, 1].set_title("(d) Profit Curves")
    axs[1, 1].legend(loc="best", fontsize=7)
    axs[1, 1].grid(True, alpha=0.3, linestyle="--")

    plt.savefig("CompressorComparison_exp1_ablation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigure saved: CompressorComparison_exp1_ablation.png")

    # ======================== 打印汇总表 ========================
    print("\n" + "=" * 70)
    print(f"{'Level':<6} {'Variant':<16} {'Reward mean±std':>22} {'Profit mean±std':>26}")
    print("-" * 70)
    _ln = min(20, NUM_EPISODES)
    for name, level, *_ in variants:
        rw = results[name][0]
        pf = results[name][1]
        r_pr = np.mean(rw[:, -_ln:], axis=1)
        p_pr = np.mean(pf[:, -_ln:], axis=1)
        print(f"  {level:<4} {name:<16} {np.mean(r_pr):>8.2f}±{_std_across_runs(r_pr):<6.2f} "
              f"{np.mean(p_pr):>12.0f}±{_std_across_runs(p_pr):<8.0f}")
    print("=" * 70)


def _pick_representatives(variants, results, metric_idx=0):
    """为每层选出表现最佳的变体, 用于学习曲线展示 (共 5 条线: L0..L4).
    按各 run 末 20 轮均值的再平均最大化。"""
    reps = []
    ln = min(20, NUM_EPISODES)
    for level in range(5):
        level_vars = [(v[0], v[1]) for v in variants if v[1] == level]
        if not level_vars:
            continue
        best_name, best_level = max(
            level_vars,
            key=lambda v: np.mean(
                np.mean(results[v[0]][metric_idx][:, -ln:], axis=1)),
        )
        reps.append((best_name, best_level))
    return reps


if __name__ == "__main__":
    main()
