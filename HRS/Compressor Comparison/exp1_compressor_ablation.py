"""
实验1: 压缩机技术消融实验

逐个关闭 VSD / 冷却 / 旁路 / 自适应压力，对比 Full 与 Naive 的收益差异。
突出级联压缩机各项技术的边际贡献。

输出:
- 图1: 柱状图 (各配置平均 Profit) + 曲线图 (训练过程中 Profit)
- 合并为 1 张图 (左柱状图, 右曲线)

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
NUM_RUNS = 1             # 每种配置重复运行次数，取平均
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
    """在指定压缩机配置下训练 SAC"""
    all_profits = []  # list of lists, each run
    for run in range(num_runs):
        set_seed(42 + run)
        _set_compressor_config(vsd, cooling, bypass, adapt_p)
        env = HydrogenEnv(enable_i2s_constraint=True)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
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

    avg_profits = np.mean(all_profits, axis=0)
    return avg_profits


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

    results = {}
    for i, (name, vsd, cooling, bypass, adapt_p) in enumerate(ABLATION_VARIANTS):
        print(f"\n[{i+1}/{len(ABLATION_VARIANTS)}] Training {name}...")
        avg_profits = train_sac_one_variant(
            name, vsd, cooling, bypass, adapt_p, NUM_EPISODES, NUM_RUNS
        )
        results[name] = avg_profits
        print(f"  {name}: Final MA Profit = ${np.mean(avg_profits[-20:]):.0f}")

    _restore_compressor_config(saved_config)

    # ========== 绘图: 1 张图 = 左柱状图 + 右曲线 ==========
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle("Compressor Technology Ablation: Impact on Station Profit",
                 fontsize=12, fontweight="bold")

    names = [v[0] for v in ABLATION_VARIANTS]
    avg_profit_last20 = [np.mean(results[n][-20:]) for n in names]

    # 左: 柱状图
    x = np.arange(len(names))
    bars = ax1.bar(x, avg_profit_last20, color=COLORS[:len(names)], edgecolor="gray", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha="right")
    ax1.set_ylabel("Avg Profit (Last 20 Ep, $)")
    ax1.set_title("(a) Average Profit by Configuration")
    ax1.grid(True, axis="y", alpha=0.3, linestyle="--")
    for b, v in zip(bars, avg_profit_last20):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 20, f"{v:.0f}",
                 ha="center", va="bottom", fontsize=8, rotation=0)

    # 右: 曲线图
    for i, name in enumerate(names):
        profits = results[name]
        ax2.plot(profits, alpha=0.2, color=COLORS[i], linewidth=0.6)
        ma = moving_average(profits, MA_WINDOW)
        ax2.plot(range(MA_WINDOW - 1, len(profits)), ma,
                 color=COLORS[i], linewidth=2, label=name)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode Profit ($)")
    ax2.set_title("(b) Training Curves (MA{})".format(MA_WINDOW))
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3, linestyle="--")

    plt.savefig("CompressorComparison_exp1_ablation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigure saved: CompressorComparison_exp1_ablation.png")


if __name__ == "__main__":
    main()
