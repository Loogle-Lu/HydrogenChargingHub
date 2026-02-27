"""
实验2: 动作空间消融实验

对比「完整 6 维动作(含压缩机智能控制)」与「Naive 固定压缩机动作(仅 2 维有效)」的收益差异。
突出压缩机智能控制的边际贡献。
(环境已移除 EV，State 11D，Action 6D)

- Full 6D: [ele, fc, c1_cool, c2_cool, c3_pressure_bias, bypass_bias]
- Naive No-Intelligence 2D: [ele, fc]，压缩机固定无冷却优化、无旁路
  [c1_cool=0.0, c2_cool=0.0, c3_pressure_bias=0.5, bypass_bias=0.0]
  代表「无智能压缩机控制」的自然 Baseline:
    - c1_cool=c2_cool=0.0: 无冷却优化 (轻度冷却, 最高温度)
    - bypass_bias=0.0: 从不旁路 (即使储罐压力充足也继续压缩)
    - c3_pressure_bias=0.5: C3 APC 默认

输出:
- 图: Reward/Profit 柱状图(含误差棒) | Reward/Profit 曲线 | Compressor/Chiller/Bypass 指标
"""

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import gym
from gym import spaces
from env import HydrogenEnv
from SAC import SAC, ReplayBuffer
from config import Config


# ======================== 配置 ========================
NUM_RUNS = 1  # 增加 run 数以计算误差棒
NUM_EPISODES = 500  # 6D 搜索空间，需要足够步骤收敛
WARMUP_STEPS = 400
BATCH_SIZE = 256
LR = 3e-4
MA_WINDOW = 20

# Naive 固定压缩机动作 (c1_cool, c2_cool, c3_pressure_bias, bypass_bias)
# [0.0, 0.0, 0.5, 0.0] = 无冷却优化/默认C3压力/无旁路
# 代表「无智能控制」的自然 Baseline
FIXED_COMPRESSOR_ACTIONS = [0.0, 0.0, 0.5, 0.0]


class FixedCompressorActionWrapper(gym.ActionWrapper):
    """
    将 6 维动作空间压缩为 2 维: [ele, fc]
    压缩机相关维度固定为常量 (c1_cool, c2_cool, c3_pressure_bias, bypass_bias)
    """
    def __init__(self, env, fixed_compressor=None):
        super().__init__(env)
        if fixed_compressor is None:
            fixed_compressor = FIXED_COMPRESSOR_ACTIONS
        self.fixed = np.array(fixed_compressor, dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    def action(self, action):
        # action: [ele, fc]
        a = np.asarray(action, dtype=np.float32).flatten()
        if len(a) < 2:
            a = np.pad(a, (0, 2 - len(a)), constant_values=0.5)
        # 映射到 6 维: [ele, fc, c1_cool, c2_cool, c3_pressure_bias, bypass_bias]
        full = np.array([
            a[0], a[1],           # ele, fc
            self.fixed[0],        # c1_load
            self.fixed[1],        # c2_load
            self.fixed[2],        # c3_pressure_bias
            self.fixed[3],        # bypass_bias
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


def train_sac_full_6d(num_episodes, num_runs):
    """
    SAC 完整 6 维动作 (含压缩机智能控制)
    返回: (all_rewards, all_profits, all_comp_energy, all_chiller_energy, all_bypass)
    均为 shape (num_runs, num_episodes)
    """
    dt = Config.dt
    all_rewards, all_profits = [], []
    all_comp_energy, all_chiller_energy, all_bypass = [], [], []
    for run in range(num_runs):
        set_seed(42 + run)
        env = HydrogenEnv(enable_i2s_constraint=True)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = SAC(state_dim, action_dim, lr=LR)
        replay_buffer = ReplayBuffer(capacity=100000)

        run_rewards, run_profits = [], []
        run_comp, run_chiller, run_bypass = [], [], []
        total_steps = 0

        for ep in range(num_episodes):
            state = env.reset()
            ep_reward, ep_profit = 0.0, 0.0
            ep_comp_kwh, ep_chiller_kwh = 0.0, 0.0
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
                ep_comp_kwh += info.get("comp_power", 0.0) * dt
                ep_chiller_kwh += info.get("chiller_power", 0.0) * dt
                total_steps += 1

            bp = info.get("bypass_activations", {"c1": 0, "c2": 0, "c3": 0})
            ep_bypass = bp.get("c1", 0) + bp.get("c2", 0) + bp.get("c3", 0)

            run_rewards.append(ep_reward)
            run_profits.append(ep_profit)
            run_comp.append(ep_comp_kwh)
            run_chiller.append(ep_chiller_kwh)
            run_bypass.append(ep_bypass)

        all_rewards.append(run_rewards)
        all_profits.append(run_profits)
        all_comp_energy.append(run_comp)
        all_chiller_energy.append(run_chiller)
        all_bypass.append(run_bypass)

    arr = lambda x: np.array(x)
    return arr(all_rewards), arr(all_profits), arr(all_comp_energy), arr(all_chiller_energy), arr(all_bypass)


def train_sac_fixed_compressor_2d(num_episodes, num_runs):
    """
    SAC 固定压缩机 (2 维有效动作: ele, fc)
    返回: (all_rewards, all_profits, all_comp_energy, all_chiller_energy, all_bypass)
    """
    dt = Config.dt
    all_rewards, all_profits = [], []
    all_comp_energy, all_chiller_energy, all_bypass = [], [], []
    for run in range(num_runs):
        set_seed(42 + run)
        env_raw = HydrogenEnv(enable_i2s_constraint=True)
        env = FixedCompressorActionWrapper(env_raw)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = SAC(state_dim, action_dim, lr=LR)
        replay_buffer = ReplayBuffer(capacity=100000)

        run_rewards, run_profits = [], []
        run_comp, run_chiller, run_bypass = [], [], []
        total_steps = 0

        for ep in range(num_episodes):
            state = env.reset()
            ep_reward, ep_profit = 0.0, 0.0
            ep_comp_kwh, ep_chiller_kwh = 0.0, 0.0
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
                ep_comp_kwh += info.get("comp_power", 0.0) * dt
                ep_chiller_kwh += info.get("chiller_power", 0.0) * dt
                total_steps += 1

            bp = info.get("bypass_activations", {"c1": 0, "c2": 0, "c3": 0})
            ep_bypass = bp.get("c1", 0) + bp.get("c2", 0) + bp.get("c3", 0)

            run_rewards.append(ep_reward)
            run_profits.append(ep_profit)
            run_comp.append(ep_comp_kwh)
            run_chiller.append(ep_chiller_kwh)
            run_bypass.append(ep_bypass)

        all_rewards.append(run_rewards)
        all_profits.append(run_profits)
        all_comp_energy.append(run_comp)
        all_chiller_energy.append(run_chiller)
        all_bypass.append(run_bypass)

    arr = lambda x: np.array(x)
    return arr(all_rewards), arr(all_profits), arr(all_comp_energy), arr(all_chiller_energy), arr(all_bypass)


def _last20_mean_std(arr):
    """arr: (num_runs, num_episodes), 返回每 run 末 20 ep 均值的 mean 和 std"""
    n_ep = arr.shape[1]
    last_n = min(20, n_ep)
    per_run = np.mean(arr[:, -last_n:], axis=1)  # (num_runs,)
    return np.mean(per_run), np.std(per_run)


def main():
    print("=" * 60)
    print("  实验2: 动作空间消融 (Full 6D vs Naive Max Power 2D)")
    print("=" * 60)
    print(f"  Runs: {NUM_RUNS}, Episodes: {NUM_EPISODES}")
    print(f"  Naive Baseline: c1_load={FIXED_COMPRESSOR_ACTIONS[0]}, c2_load={FIXED_COMPRESSOR_ACTIONS[1]}, "
          f"c3_pressure={FIXED_COMPRESSOR_ACTIONS[2]}, bypass={FIXED_COMPRESSOR_ACTIONS[3]}")
    print("=" * 60)

    print("\n[1/2] Training SAC Full 6D (compressor intelligent control)...")
    r_full, p_full, c_full, ch_full, bp_full = train_sac_full_6d(NUM_EPISODES, NUM_RUNS)
    rewards_full = np.mean(r_full, axis=0)
    profits_full = np.mean(p_full, axis=0)

    print("\n[2/2] Training SAC Naive Max Power 2D (no compressor intelligence)...")
    r_fix, p_fix, c_fix, ch_fix, bp_fix = train_sac_fixed_compressor_2d(NUM_EPISODES, NUM_RUNS)
    rewards_fixed = np.mean(r_fix, axis=0)
    profits_fixed = np.mean(p_fix, axis=0)

    # ========== 计算柱状图均值和误差 (Last 20 Ep) ==========
    r_full_m, r_full_s = _last20_mean_std(r_full)
    r_fix_m, r_fix_s = _last20_mean_std(r_fix)
    p_full_m, p_full_s = _last20_mean_std(p_full)
    p_fix_m, p_fix_s = _last20_mean_std(p_fix)
    c_full_m, c_full_s = _last20_mean_std(c_full)
    c_fix_m, c_fix_s = _last20_mean_std(c_fix)
    ch_full_m, ch_full_s = _last20_mean_std(ch_full)
    ch_fix_m, ch_fix_s = _last20_mean_std(ch_fix)
    bp_full_m, bp_full_s = _last20_mean_std(bp_full)
    bp_fix_m, bp_fix_s = _last20_mean_std(bp_fix)

    print(f"\n  Full 6D:          Reward={r_full_m:.2f}±{r_full_s:.2f}, Profit=${p_full_m:.0f}±{p_full_s:.0f}, "
          f"Comp={c_full_m:.0f}kWh, Chiller={ch_full_m:.1f}kWh, Bypass={bp_full_m:.1f}")
    print(f"  Naive Max Power: Reward={r_fix_m:.2f}±{r_fix_s:.2f}, Profit=${p_fix_m:.0f}±{p_fix_s:.0f}, "
          f"Comp={c_fix_m:.0f}kWh, Chiller={ch_fix_m:.1f}kWh, Bypass={bp_fix_m:.1f}")
    if p_fix_m != 0:
        profit_gain = (p_full_m - p_fix_m) / abs(p_fix_m) * 100
        print(f"\n  Profit improvement (6D vs Naive): {profit_gain:+.1f}%")
    if c_fix_m != 0:
        comp_saving = (c_fix_m - c_full_m) / c_fix_m * 100
        print(f"  Compressor energy saving (6D vs Naive): {comp_saving:+.1f}%")

    # ========== 绘图 2×3 ==========
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9
    fig, axs = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("Exp2: Action Space Ablation — Full 6D vs Naive Max Power 2D\n"
                 "(Compressor Intelligent Control Contribution)",
                 fontsize=11, fontweight="bold")

    names = ["Full 6D", "Naive Max Power 2D"]
    colors = ["#1f77b4", "#ff7f0e"]
    x = np.arange(2)
    width = 0.5

    # (a) Reward 柱状图 + 误差棒
    axs[0, 0].bar(x, [r_full_m, r_fix_m], width, yerr=[r_full_s, r_fix_s], color=colors,
                  edgecolor="gray", linewidth=0.5, capsize=4, error_kw={"elinewidth": 1.5})
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(names)
    axs[0, 0].set_ylabel("Avg Reward (Last 20 Ep)")
    axs[0, 0].set_title("(a) Reward")
    axs[0, 0].grid(True, axis="y", alpha=0.3, linestyle="--")
    for i, (m, s) in enumerate([(r_full_m, r_full_s), (r_fix_m, r_fix_s)]):
        axs[0, 0].text(i, m + s + 0.3, f"{m:.1f}±{s:.1f}", ha="center", va="bottom", fontsize=8)

    # (b) Profit 柱状图 + 误差棒
    axs[0, 1].bar(x, [p_full_m, p_fix_m], width, yerr=[p_full_s, p_fix_s], color=colors,
                  edgecolor="gray", linewidth=0.5, capsize=4, error_kw={"elinewidth": 1.5})
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(names)
    axs[0, 1].set_ylabel("Avg Profit (Last 20 Ep, $)")
    axs[0, 1].set_title("(b) Profit")
    axs[0, 1].grid(True, axis="y", alpha=0.3, linestyle="--")
    for i, (m, s) in enumerate([(p_full_m, p_full_s), (p_fix_m, p_fix_s)]):
        axs[0, 1].text(i, m + s + 20, f"${m:.0f}±{s:.0f}", ha="center", va="bottom", fontsize=8)

    # (c) Compressor Energy 柱状图 + 误差棒 (kWh/episode)
    axs[0, 2].bar(x, [c_full_m, c_fix_m], width, yerr=[c_full_s, c_fix_s], color=colors,
                  edgecolor="gray", linewidth=0.5, capsize=4, error_kw={"elinewidth": 1.5})
    axs[0, 2].set_xticks(x)
    axs[0, 2].set_xticklabels(names)
    axs[0, 2].set_ylabel("Compressor Energy (kWh/Ep)")
    axs[0, 2].set_title("(c) Compressor Energy (Lower=Better)")
    axs[0, 2].grid(True, axis="y", alpha=0.3, linestyle="--")
    for i, (m, s) in enumerate([(c_full_m, c_full_s), (c_fix_m, c_fix_s)]):
        axs[0, 2].text(i, m + s + 5, f"{m:.0f}±{s:.0f}", ha="center", va="bottom", fontsize=8)

    # (d) Chiller Energy 柱状图 + 误差棒
    axs[1, 0].bar(x, [ch_full_m, ch_fix_m], width, yerr=[ch_full_s, ch_fix_s], color=colors,
                  edgecolor="gray", linewidth=0.5, capsize=4, error_kw={"elinewidth": 1.5})
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(names)
    axs[1, 0].set_ylabel("Chiller Energy (kWh/Ep)")
    axs[1, 0].set_title("(d) Chiller Energy (Lower=Better)")
    axs[1, 0].grid(True, axis="y", alpha=0.3, linestyle="--")
    for i, (m, s) in enumerate([(ch_full_m, ch_full_s), (ch_fix_m, ch_fix_s)]):
        axs[1, 0].text(i, m + s + 2, f"{m:.1f}±{s:.1f}", ha="center", va="bottom", fontsize=8)

    # (e) Bypass Activations 柱状图 + 误差棒
    axs[1, 1].bar(x, [bp_full_m, bp_fix_m], width, yerr=[bp_full_s, bp_fix_s], color=colors,
                  edgecolor="gray", linewidth=0.5, capsize=4, error_kw={"elinewidth": 1.5})
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(names)
    axs[1, 1].set_ylabel("Avg Bypass/Ep (Last 20)")
    axs[1, 1].set_title("(e) Bypass Activations (Higher=More Energy Saved)")
    axs[1, 1].grid(True, axis="y", alpha=0.3, linestyle="--")
    for i, (m, s) in enumerate([(bp_full_m, bp_full_s), (bp_fix_m, bp_fix_s)]):
        axs[1, 1].text(i, m + s + 1, f"{m:.1f}±{s:.1f}", ha="center", va="bottom", fontsize=8)

    # (f) Reward 与 Profit 曲线
    ep_range = range(MA_WINDOW - 1, len(rewards_full))
    axs[1, 2].plot(ep_range, moving_average(rewards_full, MA_WINDOW), color=colors[0],
                   linewidth=2, label="Reward (Full 6D)")
    axs[1, 2].plot(ep_range, moving_average(rewards_fixed, MA_WINDOW), color=colors[1],
                   linewidth=2, label="Reward (Fixed 2D)")
    ax2 = axs[1, 2].twinx()
    ax2.plot(ep_range, moving_average(profits_full, MA_WINDOW), color=colors[0],
             linewidth=1.5, linestyle="--", alpha=0.8, label="Profit (Full 6D)")
    ax2.plot(ep_range, moving_average(profits_fixed, MA_WINDOW), color=colors[1],
             linewidth=1.5, linestyle="--", alpha=0.8, label="Profit (Fixed 2D)")
    axs[1, 2].set_xlabel("Episode")
    axs[1, 2].set_ylabel("Reward", color="black")
    ax2.set_ylabel("Profit ($)", color="gray")
    axs[1, 2].set_title("(f) Reward & Profit Curves (MA{})".format(MA_WINDOW))
    axs[1, 2].legend(loc="upper left", fontsize=7)
    ax2.legend(loc="center right", fontsize=7)
    axs[1, 2].grid(True, alpha=0.3, linestyle="--")

    savename = "Figure_2_exp2_7D_vs_NaiveMaxPower.png"
    plt.savefig(savename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved: {savename}")


if __name__ == "__main__":
    main()
