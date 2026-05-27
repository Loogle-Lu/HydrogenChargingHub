"""
实验2: 动作空间消融实验

对比「完整 6 维动作(含压缩机智能控制)」与「Naive 固定压缩机动作(仅 2 维有效)」的收益差异。
突出压缩机智能控制的边际贡献。
(环境已移除 EV，State 11D，Action 6D)

- Full 6D: [ele, fc, c1_cool, c2_cool, c3_pressure_bias, bypass_bias]
    c1_cool/c2_cool 控制级间冷却强度 (0=轻度省冷却电, 1=深度省压缩功)
    流量由储罐需求自动驱动, RL Agent 可学习「低电价深冷却, 高电价轻冷却」策略
- Naive Max Power 2D: [ele, fc], 压缩机固定最大功率参数
  [c1_cool=0.0, c2_cool=0.0, c3_pressure_bias=1.0, bypass_bias=0.0]
  代表「工业常规: 无智能冷却/旁路/APC, 固定最大功率运行」Baseline:
    - c1_cool=c2_cool=0.0: 无级间深度冷却 → 压缩出口温度高 → 功耗大
    - c3_pressure_bias=1.0: 始终最大压力输出 → C3 功耗浪费
    - bypass_bias=0.0: 从不旁路 → 无法跳过不必要的压缩
  + 固定控制 → 温控不稳 → SAE J2601 合规性降低 → 保守充装 → 有效吞吐降低

输出:
- Figure_2_exp2_reward_profit.png: 1×2 — Reward / Profit 训练曲线（20 episode 滑动平均）
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

# Naive 固定压缩机参数 (c1_cool, c2_cool, c3_pressure_bias, bypass_bias)
# [0.0, 0.0, 1.0, 0.0] = 无动态冷却 / 最大C3输出压力 / 无旁路
# 代表「工业常规: 固定最大参数运行, 无智能控制」的 Baseline
# c1/c2_cool=0: 不进行级间深度冷却 → 压缩功耗高
# c3_pressure_bias=1.0: 始终压缩至最大压力 → C3 功耗浪费
# bypass_bias=0.0: 从不旁路 → 无法跳过不必要的压缩
FIXED_COMPRESSOR_ACTIONS = [0.0, 0.0, 1.0, 0.0]


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
            self.fixed[0],        # c1_cool
            self.fixed[1],        # c2_cool
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
    返回: (all_rewards, all_profits, all_comp_energy, all_chiller_energy, all_bypass, all_c3_energy)
    均为 shape (num_runs, num_episodes)
    """
    dt = Config.dt
    all_rewards, all_profits = [], []
    all_comp_energy, all_chiller_energy, all_bypass, all_c3_energy = [], [], [], []
    for run in range(num_runs):
        set_seed(42 + run)
        env = HydrogenEnv(enable_i2s_constraint=True)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = SAC(state_dim, action_dim, lr=LR)
        replay_buffer = ReplayBuffer(capacity=100000)

        run_rewards, run_profits = [], []
        run_comp, run_chiller, run_bypass, run_c3 = [], [], [], []
        total_steps = 0

        for ep in range(num_episodes):
            state = env.reset()
            ep_reward, ep_profit = 0.0, 0.0
            ep_comp_kwh, ep_chiller_kwh, ep_c3_kwh = 0.0, 0.0, 0.0
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
                ep_c3_kwh += info.get("comp_c3_power", 0.0) * dt
                total_steps += 1

            bp = info.get("bypass_activations", {"c1": 0, "c2": 0, "c3": 0})
            ep_bypass = bp.get("c1", 0) + bp.get("c2", 0) + bp.get("c3", 0)

            run_rewards.append(ep_reward)
            run_profits.append(ep_profit)
            run_comp.append(ep_comp_kwh)
            run_chiller.append(ep_chiller_kwh)
            run_bypass.append(ep_bypass)
            run_c3.append(ep_c3_kwh)

        all_rewards.append(run_rewards)
        all_profits.append(run_profits)
        all_comp_energy.append(run_comp)
        all_chiller_energy.append(run_chiller)
        all_bypass.append(run_bypass)
        all_c3_energy.append(run_c3)

    arr = lambda x: np.array(x)
    return (
        arr(all_rewards),
        arr(all_profits),
        arr(all_comp_energy),
        arr(all_chiller_energy),
        arr(all_bypass),
        arr(all_c3_energy),
    )


def train_sac_fixed_compressor_2d(num_episodes, num_runs):
    """
    SAC 固定压缩机 (2 维有效动作: ele, fc)
    返回: (all_rewards, all_profits, all_comp_energy, all_chiller_energy, all_bypass, all_c3_energy)
    """
    dt = Config.dt
    all_rewards, all_profits = [], []
    all_comp_energy, all_chiller_energy, all_bypass, all_c3_energy = [], [], [], []
    for run in range(num_runs):
        set_seed(42 + run)
        env_raw = HydrogenEnv(enable_i2s_constraint=True)
        env = FixedCompressorActionWrapper(env_raw)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = SAC(state_dim, action_dim, lr=LR)
        replay_buffer = ReplayBuffer(capacity=100000)

        run_rewards, run_profits = [], []
        run_comp, run_chiller, run_bypass, run_c3 = [], [], [], []
        total_steps = 0

        for ep in range(num_episodes):
            state = env.reset()
            ep_reward, ep_profit = 0.0, 0.0
            ep_comp_kwh, ep_chiller_kwh, ep_c3_kwh = 0.0, 0.0, 0.0
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
                ep_c3_kwh += info.get("comp_c3_power", 0.0) * dt
                total_steps += 1

            bp = info.get("bypass_activations", {"c1": 0, "c2": 0, "c3": 0})
            ep_bypass = bp.get("c1", 0) + bp.get("c2", 0) + bp.get("c3", 0)

            run_rewards.append(ep_reward)
            run_profits.append(ep_profit)
            run_comp.append(ep_comp_kwh)
            run_chiller.append(ep_chiller_kwh)
            run_bypass.append(ep_bypass)
            run_c3.append(ep_c3_kwh)

        all_rewards.append(run_rewards)
        all_profits.append(run_profits)
        all_comp_energy.append(run_comp)
        all_chiller_energy.append(run_chiller)
        all_bypass.append(run_bypass)
        all_c3_energy.append(run_c3)

    arr = lambda x: np.array(x)
    return (
        arr(all_rewards),
        arr(all_profits),
        arr(all_comp_energy),
        arr(all_chiller_energy),
        arr(all_bypass),
        arr(all_c3_energy),
    )


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
    print(f"  Naive Baseline: c1_cool={FIXED_COMPRESSOR_ACTIONS[0]}, c2_cool={FIXED_COMPRESSOR_ACTIONS[1]}, "
          f"c3_pressure={FIXED_COMPRESSOR_ACTIONS[2]}, bypass={FIXED_COMPRESSOR_ACTIONS[3]}")
    print("=" * 60)

    print("\n[1/2] Training SAC Full 6D (compressor intelligent control)...")
    r_full, p_full, c_full, ch_full, bp_full, c3_full = train_sac_full_6d(
        NUM_EPISODES, NUM_RUNS
    )
    rewards_full = np.mean(r_full, axis=0)
    profits_full = np.mean(p_full, axis=0)

    print("\n[2/2] Training SAC Naive Max Power 2D (no compressor intelligence)...")
    r_fix, p_fix, c_fix, ch_fix, bp_fix, c3_fix = train_sac_fixed_compressor_2d(
        NUM_EPISODES, NUM_RUNS
    )
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
    c3_full_m, c3_full_s = _last20_mean_std(c3_full)
    c3_fix_m, c3_fix_s = _last20_mean_std(c3_fix)

    print(f"\n  Full 6D:          Reward={r_full_m:.2f}±{r_full_s:.2f}, Profit=${p_full_m:.0f}±{p_full_s:.0f}, "
          f"Comp={c_full_m:.0f}kWh, DI/Chiller={ch_full_m:.1f}kWh, Bypass={bp_full_m:.1f}, C3={c3_full_m:.1f}kWh")
    print(f"  Naive Max Power: Reward={r_fix_m:.2f}±{r_fix_s:.2f}, Profit=${p_fix_m:.0f}±{p_fix_s:.0f}, "
          f"Comp={c_fix_m:.0f}kWh, DI/Chiller={ch_fix_m:.1f}kWh, Bypass={bp_fix_m:.1f}, C3={c3_fix_m:.1f}kWh")
    if p_fix_m != 0:
        profit_gain = (p_full_m - p_fix_m) / abs(p_fix_m) * 100
        print(f"\n  Profit improvement (6D vs Naive): {profit_gain:+.1f}%")
    if c_fix_m != 0:
        comp_saving = (c_fix_m - c_full_m) / c_fix_m * 100
        print(f"  Compressor energy saving (6D vs Naive): {comp_saving:+.1f}%")

    # ========== 绘图：仅 Reward / Profit 训练曲线（1×2）==========
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9

    colors = ["#1f77b4", "#ff7f0e"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    fig.suptitle(
        "Exp2: Action Ablation — Full 6D vs Naive 2D (Moving Average)",
        fontsize=11,
        fontweight="bold",
    )

    ep_range = np.arange(MA_WINDOW - 1, len(rewards_full))
    axs[0].plot(ep_range, moving_average(rewards_full, MA_WINDOW), color=colors[0],
                 linewidth=2, label="Full 6D")
    axs[0].plot(ep_range, moving_average(rewards_fixed, MA_WINDOW), color=colors[1],
                 linewidth=2, label="Naive 2D")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward (scaled)")
    axs[0].set_title("(a) Reward")
    axs[0].legend(loc="best", fontsize=8)
    axs[0].grid(True, alpha=0.3, linestyle="--")

    axs[1].plot(ep_range, moving_average(profits_full, MA_WINDOW), color=colors[0],
                 linewidth=2, label="Full 6D")
    axs[1].plot(ep_range, moving_average(profits_fixed, MA_WINDOW), color=colors[1],
                 linewidth=2, label="Naive 2D")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Profit ($)")
    axs[1].set_title("(b) Profit")
    axs[1].legend(loc="best", fontsize=8)
    axs[1].grid(True, alpha=0.3, linestyle="--")

    out_name = "Figure_2_exp2_reward_profit.png"
    plt.savefig(out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigure saved:  {out_name}")


if __name__ == "__main__":
    main()
