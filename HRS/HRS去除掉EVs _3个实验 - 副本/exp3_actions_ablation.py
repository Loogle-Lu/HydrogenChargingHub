"""
实验3: DRL 与 Rule-Based Baseline 对比

对比:
1) Full 6D DDPG (完整智能控制)
2) Rule-Service (服务优先, 不训练)
3) Rule-Profit (利润优先, 不训练)

输出:
- Figure_2_exp3_reward_profit.png: 1×2 — Reward / Profit 滑动平均曲线
"""

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import gym
from pathlib import Path
from tqdm.auto import tqdm
from gym import spaces
from env import HydrogenEnv
from DDPG import DDPG, ReplayBuffer
from config import Config


# ======================== 配置（与 compare.py 中 DDPG 公平基线对齐）====================
NUM_RUNS = 5
NUM_EPISODES = 1000
WARMUP_STEPS = 250
BATCH_SIZE = 256
LR = 3e-4
MA_WINDOW = 20
BUFFER_CAPACITY = 200_000
SCRIPT_DIR = Path(__file__).resolve().parent

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

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

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


def _mean_std_ma_curves(R, window):
    """R: (n_runs, n_ep)"""
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


def train_ddpg_full_6d(num_episodes, num_runs):
    """
    DDPG 完整 6 维动作 (含压缩机智能控制)
    返回: (all_rewards, all_profits, all_comp_energy, all_chiller_energy, all_bypass, all_c3_energy)
    均为 shape (num_runs, num_episodes)
    """
    dt = Config.dt
    all_rewards, all_profits = [], []
    all_comp_energy, all_chiller_energy, all_bypass, all_c3_energy = [], [], [], []
    progress = tqdm(total=num_runs * num_episodes, desc="Full 6D training",
                    unit="ep", leave=False)
    for run in range(num_runs):
        set_seed(42 + run)
        env = HydrogenEnv()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = DDPG(state_dim, action_dim, lr=LR)
        replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)

        run_rewards, run_profits = [], []
        run_comp, run_chiller, run_bypass, run_c3 = [], [], [], []
        total_steps = 0

        for ep in range(num_episodes):
            state = env.reset(episode_index=ep)
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
            progress.update(1)
            progress.set_postfix(run=f"{run + 1}/{num_runs}",
                                 episode=f"{ep + 1}/{num_episodes}",
                                 reward=f"{ep_reward:.2f}",
                                 profit=f"{ep_profit:.2f}")

        all_rewards.append(run_rewards)
        all_profits.append(run_profits)
        all_comp_energy.append(run_comp)
        all_chiller_energy.append(run_chiller)
        all_bypass.append(run_bypass)
        all_c3_energy.append(run_c3)
    progress.close()

    arr = lambda x: np.array(x)
    return (
        arr(all_rewards),
        arr(all_profits),
        arr(all_comp_energy),
        arr(all_chiller_energy),
        arr(all_bypass),
        arr(all_c3_energy),
    )


def train_ddpg_fixed_compressor_2d(num_episodes, num_runs):
    """
    DDPG 固定压缩机 (2 维有效动作: ele, fc)
    返回: (all_rewards, all_profits, all_comp_energy, all_chiller_energy, all_bypass, all_c3_energy)
    """
    dt = Config.dt
    all_rewards, all_profits = [], []
    all_comp_energy, all_chiller_energy, all_bypass, all_c3_energy = [], [], [], []
    progress = tqdm(total=num_runs * num_episodes, desc="Naive 2D training",
                    unit="ep", leave=False)
    for run in range(num_runs):
        set_seed(42 + run)
        env_raw = HydrogenEnv()
        env = FixedCompressorActionWrapper(env_raw)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = DDPG(state_dim, action_dim, lr=LR)
        replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)

        run_rewards, run_profits = [], []
        run_comp, run_chiller, run_bypass, run_c3 = [], [], [], []
        total_steps = 0

        for ep in range(num_episodes):
            state = env.reset(episode_index=ep)
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
            progress.update(1)
            progress.set_postfix(run=f"{run + 1}/{num_runs}",
                                 episode=f"{ep + 1}/{num_episodes}",
                                 reward=f"{ep_reward:.2f}",
                                 profit=f"{ep_profit:.2f}")

        all_rewards.append(run_rewards)
        all_profits.append(run_profits)
        all_comp_energy.append(run_comp)
        all_chiller_energy.append(run_chiller)
        all_bypass.append(run_bypass)
        all_c3_energy.append(run_c3)
    progress.close()

    arr = lambda x: np.array(x)
    return (
        arr(all_rewards),
        arr(all_profits),
        arr(all_comp_energy),
        arr(all_chiller_energy),
        arr(all_bypass),
        arr(all_c3_energy),
    )


def _build_rule_action(state, mode):
    """
    基于当前 state 构造 6 维固定规则动作:
    [ele, fc, c1_cool, c2_cool, c3_pressure_bias, bypass_bias]
    """
    s = np.asarray(state, dtype=np.float32).flatten()
    price_norm = float(np.clip(s[0], 0.0, 1.5))
    t3_avg_soc = float(np.mean(s[4:7]))
    queue_pressure = float(s[7] + s[8])  # 两类排队归一化和

    low_price_norm = Config.price_threshold_low / Config.price_max
    high_price_norm = Config.price_threshold_high / Config.price_max

    if mode == "service":
        # 有排队/库存偏低时优先补氢，不进行价格择时。
        urgent = (queue_pressure > 0.05) or (t3_avg_soc < 0.60)
        ele = 1.0 if urgent else 0.65
        fc = 0.0 if urgent else 0.20
        c1_cool = 1.0 if price_norm < low_price_norm else 0.30
        c2_cool = 1.0 if price_norm < low_price_norm else 0.30
        c3_pressure_bias = 1.0   # 固定最大压力策略
        bypass_bias = 0.0        # 固定不旁路
    elif mode == "profit":
        # 保守利润规则: 只做低价补氢，不做激进 FC 高价套利。
        # 这样代表常见 rule-based energy management，而不是手工调优的套利策略。
        if price_norm < low_price_norm:
            ele = 0.75 if t3_avg_soc < 0.70 else 0.25
            fc = 0.0
            c1_cool, c2_cool = 0.50, 0.50
        elif price_norm > high_price_norm:
            ele = 0.0
            fc = 0.20 if (t3_avg_soc > 0.80 and queue_pressure < 0.03) else 0.0
            c1_cool, c2_cool = 0.0, 0.0
        else:
            ele = 0.10 if t3_avg_soc < 0.55 else 0.0
            fc = 0.0
            c1_cool, c2_cool = 0.30, 0.30
        c3_pressure_bias = 1.0
        bypass_bias = 0.0
    else:
        raise ValueError(f"Unknown rule mode: {mode}")

    return np.array(
        [ele, fc, c1_cool, c2_cool, c3_pressure_bias, bypass_bias],
        dtype=np.float32,
    )


def evaluate_rule_baseline(num_episodes, num_runs, mode):
    """
    评估纯 Rule-based 策略 (不训练, 不更新参数)。
    返回: (all_rewards, all_profits), shape 均为 (num_runs, num_episodes)
    """
    all_rewards, all_profits = [], []
    desc = "Rule-Service" if mode == "service" else "Rule-Profit"
    progress = tqdm(total=num_runs * num_episodes, desc=f"{desc} eval", unit="ep", leave=False)

    for run in range(num_runs):
        set_seed(42 + run)
        env = HydrogenEnv()
        run_rewards, run_profits = [], []

        for ep in range(num_episodes):
            state = env.reset(episode_index=ep)
            ep_reward, ep_profit = 0.0, 0.0
            done = False

            while not done:
                action = _build_rule_action(state, mode)
                next_state, reward, done, info = env.step(action)
                state = next_state
                ep_reward += reward
                ep_profit += info.get("profit", 0.0)

            run_rewards.append(ep_reward)
            run_profits.append(ep_profit)
            progress.update(1)
            progress.set_postfix(run=f"{run + 1}/{num_runs}", episode=f"{ep + 1}/{num_episodes}")

        all_rewards.append(run_rewards)
        all_profits.append(run_profits)

    progress.close()
    return np.array(all_rewards), np.array(all_profits)


def _last20_mean_std(arr):
    """arr: (num_runs, num_episodes), 返回每 run 末 20 ep 均值的 mean 和 std"""
    n_ep = arr.shape[1]
    last_n = min(20, n_ep)
    per_run = np.mean(arr[:, -last_n:], axis=1)  # (num_runs,)
    return np.mean(per_run), np.std(per_run)


def main():
    print("=" * 60)
    print("  实验3: DRL vs Rule-Based Baselines")
    print("=" * 60)
    print(f"  Runs: {NUM_RUNS}, Episodes: {NUM_EPISODES}")
    print("=" * 60)

    print("\n[1/3] Training DDPG Full 6D...")
    r_full, p_full, c_full, ch_full, bp_full, c3_full = train_ddpg_full_6d(
        NUM_EPISODES, NUM_RUNS
    )

    print("\n[2/3] Evaluating Rule-Service (no DRL)...")
    r_srv, p_srv = evaluate_rule_baseline(NUM_EPISODES, NUM_RUNS, mode="service")

    print("\n[3/3] Evaluating Rule-Profit (no DRL)...")
    r_pft, p_pft = evaluate_rule_baseline(NUM_EPISODES, NUM_RUNS, mode="profit")

    r_full_m, r_full_s = _last20_mean_std(r_full)
    r_srv_m, r_srv_s = _last20_mean_std(r_srv)
    r_pft_m, r_pft_s = _last20_mean_std(r_pft)
    p_full_m, p_full_s = _last20_mean_std(p_full)
    p_srv_m, p_srv_s = _last20_mean_std(p_srv)
    p_pft_m, p_pft_s = _last20_mean_std(p_pft)

    print(f"\n  Full 6D      : Reward={r_full_m:.2f}±{r_full_s:.2f}, Profit=${p_full_m:.0f}±{p_full_s:.0f}")
    print(f"  Rule-Service : Reward={r_srv_m:.2f}±{r_srv_s:.2f}, Profit=${p_srv_m:.0f}±{p_srv_s:.0f}")
    print(f"  Rule-Profit  : Reward={r_pft_m:.2f}±{r_pft_s:.2f}, Profit=${p_pft_m:.0f}±{p_pft_s:.0f}")

    # ========== 绘图：仅 1 张 1×2 ==========
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9

    colors = {
        "full": "#ff7f0e",      # DRL
        "service": "#1f3a5f",   # 深蓝 baseline
        "profit": "#4d4d4d",    # 深灰 baseline
    }

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    fig.suptitle("Exp3: DRL vs Rule-Based Baselines", fontsize=13, fontweight="bold")

    x_r, m_r_full, s_r_full = _mean_std_ma_curves(r_full, MA_WINDOW)
    _, m_r_srv, s_r_srv = _mean_std_ma_curves(r_srv, MA_WINDOW)
    _, m_r_pft, s_r_pft = _mean_std_ma_curves(r_pft, MA_WINDOW)
    axs[0].fill_between(x_r, m_r_full - s_r_full, m_r_full + s_r_full, color=colors["full"], alpha=0.20, linewidth=0)
    axs[0].fill_between(x_r, m_r_srv - s_r_srv, m_r_srv + s_r_srv, color=colors["service"], alpha=0.20, linewidth=0)
    axs[0].fill_between(x_r, m_r_pft - s_r_pft, m_r_pft + s_r_pft, color=colors["profit"], alpha=0.20, linewidth=0)
    axs[0].plot(x_r, m_r_full, color=colors["full"], linewidth=2.0, label="Full 6D (DDPG)")
    axs[0].plot(x_r, m_r_srv, color=colors["service"], linewidth=2.0, label="Rule-Service")
    axs[0].plot(x_r, m_r_pft, color=colors["profit"], linewidth=2.0, label="Rule-Profit")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")
    axs[0].set_title("Reward")
    axs[0].legend(loc="best", fontsize=8)
    axs[0].grid(True, alpha=0.3, linestyle="--")

    _, m_p_full, s_p_full = _mean_std_ma_curves(p_full, MA_WINDOW)
    _, m_p_srv, s_p_srv = _mean_std_ma_curves(p_srv, MA_WINDOW)
    _, m_p_pft, s_p_pft = _mean_std_ma_curves(p_pft, MA_WINDOW)
    axs[1].fill_between(x_r, m_p_full - s_p_full, m_p_full + s_p_full, color=colors["full"], alpha=0.20, linewidth=0)
    axs[1].fill_between(x_r, m_p_srv - s_p_srv, m_p_srv + s_p_srv, color=colors["service"], alpha=0.20, linewidth=0)
    axs[1].fill_between(x_r, m_p_pft - s_p_pft, m_p_pft + s_p_pft, color=colors["profit"], alpha=0.20, linewidth=0)
    axs[1].plot(x_r, m_p_full, color=colors["full"], linewidth=2.0, label="Full 6D (DDPG)")
    axs[1].plot(x_r, m_p_srv, color=colors["service"], linewidth=2.0, label="Rule-Service")
    axs[1].plot(x_r, m_p_pft, color=colors["profit"], linewidth=2.0, label="Rule-Profit")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Profit ($)")
    axs[1].set_title("Profit")
    axs[1].legend(loc="best", fontsize=8)
    axs[1].grid(True, alpha=0.3, linestyle="--")

    fig_name = SCRIPT_DIR / "Figure_2_exp3_reward_profit.png"
    plt.savefig(fig_name, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigure saved:\n  {fig_name}")


if __name__ == "__main__":
    main()
