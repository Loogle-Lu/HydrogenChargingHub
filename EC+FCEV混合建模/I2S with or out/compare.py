"""
七算法+双 Baseline 对比脚本: PPO vs A2C vs SAC vs TD3 vs DDPG vs REINFORCE vs Random

对比维度:
- I2S 约束开启 / 关闭
- Reward 曲线
- Average Profit 曲线

使用方法:
    cd "I2S with or out"
    python compare.py
"""

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import time
from env import HydrogenEnv
from PPO import PPO
from A2C import A2C
from SAC import SAC, ReplayBuffer
from TD3 import TD3
from DDPG import DDPG
from REINFORCE import REINFORCE
from RandomBaseline import RandomBaseline


# ======================== 配置 ========================
NUM_RUNS = 3             # 每个算法重复运行次数，取平均值 (可调整)
NUM_EPISODES = 100       # 每个算法训练的 Episode 数
WARMUP_STEPS = 500       # Off-Policy 算法的随机探索步数
BATCH_SIZE = 256         # Off-Policy 算法的 mini-batch 大小
LR = 3e-4                # 统一学习率
MA_WINDOW = 20           # 移动平均窗口大小

# 算法列表 (RL 算法 + Baseline)
ALGO_NAMES = ['PPO', 'A2C', 'SAC', 'TD3', 'DDPG', 'REINFORCE', 'Random']

# I2S 版本
I2S_SETTINGS = [
    (True, "With I2S"),
    (False, "Without I2S"),
]

# 颜色方案 (7色)
COLORS = {
    'PPO':       '#1f77b4',   # 蓝色
    'A2C':       '#9467bd',   # 紫色
    'SAC':       '#ff7f0e',   # 橙色
    'TD3':       '#2ca02c',   # 绿色
    'DDPG':      '#d62728',   # 红色
    'REINFORCE': '#8c564b',   # 棕色
    'Random':    '#7f7f7f',   # 灰色
}


# ======================== 工具函数 ========================

def set_seed(seed):
    """设置随机种子以确保可复现性"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================== 训练函数 ========================

def train_on_policy(algo_name, agent, enable_i2s, num_episodes=NUM_EPISODES):
    """训练 On-Policy 算法 (PPO / A2C)"""
    env = HydrogenEnv(enable_i2s_constraint=enable_i2s)

    all_rewards = []
    all_profits = []

    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0
        ep_profit = 0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, done)
            state = next_state
            ep_reward += reward
            ep_profit += info.get('profit', 0.0)

        agent.update()
        agent.step_scheduler()
        all_rewards.append(ep_reward)
        all_profits.append(ep_profit)

        if (ep + 1) % 20 == 0:
            print(f"    {algo_name:<5s} Episode {ep+1:>3d}/{num_episodes}, "
                  f"Reward: {ep_reward:>8.2f}, Profit: {ep_profit:>10.2f}")

    return all_rewards, all_profits


def train_baseline_random(algo_name, enable_i2s, num_episodes=NUM_EPISODES):
    """运行 Random 基线 (无学习，仅随机采样)"""
    env = HydrogenEnv(enable_i2s_constraint=enable_i2s)
    agent = RandomBaseline(env.action_space)

    all_rewards = []
    all_profits = []

    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0
        ep_profit = 0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
            ep_reward += reward
            ep_profit += info.get('profit', 0.0)

        all_rewards.append(ep_reward)
        all_profits.append(ep_profit)

        if (ep + 1) % 20 == 0:
            print(f"    {algo_name:<5s} Episode {ep+1:>3d}/{num_episodes}, "
                  f"Reward: {ep_reward:>8.2f}, Profit: {ep_profit:>10.2f}")

    return all_rewards, all_profits


def train_off_policy(algo_name, agent, enable_i2s, num_episodes=NUM_EPISODES,
                     warmup_steps=WARMUP_STEPS, batch_size=BATCH_SIZE):
    """训练 Off-Policy 算法 (SAC / TD3 / DDPG)"""
    env = HydrogenEnv(enable_i2s_constraint=enable_i2s)
    replay_buffer = ReplayBuffer(capacity=100000)

    all_rewards = []
    all_profits = []
    total_steps = 0

    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0
        ep_profit = 0
        done = False

        while not done:
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, evaluate=False)

            next_state, reward, done, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, float(done))

            if total_steps >= warmup_steps and len(replay_buffer) >= batch_size:
                agent.update(replay_buffer, batch_size)

            state = next_state
            ep_reward += reward
            ep_profit += info.get('profit', 0.0)
            total_steps += 1

        all_rewards.append(ep_reward)
        all_profits.append(ep_profit)

        if (ep + 1) % 20 == 0:
            print(f"    {algo_name:<5s} Episode {ep+1:>3d}/{num_episodes}, "
                  f"Reward: {ep_reward:>8.2f}, Profit: {ep_profit:>10.2f}")

    return all_rewards, all_profits


# ======================== 可视化 ========================

def moving_average(data, window):
    """计算移动平均"""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_i2s_comparison(results_by_condition):
    """
    绘制四张图:
    - With I2S: Reward / Avg Profit
    - Without I2S: Reward / Avg Profit
    """
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8

    fig, axs = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
    fig.suptitle('Algorithm Comparison with/without I2S Constraint',
                 fontsize=12, fontweight='bold')

    panels = [
        (True, 0, 0, 'With I2S: Reward'),
        (True, 0, 1, 'With I2S: Average Profit'),
        (False, 1, 0, 'Without I2S: Reward'),
        (False, 1, 1, 'Without I2S: Average Profit'),
    ]

    for enable_i2s, row, col, title in panels:
        ax = axs[row, col]
        results = results_by_condition[enable_i2s]

        for name in ALGO_NAMES:
            rewards, profits = results[name]
            color = COLORS[name]

            if "Reward" in title:
                ax.plot(rewards, alpha=0.10, color=color, linewidth=0.9)
                ma = moving_average(rewards, MA_WINDOW)
                ax.plot(range(MA_WINDOW - 1, len(rewards)), ma,
                        color=color, linewidth=1.6, label=f'{name} (MA{MA_WINDOW})')
                ax.set_ylabel('Episode Reward')
            else:
                ax.plot(profits, alpha=0.10, color=color, linewidth=0.9)
                ma = moving_average(profits, MA_WINDOW)
                ax.plot(range(MA_WINDOW - 1, len(profits)), ma,
                        color=color, linewidth=1.6, label=f'{name} (MA{MA_WINDOW})')
                ax.set_ylabel('Episode Profit ($)')

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.legend(loc='best', ncol=3, frameon=False)
        ax.grid(True, alpha=0.3, linestyle='--')
    plt.show()


def print_summary(results_by_condition):
    """打印性能统计表"""
    col_w = 12
    header = f"{'Metric':<30s}" + "".join(f"{n:>{col_w}s}" for n in ALGO_NAMES)
    sep_len = 30 + col_w * len(ALGO_NAMES)

    for enable_i2s, label in I2S_SETTINGS:
        results = results_by_condition[enable_i2s]
        print("\n" + "=" * sep_len)
        print(f"           ALGORITHM COMPARISON SUMMARY ({label})")
        print("=" * sep_len)
        print(header)
        print("-" * sep_len)

        metrics = {}
        for name in ALGO_NAMES:
            rewards, profits = results[name]
            metrics[name] = {
                'avg_reward_20': np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards),
                'std_reward_20': np.std(rewards[-20:]) if len(rewards) >= 20 else np.std(rewards),
                'best_reward': np.max(rewards),
                'worst_reward': np.min(rewards),
                'avg_profit_20': np.mean(profits[-20:]) if len(profits) >= 20 else np.mean(profits),
                'total_profit': np.sum(profits),
            }

        def row(label, key, fmt='.2f'):
            line = f"{label:<30s}"
            for name in ALGO_NAMES:
                v = metrics[name][key]
                line += f"{v:{col_w}{fmt}}"
            print(line)

        row('Avg Reward (Last 20 Ep)', 'avg_reward_20')
        row('Std Reward (Last 20 Ep)', 'std_reward_20')
        row('Best Episode Reward', 'best_reward')
        row('Worst Episode Reward', 'worst_reward')
        print("-" * sep_len)
        row('Avg Profit (Last 20 Ep)', 'avg_profit_20')
        row('Total Profit (All Ep)', 'total_profit')
        print("=" * sep_len)


# ======================== 主函数 ========================

def main():
    print("=" * 70)
    print("  RL Algorithm Comparison: PPO vs A2C vs SAC vs TD3 vs DDPG vs REINFORCE vs Random")
    print("=" * 70)
    print(f"  Runs per Algorithm:     {NUM_RUNS} (取平均)")
    print(f"  Episodes per Algorithm: {NUM_EPISODES}")
    print(f"  Off-Policy Warmup:      {WARMUP_STEPS} steps")
    print(f"  Learning Rate:          {LR}")
    print("=" * 70)

    results_by_condition = {}

    # 获取环境维度
    env_tmp = HydrogenEnv()
    state_dim = env_tmp.observation_space.shape[0]
    action_dim = env_tmp.action_space.shape[0]
    del env_tmp

    for enable_i2s, label in I2S_SETTINGS:
        print("\n" + "=" * 70)
        print(f"  Running Condition: {label}")
        print("=" * 70)

        results = {}
        times = {}

        # --- 1. PPO (On-Policy) ---
        print(f"\n[1/7] Training PPO ({NUM_RUNS} runs × {NUM_EPISODES} episodes)...")
        all_r, all_p = [], []
        t0 = time.time()
        for run in range(NUM_RUNS):
            set_seed(42 + run)
            agent = PPO(state_dim, action_dim, lr=LR)
            r, p = train_on_policy('PPO', agent, enable_i2s, NUM_EPISODES)
            all_r.append(r)
            all_p.append(p)
        times['PPO'] = time.time() - t0
        results['PPO'] = (np.mean(all_r, axis=0), np.mean(all_p, axis=0))
        print(f"  PPO  done in {times['PPO']:.1f}s, Final MA: {np.mean(results['PPO'][0][-20:]):.2f}")

        # --- 2. A2C (On-Policy) ---
        print(f"\n[2/7] Training A2C ({NUM_RUNS} runs × {NUM_EPISODES} episodes)...")
        all_r, all_p = [], []
        t0 = time.time()
        for run in range(NUM_RUNS):
            set_seed(42 + run)
            agent = A2C(state_dim, action_dim, lr=LR)
            r, p = train_on_policy('A2C', agent, enable_i2s, NUM_EPISODES)
            all_r.append(r)
            all_p.append(p)
        times['A2C'] = time.time() - t0
        results['A2C'] = (np.mean(all_r, axis=0), np.mean(all_p, axis=0))
        print(f"  A2C  done in {times['A2C']:.1f}s, Final MA: {np.mean(results['A2C'][0][-20:]):.2f}")

        # --- 3. SAC (Off-Policy) ---
        print(f"\n[3/7] Training SAC ({NUM_RUNS} runs × {NUM_EPISODES} episodes)...")
        all_r, all_p = [], []
        t0 = time.time()
        for run in range(NUM_RUNS):
            set_seed(42 + run)
            agent = SAC(state_dim, action_dim, lr=LR)
            r, p = train_off_policy('SAC', agent, enable_i2s, NUM_EPISODES)
            all_r.append(r)
            all_p.append(p)
        times['SAC'] = time.time() - t0
        results['SAC'] = (np.mean(all_r, axis=0), np.mean(all_p, axis=0))
        print(f"  SAC  done in {times['SAC']:.1f}s, Final MA: {np.mean(results['SAC'][0][-20:]):.2f}")

        # --- 4. TD3 (Off-Policy) ---
        print(f"\n[4/7] Training TD3 ({NUM_RUNS} runs × {NUM_EPISODES} episodes)...")
        all_r, all_p = [], []
        t0 = time.time()
        for run in range(NUM_RUNS):
            set_seed(42 + run)
            agent = TD3(state_dim, action_dim, lr=LR)
            r, p = train_off_policy('TD3', agent, enable_i2s, NUM_EPISODES)
            all_r.append(r)
            all_p.append(p)
        times['TD3'] = time.time() - t0
        results['TD3'] = (np.mean(all_r, axis=0), np.mean(all_p, axis=0))
        print(f"  TD3  done in {times['TD3']:.1f}s, Final MA: {np.mean(results['TD3'][0][-20:]):.2f}")

        # --- 5. DDPG (Off-Policy) ---
        print(f"\n[5/7] Training DDPG ({NUM_RUNS} runs × {NUM_EPISODES} episodes)...")
        all_r, all_p = [], []
        t0 = time.time()
        for run in range(NUM_RUNS):
            set_seed(42 + run)
            agent = DDPG(state_dim, action_dim, lr=LR)
            r, p = train_off_policy('DDPG', agent, enable_i2s, NUM_EPISODES)
            all_r.append(r)
            all_p.append(p)
        times['DDPG'] = time.time() - t0
        results['DDPG'] = (np.mean(all_r, axis=0), np.mean(all_p, axis=0))
        print(f"  DDPG done in {times['DDPG']:.1f}s, Final MA: {np.mean(results['DDPG'][0][-20:]):.2f}")

        # --- 6. REINFORCE (On-Policy Baseline) ---
        print(f"\n[6/7] Training REINFORCE ({NUM_RUNS} runs × {NUM_EPISODES} episodes)...")
        all_r, all_p = [], []
        t0 = time.time()
        for run in range(NUM_RUNS):
            set_seed(42 + run)
            agent = REINFORCE(state_dim, action_dim, lr=LR)
            r, p = train_on_policy('REINFORCE', agent, enable_i2s, NUM_EPISODES)
            all_r.append(r)
            all_p.append(p)
        times['REINFORCE'] = time.time() - t0
        results['REINFORCE'] = (np.mean(all_r, axis=0), np.mean(all_p, axis=0))
        print(f"  REINFORCE done in {times['REINFORCE']:.1f}s, Final MA: {np.mean(results['REINFORCE'][0][-20:]):.2f}")

        # --- 7. Random (Baseline, 无训练) ---
        print(f"\n[7/7] Running Random baseline ({NUM_RUNS} runs × {NUM_EPISODES} episodes)...")
        all_r, all_p = [], []
        t0 = time.time()
        for run in range(NUM_RUNS):
            set_seed(42 + run)
            r, p = train_baseline_random('Random', enable_i2s, NUM_EPISODES)
            all_r.append(r)
            all_p.append(p)
        times['Random'] = time.time() - t0
        results['Random'] = (np.mean(all_r, axis=0), np.mean(all_p, axis=0))
        print(f"  Random done in {times['Random']:.1f}s, Final MA: {np.mean(results['Random'][0][-20:]):.2f}")

        results_by_condition[enable_i2s] = results

        # --- 训练时间 ---
        time_str = " | ".join(f"{n}={times[n]:.1f}s" for n in ALGO_NAMES)
        print(f"\n  Training Time: {time_str}")

    # --- 统计表 ---
    print_summary(results_by_condition)

    # --- 可视化 ---
    print("Generating comparison plots...")
    plot_i2s_comparison(results_by_condition)


if __name__ == "__main__":
    main()
