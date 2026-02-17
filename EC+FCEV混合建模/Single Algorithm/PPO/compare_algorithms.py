"""
五算法对比脚本: PPO vs A2C vs SAC vs TD3 vs DDPG

On-Policy:  PPO, A2C
Off-Policy: SAC, TD3, DDPG

功能:
1. 分别训练 5 种强化学习算法
2. 收集每个 Episode 的 Reward 和最后一个 Episode 的详细收益数据
3. 生成对比可视化图表 (1×3):
   - 训练奖励曲线对比
   - 最后 Episode 累计利润对比
   - 最终性能指标柱状图

使用方法:
    cd PPO
    python compare_algorithms.py
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from env import HydrogenEnv
from PPO import PPO
from A2C import A2C
from SAC import SAC, ReplayBuffer
from TD3 import TD3
from DDPG import DDPG


# ======================== 配置 ========================
NUM_EPISODES = 200       # 每个算法训练的 Episode 数
WARMUP_STEPS = 500       # Off-Policy 算法的随机探索步数
BATCH_SIZE = 256         # Off-Policy 算法的 mini-batch 大小
LR = 3e-4                # 统一学习率
MA_WINDOW = 20           # 移动平均窗口大小

# 算法列表 (按 On-Policy → Off-Policy 排列)
ALGO_NAMES = ['PPO', 'A2C', 'SAC', 'TD3', 'DDPG']

# 颜色方案 (5色)
COLORS = {
    'PPO':  '#1f77b4',   # 蓝色
    'A2C':  '#9467bd',   # 紫色
    'SAC':  '#ff7f0e',   # 橙色
    'TD3':  '#2ca02c',   # 绿色
    'DDPG': '#d62728',   # 红色
}


# ======================== 训练函数 ========================

def _collect_step_data(info):
    """从 env info 中提取真实经济收益 (不含 arbitrage shaping reward)"""
    ev_r = info.get('revenue_ev', 0)
    fcev_r = info.get('revenue_fcev', 0)
    grid_r = info.get('revenue_grid', 0)
    grid_c = info.get('cost_grid', 0)
    real_profit = ev_r + fcev_r + grid_r - grid_c
    return real_profit, ev_r, fcev_r, grid_r, grid_c


def train_on_policy(algo_name, agent, num_episodes=NUM_EPISODES):
    """训练 On-Policy 算法 (PPO / A2C)"""
    env = HydrogenEnv()

    all_rewards = []
    last_ep_profits = []
    last_ep_revenues = {'ev': [], 'fcev': [], 'grid_sell': [], 'grid_cost': []}

    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, done)
            state = next_state
            ep_reward += reward

            if ep == num_episodes - 1:
                profit, ev_r, fcev_r, grid_r, grid_c = _collect_step_data(info)
                last_ep_profits.append(profit)
                last_ep_revenues['ev'].append(ev_r)
                last_ep_revenues['fcev'].append(fcev_r)
                last_ep_revenues['grid_sell'].append(grid_r)
                last_ep_revenues['grid_cost'].append(grid_c)

        agent.update()
        agent.step_scheduler()
        all_rewards.append(ep_reward)

        if (ep + 1) % 20 == 0:
            print(f"    {algo_name:<5s} Episode {ep+1:>3d}/{num_episodes}, "
                  f"Reward: {ep_reward:>8.2f}, SOC: {state[0]:.2f}")

    return all_rewards, last_ep_profits, last_ep_revenues


def train_off_policy(algo_name, agent, num_episodes=NUM_EPISODES,
                     warmup_steps=WARMUP_STEPS, batch_size=BATCH_SIZE):
    """训练 Off-Policy 算法 (SAC / TD3 / DDPG)"""
    env = HydrogenEnv()
    replay_buffer = ReplayBuffer(capacity=100000)

    all_rewards = []
    last_ep_profits = []
    last_ep_revenues = {'ev': [], 'fcev': [], 'grid_sell': [], 'grid_cost': []}
    total_steps = 0

    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0
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
            total_steps += 1

            if ep == num_episodes - 1:
                profit, ev_r, fcev_r, grid_r, grid_c = _collect_step_data(info)
                last_ep_profits.append(profit)
                last_ep_revenues['ev'].append(ev_r)
                last_ep_revenues['fcev'].append(fcev_r)
                last_ep_revenues['grid_sell'].append(grid_r)
                last_ep_revenues['grid_cost'].append(grid_c)

        all_rewards.append(ep_reward)

        if (ep + 1) % 20 == 0:
            print(f"    {algo_name:<5s} Episode {ep+1:>3d}/{num_episodes}, "
                  f"Reward: {ep_reward:>8.2f}, SOC: {state[0]:.2f}")

    return all_rewards, last_ep_profits, last_ep_revenues


# ======================== 可视化 ========================

def moving_average(data, window):
    """计算移动平均"""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_comparison(results):
    """
    绘制五算法对比图 (1×3 布局)

    results: dict  {algo_name: (rewards, profits, revenues)}
    """
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 11

    fig, axs = plt.subplots(1, 3, figsize=(32, 9))
    fig.suptitle('PPO vs A2C vs SAC vs TD3 vs DDPG — Algorithm Comparison',
                 fontsize=18, fontweight='bold', y=1.02)

    # ====== 图1: 训练奖励曲线 (原始 + 移动平均) ======
    ax = axs[0]
    for name in ALGO_NAMES:
        rewards = results[name][0]
        color = COLORS[name]
        ax.plot(rewards, alpha=0.12, color=color, linewidth=1)
        ma = moving_average(rewards, MA_WINDOW)
        ax.plot(range(MA_WINDOW - 1, len(rewards)), ma,
                color=color, linewidth=2.5, label=f'{name} (MA{MA_WINDOW})')
    ax.set_title('Training Reward Curves', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')

    # ====== 图2: 最后 Episode 累计利润曲线 ======
    ax = axs[1]
    for name in ALGO_NAMES:
        profits = results[name][1]
        if len(profits) > 0:
            cum_profit = np.cumsum(profits)
            color = COLORS[name]
            ax.plot(cum_profit, color=color, linewidth=2.5, label=name)
    ax.set_title('Cumulative Profit (Last Episode)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Step (15min)')
    ax.set_ylabel('Cumulative Profit ($)')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=0.8)

    # ====== 图3: 最终性能对比柱状图 ======
    ax = axs[2]

    metrics = {}
    for name in ALGO_NAMES:
        rewards = results[name][0]
        profits = results[name][1]

        metrics[name] = {
            'avg_reward': np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards),
            'best_reward': np.max(rewards),
            'total_profit': np.sum(profits) if len(profits) > 0 else 0,
        }

    metric_labels = ['Avg Reward\n(Last 20 Ep)', 'Best Reward', 'Total Profit\n(Last Ep, $)']
    metric_keys = ['avg_reward', 'best_reward', 'total_profit']
    x = np.arange(len(metric_labels))
    n = len(ALGO_NAMES)
    bar_width = 0.8 / n  # 自适应宽度

    for i, name in enumerate(ALGO_NAMES):
        values = [metrics[name][k] for k in metric_keys]
        bars = ax.bar(x + i * bar_width - 0.4 + bar_width / 2, values, bar_width,
                      color=COLORS[name], alpha=0.85, label=name, edgecolor='white')
        for bar, val in zip(bars, values):
            va = 'bottom' if val >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.0f}', ha='center', va=va, fontsize=8, fontweight='bold')

    ax.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.axhline(y=0, color='black', linewidth=0.8)

    plt.tight_layout()
    plt.show()


def print_summary(results):
    """打印性能统计表"""
    col_w = 12
    header = f"{'Metric':<30s}" + "".join(f"{n:>{col_w}s}" for n in ALGO_NAMES)
    sep_len = 30 + col_w * len(ALGO_NAMES)

    print("\n" + "=" * sep_len)
    print("                   ALGORITHM COMPARISON SUMMARY")
    print("=" * sep_len)
    print(header)
    print("-" * sep_len)

    metrics = {}
    for name in ALGO_NAMES:
        rewards = results[name][0]
        profits = results[name][1]
        revenues = results[name][2]

        metrics[name] = {
            'avg_reward_20': np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards),
            'std_reward_20': np.std(rewards[-20:]) if len(rewards) >= 20 else np.std(rewards),
            'best_reward': np.max(rewards),
            'worst_reward': np.min(rewards),
            'total_profit': np.sum(profits),
            'ev_rev': np.sum(revenues['ev']),
            'fcev_rev': np.sum(revenues['fcev']),
            'grid_sell': np.sum(revenues['grid_sell']),
            'grid_cost': np.sum(revenues['grid_cost']),
        }

    def row(label, key, fmt='.2f'):
        line = f"{label:<30s}"
        for name in ALGO_NAMES:
            v = metrics[name][key]
            line += f"{v:{col_w}{fmt}}"
        print(line)

    row('Avg Reward (Last 20 Ep)',   'avg_reward_20')
    row('Std Reward (Last 20 Ep)',   'std_reward_20')
    row('Best Episode Reward',       'best_reward')
    row('Worst Episode Reward',      'worst_reward')
    print("-" * sep_len)
    row('Total Profit (Last Ep, $)', 'total_profit')
    row('EV Revenue ($)',            'ev_rev')
    row('FCEV Revenue ($)',          'fcev_rev')
    row('Grid Sell Revenue ($)',     'grid_sell')
    row('Grid Cost ($)',             'grid_cost')
    print("=" * sep_len)

    best_algo = max(ALGO_NAMES, key=lambda n: metrics[n]['avg_reward_20'])
    print(f"\n  >>> Best Algorithm (by Avg Reward): {best_algo} "
          f"({metrics[best_algo]['avg_reward_20']:.2f})")

    best_profit = max(ALGO_NAMES, key=lambda n: metrics[n]['total_profit'])
    print(f"  >>> Best Algorithm (by Profit):     {best_profit} "
          f"(${metrics[best_profit]['total_profit']:.2f})")
    print()


# ======================== 主函数 ========================

def main():
    print("=" * 70)
    print("  RL Algorithm Comparison: PPO vs A2C vs SAC vs TD3 vs DDPG")
    print("=" * 70)
    print(f"  On-Policy:  PPO, A2C")
    print(f"  Off-Policy: SAC, TD3, DDPG")
    print(f"  Episodes per Algorithm: {NUM_EPISODES}")
    print(f"  Off-Policy Warmup:      {WARMUP_STEPS} steps")
    print(f"  Learning Rate:          {LR}")
    print("=" * 70)

    results = {}
    times = {}

    # 获取环境维度
    env_tmp = HydrogenEnv()
    state_dim = env_tmp.observation_space.shape[0]
    action_dim = env_tmp.action_space.shape[0]
    del env_tmp

    # --- 1. PPO (On-Policy) ---
    print(f"\n[1/5] Training PPO ({NUM_EPISODES} episodes)...")
    agent = PPO(state_dim, action_dim, lr=LR)
    t0 = time.time()
    r, p, v = train_on_policy('PPO', agent, NUM_EPISODES)
    times['PPO'] = time.time() - t0
    results['PPO'] = (r, p, v)
    print(f"  PPO  done in {times['PPO']:.1f}s, Final MA: {np.mean(r[-20:]):.2f}")

    # --- 2. A2C (On-Policy) ---
    print(f"\n[2/5] Training A2C ({NUM_EPISODES} episodes)...")
    agent = A2C(state_dim, action_dim, lr=LR)
    t0 = time.time()
    r, p, v = train_on_policy('A2C', agent, NUM_EPISODES)
    times['A2C'] = time.time() - t0
    results['A2C'] = (r, p, v)
    print(f"  A2C  done in {times['A2C']:.1f}s, Final MA: {np.mean(r[-20:]):.2f}")

    # --- 3. SAC (Off-Policy) ---
    print(f"\n[3/5] Training SAC ({NUM_EPISODES} episodes)...")
    agent = SAC(state_dim, action_dim, lr=LR)
    t0 = time.time()
    r, p, v = train_off_policy('SAC', agent, NUM_EPISODES)
    times['SAC'] = time.time() - t0
    results['SAC'] = (r, p, v)
    print(f"  SAC  done in {times['SAC']:.1f}s, Final MA: {np.mean(r[-20:]):.2f}")

    # --- 4. TD3 (Off-Policy) ---
    print(f"\n[4/5] Training TD3 ({NUM_EPISODES} episodes)...")
    agent = TD3(state_dim, action_dim, lr=LR)
    t0 = time.time()
    r, p, v = train_off_policy('TD3', agent, NUM_EPISODES)
    times['TD3'] = time.time() - t0
    results['TD3'] = (r, p, v)
    print(f"  TD3  done in {times['TD3']:.1f}s, Final MA: {np.mean(r[-20:]):.2f}")

    # --- 5. DDPG (Off-Policy) ---
    print(f"\n[5/5] Training DDPG ({NUM_EPISODES} episodes)...")
    agent = DDPG(state_dim, action_dim, lr=LR)
    t0 = time.time()
    r, p, v = train_off_policy('DDPG', agent, NUM_EPISODES)
    times['DDPG'] = time.time() - t0
    results['DDPG'] = (r, p, v)
    print(f"  DDPG done in {times['DDPG']:.1f}s, Final MA: {np.mean(r[-20:]):.2f}")

    # --- 训练时间 ---
    time_str = " | ".join(f"{n}={times[n]:.1f}s" for n in ALGO_NAMES)
    print(f"\n  Training Time: {time_str}")

    # --- 统计表 ---
    print_summary(results)

    # --- 可视化 ---
    print("Generating comparison plots...")
    plot_comparison(results)


if __name__ == "__main__":
    main()
