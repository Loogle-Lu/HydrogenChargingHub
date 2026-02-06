"""
三算法对比脚本: PPO vs SAC vs TD3

功能:
1. 分别训练 PPO、SAC、TD3 三种强化学习算法
2. 收集每个 Episode 的 Reward 和最后一个 Episode 的详细收益数据
3. 生成对比可视化图表:
   - 训练奖励曲线对比
   - 移动平均奖励对比
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
from SAC import SAC, ReplayBuffer
from TD3 import TD3


# ======================== 配置 ========================
NUM_EPISODES = 200       # 每个算法训练的 Episode 数
WARMUP_STEPS = 500       # Off-Policy 算法的随机探索步数
BATCH_SIZE = 256         # Off-Policy 算法的 mini-batch 大小
LR = 3e-4                # 统一学习率
MA_WINDOW = 20           # 移动平均窗口大小

# 颜色方案
COLORS = {
    'PPO':  '#1f77b4',   # 蓝色
    'SAC':  '#ff7f0e',   # 橙色
    'TD3':  '#2ca02c',   # 绿色
}


# ======================== 训练函数 ========================

def train_ppo(num_episodes=NUM_EPISODES):
    """训练 PPO (On-Policy)"""
    env = HydrogenEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPO(state_dim, action_dim, lr=LR)

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

            # 最后一个 Episode 收集详细数据
            if ep == num_episodes - 1:
                # 使用真实经济收益 (不含 arbitrage shaping reward)
                ev_r = info.get('revenue_ev', 0)
                fcev_r = info.get('revenue_fcev', 0)
                grid_r = info.get('revenue_grid', 0)
                grid_c = info.get('cost_grid', 0)
                real_profit = ev_r + fcev_r + grid_r - grid_c
                last_ep_profits.append(real_profit)
                last_ep_revenues['ev'].append(ev_r)
                last_ep_revenues['fcev'].append(fcev_r)
                last_ep_revenues['grid_sell'].append(grid_r)
                last_ep_revenues['grid_cost'].append(grid_c)

        # PPO: Episode 结束后更新
        agent.update()
        agent.step_scheduler()
        all_rewards.append(ep_reward)

        if (ep + 1) % 20 == 0:
            print(f"    PPO  Episode {ep+1:>3d}/{num_episodes}, "
                  f"Reward: {ep_reward:>8.2f}, SOC: {state[0]:.2f}")

    return all_rewards, last_ep_profits, last_ep_revenues


def train_off_policy(algo_name, agent, num_episodes=NUM_EPISODES,
                     warmup_steps=WARMUP_STEPS, batch_size=BATCH_SIZE):
    """训练 Off-Policy 算法 (SAC / TD3)"""
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
            # Warmup: 随机探索
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, evaluate=False)

            next_state, reward, done, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, float(done))

            # Warmup 结束后每步更新
            if total_steps >= warmup_steps and len(replay_buffer) >= batch_size:
                agent.update(replay_buffer, batch_size)

            state = next_state
            ep_reward += reward
            total_steps += 1

            # 最后一个 Episode 收集详细数据
            if ep == num_episodes - 1:
                # 使用真实经济收益 (不含 arbitrage shaping reward)
                ev_r = info.get('revenue_ev', 0)
                fcev_r = info.get('revenue_fcev', 0)
                grid_r = info.get('revenue_grid', 0)
                grid_c = info.get('cost_grid', 0)
                real_profit = ev_r + fcev_r + grid_r - grid_c
                last_ep_profits.append(real_profit)
                last_ep_revenues['ev'].append(ev_r)
                last_ep_revenues['fcev'].append(fcev_r)
                last_ep_revenues['grid_sell'].append(grid_r)
                last_ep_revenues['grid_cost'].append(grid_c)

        all_rewards.append(ep_reward)

        if (ep + 1) % 20 == 0:
            print(f"    {algo_name:<4s} Episode {ep+1:>3d}/{num_episodes}, "
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
    绘制三算法对比图 (1×3 布局)

    results: dict  {algo_name: (rewards, profits, revenues)}
    """
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 11

    fig, axs = plt.subplots(1, 3, figsize=(30, 8))
    fig.suptitle('PPO vs SAC vs TD3 — Algorithm Comparison',
                 fontsize=18, fontweight='bold', y=1.02)

    algo_names = ['PPO', 'SAC', 'TD3']

    # ====== 图1: 训练奖励曲线 (原始 + 移动平均) ======
    ax = axs[0]
    for name in algo_names:
        rewards = results[name][0]
        color = COLORS[name]
        ax.plot(rewards, alpha=0.15, color=color, linewidth=1)
        ma = moving_average(rewards, MA_WINDOW)
        ax.plot(range(MA_WINDOW - 1, len(rewards)), ma,
                color=color, linewidth=2.5, label=f'{name} (MA{MA_WINDOW})')
    ax.set_title('Training Reward Curves', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')

    # ====== 图2: 最后 Episode 累计利润曲线 ======
    ax = axs[1]
    for name in algo_names:
        profits = results[name][1]
        if len(profits) > 0:
            cum_profit = np.cumsum(profits)
            color = COLORS[name]
            ax.plot(cum_profit, color=color, linewidth=2.5, label=name)
    ax.set_title('Cumulative Profit (Last Episode)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Step (15min)')
    ax.set_ylabel('Cumulative Profit ($)')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=0.8)

    # ====== 图3: 最终性能对比柱状图 ======
    ax = axs[2]

    # 收集指标
    metrics = {}
    for name in algo_names:
        rewards = results[name][0]
        profits = results[name][1]
        revenues = results[name][2]

        avg_reward_last20 = np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards)
        best_reward = np.max(rewards)
        total_profit = np.sum(profits) if len(profits) > 0 else 0
        total_ev_rev = np.sum(revenues['ev']) if len(revenues['ev']) > 0 else 0
        total_fcev_rev = np.sum(revenues['fcev']) if len(revenues['fcev']) > 0 else 0
        total_cost = np.sum(revenues['grid_cost']) if len(revenues['grid_cost']) > 0 else 0

        metrics[name] = {
            'avg_reward': avg_reward_last20,
            'best_reward': best_reward,
            'total_profit': total_profit,
            'ev_revenue': total_ev_rev,
            'fcev_revenue': total_fcev_rev,
            'grid_cost': total_cost,
        }

    # 绘制分组柱状图
    metric_labels = ['Avg Reward\n(Last 20 Ep)', 'Best Reward', 'Total Profit\n(Last Ep, $)']
    metric_keys = ['avg_reward', 'best_reward', 'total_profit']
    x = np.arange(len(metric_labels))
    bar_width = 0.25

    for i, name in enumerate(algo_names):
        values = [metrics[name][k] for k in metric_keys]
        bars = ax.bar(x + i * bar_width, values, bar_width,
                      color=COLORS[name], alpha=0.85, label=name, edgecolor='white')
        # 柱顶数值标注
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.axhline(y=0, color='black', linewidth=0.8)

    plt.tight_layout()
    plt.show()


def print_summary(results):
    """打印性能统计表"""
    print("\n" + "=" * 80)
    print("                     ALGORITHM COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<30s} {'PPO':>14s} {'SAC':>14s} {'TD3':>14s}")
    print("-" * 80)

    # 使用独立字典，不污染 results
    metrics = {}
    for name in ['PPO', 'SAC', 'TD3']:
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
        vals = []
        for name in ['PPO', 'SAC', 'TD3']:
            v = metrics[name][key]
            vals.append(f'{v:{fmt}}')
        print(f"{label:<30s} {vals[0]:>14s} {vals[1]:>14s} {vals[2]:>14s}")

    row('Avg Reward (Last 20 Ep)',   'avg_reward_20')
    row('Std Reward (Last 20 Ep)',   'std_reward_20')
    row('Best Episode Reward',       'best_reward')
    row('Worst Episode Reward',      'worst_reward')
    print("-" * 80)
    row('Total Profit (Last Ep, $)', 'total_profit')
    row('EV Revenue ($)',            'ev_rev')
    row('FCEV Revenue ($)',          'fcev_rev')
    row('Grid Sell Revenue ($)',     'grid_sell')
    row('Grid Cost ($)',             'grid_cost')
    print("=" * 80)

    # 判断最优算法
    best_algo = max(['PPO', 'SAC', 'TD3'],
                    key=lambda n: metrics[n]['avg_reward_20'])
    print(f"\n  >>> Best Algorithm (by Avg Reward): {best_algo} "
          f"({metrics[best_algo]['avg_reward_20']:.2f})")

    best_profit = max(['PPO', 'SAC', 'TD3'],
                      key=lambda n: metrics[n]['total_profit'])
    print(f"  >>> Best Algorithm (by Profit):     {best_profit} "
          f"(${metrics[best_profit]['total_profit']:.2f})")
    print()


# ======================== 主函数 ========================

def main():
    print("=" * 70)
    print("       RL Algorithm Comparison: PPO vs SAC vs TD3")
    print("=" * 70)
    print(f"  Episodes per Algorithm: {NUM_EPISODES}")
    print(f"  Off-Policy Warmup:      {WARMUP_STEPS} steps")
    print(f"  Learning Rate:          {LR}")
    print(f"  Moving Average Window:  {MA_WINDOW}")
    print("=" * 70)

    results = {}

    # --- 1. 训练 PPO ---
    print(f"\n[1/3] Training PPO ({NUM_EPISODES} episodes)...")
    t0 = time.time()
    ppo_rewards, ppo_profits, ppo_revenues = train_ppo(NUM_EPISODES)
    ppo_time = time.time() - t0
    results['PPO'] = (ppo_rewards, ppo_profits, ppo_revenues)
    print(f"  PPO done in {ppo_time:.1f}s, Final MA Reward: "
          f"{np.mean(ppo_rewards[-20:]):.2f}")

    # --- 2. 训练 SAC ---
    print(f"\n[2/3] Training SAC ({NUM_EPISODES} episodes)...")
    env_tmp = HydrogenEnv()
    state_dim = env_tmp.observation_space.shape[0]
    action_dim = env_tmp.action_space.shape[0]
    del env_tmp

    sac_agent = SAC(state_dim, action_dim, lr=LR)
    t0 = time.time()
    sac_rewards, sac_profits, sac_revenues = train_off_policy(
        'SAC', sac_agent, NUM_EPISODES)
    sac_time = time.time() - t0
    results['SAC'] = (sac_rewards, sac_profits, sac_revenues)
    print(f"  SAC done in {sac_time:.1f}s, Final MA Reward: "
          f"{np.mean(sac_rewards[-20:]):.2f}")

    # --- 3. 训练 TD3 ---
    print(f"\n[3/3] Training TD3 ({NUM_EPISODES} episodes)...")
    td3_agent = TD3(state_dim, action_dim, lr=LR)
    t0 = time.time()
    td3_rewards, td3_profits, td3_revenues = train_off_policy(
        'TD3', td3_agent, NUM_EPISODES)
    td3_time = time.time() - t0
    results['TD3'] = (td3_rewards, td3_profits, td3_revenues)
    print(f"  TD3 done in {td3_time:.1f}s, Final MA Reward: "
          f"{np.mean(td3_rewards[-20:]):.2f}")

    # --- 训练时间对比 ---
    print(f"\n  Training Time: PPO={ppo_time:.1f}s | SAC={sac_time:.1f}s | TD3={td3_time:.1f}s")

    # --- 统计表 ---
    print_summary(results)

    # --- 可视化 ---
    print("Generating comparison plots...")
    plot_comparison(results)


if __name__ == "__main__":
    main()
