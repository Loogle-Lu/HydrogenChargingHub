import numpy as np
import matplotlib.pyplot as plt
from env import HydrogenEnv
from sac import SAC, ReplayBuffer
from config import Config


def plot_results(rewards, profits, soc_history, power_balance, green_h2_stats=None):
    plt.rcParams['axes.unicode_minus'] = False

    # 如果有绿氢统计，创建3x2布局，否则保持2x2
    if green_h2_stats is not None:
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    else:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Training Reward
    axs[0, 0].plot(rewards)
    axs[0, 0].set_title("Training Reward Curve (w/ Green H2 Bonus)")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Total Reward")
    axs[0, 0].grid(True, alpha=0.3)

    # 2. Cumulative Profit
    cum_profit = np.cumsum(profits)
    axs[0, 1].plot(cum_profit, color='green', linewidth=2)
    axs[0, 1].set_title("Cumulative Profit (Last Episode)")
    axs[0, 1].set_ylabel("Profit ($)")
    axs[0, 1].grid(True, alpha=0.3)

    # 3. SOC History
    axs[1, 0].plot(soc_history, color='orange', label='SOC', linewidth=2)
    axs[1, 0].axhline(y=Config.storage_initial, color='r', linestyle='--', label='Initial Target')
    axs[1, 0].set_title("H2 Storage Level (Should Match Target)")
    axs[1, 0].set_ylim(0, 1.0)
    axs[1, 0].set_ylabel("SOC")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # 4. Energy Balance
    steps = range(len(power_balance['re']))
    re_gen = np.array(power_balance['re'])
    grid_power = np.array(power_balance['net'])
    fc_power = np.array(power_balance['fc'])
    load_power = np.array(power_balance['load'])

    axs[1, 1].plot(steps, re_gen, label='RE Gen (+)', alpha=0.7, linewidth=1.5)
    axs[1, 1].plot(steps, fc_power, label='Fuel Cell (+)', color='purple', linewidth=2)
    axs[1, 1].plot(steps, -load_power, label='Total Load (-)', color='red', alpha=0.5)

    axs[1, 1].fill_between(steps, grid_power, 0, where=(grid_power < 0), color='gray', alpha=0.3,
                           label='Grid Import (-)')
    axs[1, 1].fill_between(steps, grid_power, 0, where=(grid_power > 0), color='green', alpha=0.3,
                           label='Grid Export (+)')

    axs[1, 1].set_title("Energy Balance")
    axs[1, 1].set_ylabel("Power (kW)")
    axs[1, 1].legend(loc='upper right', fontsize='small')
    axs[1, 1].grid(True, alpha=0.3)

    # 5. 绿氢生产统计 (如果启用了阈值策略)
    if green_h2_stats is not None:
        # 绿氢占比时间序列
        steps = range(len(green_h2_stats['green_ratio']))
        green_ratio = np.array(green_h2_stats['green_ratio']) * 100
        threshold = np.array(green_h2_stats['threshold'])
        
        axs[2, 0].plot(steps, green_ratio, color='green', linewidth=2, label='Green H2 Ratio')
        axs[2, 0].set_title("Green Hydrogen Production Ratio")
        axs[2, 0].set_ylabel("Green H2 %")
        axs[2, 0].set_ylim(0, 105)
        axs[2, 0].grid(True, alpha=0.3)
        
        # 在右侧y轴显示阈值
        ax2 = axs[2, 0].twinx()
        ax2.plot(steps, threshold, color='orange', linestyle='--', alpha=0.7, label='Threshold')
        ax2.set_ylabel("Power Threshold (kW)", color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # 合并图例
        lines1, labels1 = axs[2, 0].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axs[2, 0].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # 6. 电力来源分解
        power_from_re = np.array(green_h2_stats['power_from_re'])
        power_from_grid = np.array(green_h2_stats['power_from_grid'])
        
        axs[2, 1].fill_between(steps, 0, power_from_re, color='green', alpha=0.6, label='RE Power')
        axs[2, 1].fill_between(steps, power_from_re, power_from_re + power_from_grid, 
                              color='gray', alpha=0.6, label='Grid Power')
        axs[2, 1].set_title("Electrolyzer Power Source Breakdown")
        axs[2, 1].set_ylabel("Power (kW)")
        axs[2, 1].set_xlabel("Time Step")
        axs[2, 1].legend()
        axs[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    print("Plotting results...")
    plt.show()


def train():
    env = HydrogenEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SAC(state_dim, action_dim)
    replay_buffer = ReplayBuffer(capacity=100000, state_dim=state_dim, action_dim=action_dim)

    num_episodes = 200
    batch_size = 64
    all_rewards = []

    last_episode_profits = []
    last_episode_soc = []
    last_episode_power = {'re': [], 'net': [], 'fc': [], 'load': []}
    last_episode_green_h2 = {'green_ratio': [], 'threshold': [], 'power_from_re': [], 'power_from_grid': []}

    print("Start Training Hydrogen Project (SAC + Variable Power Threshold Strategy)...")
    print(f"Green Hydrogen Strategy: {'Enabled' if Config.enable_threshold_strategy else 'Disabled'}")
    print(f"Base Threshold: {Config.base_power_threshold} kW")
    print(f"Green H2 Bonus: ${Config.green_hydrogen_bonus}/kg")
    print("-" * 60)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        if episode == num_episodes - 1:
            last_episode_profits = []
            last_episode_soc = [state[0]]
            last_episode_power = {'re': [], 'net': [], 'fc': [], 'load': []}
            last_episode_green_h2 = {'green_ratio': [], 'threshold': [], 'power_from_re': [], 'power_from_grid': []}

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if replay_buffer.size > batch_size:
                agent.update(replay_buffer, batch_size)

            if episode == num_episodes - 1:
                last_episode_profits.append(info['profit'])
                last_episode_soc.append(info['soc'])
                last_episode_power['re'].append(info['re_gen'])
                last_episode_power['net'].append(info['net_power'])
                last_episode_power['fc'].append(info['fc_power'])
                last_episode_power['load'].append(info['load_power'])
                
                # 收集绿氢统计
                if Config.enable_threshold_strategy:
                    last_episode_green_h2['green_ratio'].append(info['green_h2_ratio'])
                    last_episode_green_h2['threshold'].append(info['power_threshold'])
                    last_episode_green_h2['power_from_re'].append(info['power_from_re'])
                    last_episode_green_h2['power_from_grid'].append(info['power_from_grid'])

        all_rewards.append(episode_reward)
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Final SOC: {state[0]:.2f}")

    # 打印绿氢生产统计
    if Config.enable_threshold_strategy:
        ele_stats = env.ele.get_statistics()
        print("\n" + "=" * 60)
        print("GREEN HYDROGEN PRODUCTION STATISTICS (Training Summary)")
        print("=" * 60)
        print(f"Total Green H2 Produced:  {ele_stats['total_green_h2_kg']:.2f} kg")
        print(f"Total Grid H2 Produced:   {ele_stats['total_grid_h2_kg']:.2f} kg")
        print(f"Total H2 Produced:        {ele_stats['total_h2_kg']:.2f} kg")
        print(f"Green H2 Percentage:      {ele_stats['green_h2_percentage']:.1f}%")
        print(f"Total Green Energy Used:  {ele_stats['total_green_energy_kwh']:.2f} kWh")
        print(f"Total Grid Energy Used:   {ele_stats['total_grid_energy_kwh']:.2f} kWh")
        print("=" * 60 + "\n")
        
        green_h2_stats = last_episode_green_h2 if len(last_episode_green_h2['green_ratio']) > 0 else None
    else:
        green_h2_stats = None

    plot_results(all_rewards, last_episode_profits, last_episode_soc, last_episode_power, green_h2_stats)


if __name__ == "__main__":
    train()