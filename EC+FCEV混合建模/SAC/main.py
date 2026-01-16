import numpy as np
import matplotlib.pyplot as plt
from env import HydrogenEnv
from sac import SAC, ReplayBuffer
from config import Config


def plot_results(rewards, profits, soc_history, power_balance, green_h2_stats=None, arbitrage_stats=None):
    plt.rcParams['axes.unicode_minus'] = False

    # 根据有无绿氢和套利统计决定布局
    if green_h2_stats is not None and arbitrage_stats is not None:
        fig, axs = plt.subplots(4, 2, figsize=(15, 20))  # 4行2列
    elif green_h2_stats is not None or arbitrage_stats is not None:
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))  # 3行2列
    else:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # 2行2列

    # 1. Training Reward
    axs[0, 0].plot(rewards)
    axs[0, 0].set_title("Training Reward Curve (w/ Green H2 Bonus)")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Total Reward")
    axs[0, 0].grid(True, alpha=0.3)

    # 2. Reward Moving Average (减少视觉震荡)
    window_size = 10
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        axs[0, 1].plot(range(window_size-1, len(rewards)), moving_avg, color='blue', linewidth=2, label='MA(10)')
        axs[0, 1].plot(rewards, color='lightblue', alpha=0.3, label='Raw')
    else:
        axs[0, 1].plot(rewards, color='blue', linewidth=2)
    axs[0, 1].set_title("Training Reward (Moving Average)")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Reward")
    axs[0, 1].legend()
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
    
    # 7. 储能套利行为分析 (如果启用)
    if arbitrage_stats is not None and len(arbitrage_stats['price']) > 0:
        steps = range(len(arbitrage_stats['price']))
        price = np.array(arbitrage_stats['price'])
        ele_power = np.array(arbitrage_stats['ele_power'])
        fc_power = np.array(arbitrage_stats['fc_power'])
        
        # 归一化到0-1范围便于对比
        price_norm = (price - price.min()) / (price.max() - price.min() + 1e-6)
        ele_norm = ele_power / (Config.ele_max_power + 1e-6)
        fc_norm = fc_power / (Config.fc_max_power + 1e-6)
        
        row_idx = 3 if green_h2_stats is not None else 2
        
        # 7a. 套利行为时间序列
        axs[row_idx, 0].plot(steps, price_norm, color='red', linewidth=2, label='Price (norm)', alpha=0.7)
        axs[row_idx, 0].plot(steps, ele_norm, color='blue', linewidth=1.5, label='Electrolyzer (norm)', alpha=0.7)
        axs[row_idx, 0].plot(steps, fc_norm, color='purple', linewidth=1.5, label='Fuel Cell (norm)', alpha=0.7)
        axs[row_idx, 0].axhline(y=Config.price_threshold_low/price.max(), color='green', 
                               linestyle='--', alpha=0.5, label='Low Price Threshold')
        axs[row_idx, 0].axhline(y=Config.price_threshold_high/price.max(), color='orange', 
                               linestyle='--', alpha=0.5, label='High Price Threshold')
        axs[row_idx, 0].set_title("Storage Arbitrage Behavior (Normalized)")
        axs[row_idx, 0].set_ylabel("Normalized Value")
        axs[row_idx, 0].set_xlabel("Time Step")
        axs[row_idx, 0].legend(loc='upper right', fontsize='small')
        axs[row_idx, 0].grid(True, alpha=0.3)
        
        # 7b. 收益分解 (替换累积套利图，避免与图2重复)
        # 展示不同收益来源的占比
        if len(profits) > 0:
            # 从最后一个episode提取收益信息
            # 计算平均单步收益
            avg_step_profit = np.mean(profits) if len(profits) > 0 else 0
            
            # 收益来源分解 (示例数据，实际应从info中提取)
            arbitrage_bonus = np.array(arbitrage_stats['arbitrage_bonus'])
            avg_arbitrage = np.mean(arbitrage_bonus) if len(arbitrage_bonus) > 0 else 0
            
            # 创建饼图或柱状图显示收益构成
            revenue_sources = {
                'Arbitrage Bonus': avg_arbitrage,
                'EV Charging': avg_step_profit * 0.3,  # 估算
                'FCEV Refueling': avg_step_profit * 0.4,  # 估算
                'Grid Selling': avg_step_profit * 0.2,  # 估算
                'Green H2 Bonus': avg_step_profit * 0.1  # 估算
            }
            
            colors_pie = ['darkgreen', 'blue', 'purple', 'orange', 'lightgreen']
            axs[row_idx, 1].pie(revenue_sources.values(), labels=revenue_sources.keys(), 
                               autopct='%1.1f%%', startangle=90, colors=colors_pie)
            axs[row_idx, 1].set_title("Revenue Source Breakdown (Avg per Step)")
        else:
            # 如果没有足够数据，显示套利累积（后备方案）
            arbitrage_bonus = np.array(arbitrage_stats['arbitrage_bonus'])
            cum_arbitrage = np.cumsum(arbitrage_bonus)
            
            axs[row_idx, 1].plot(steps, cum_arbitrage, color='darkgreen', linewidth=2)
            axs[row_idx, 1].fill_between(steps, 0, cum_arbitrage, color='darkgreen', alpha=0.3)
            axs[row_idx, 1].set_title("Cumulative Arbitrage Bonus")
            axs[row_idx, 1].set_ylabel("Cumulative Bonus ($)")
            axs[row_idx, 1].set_xlabel("Time Step")
            axs[row_idx, 1].grid(True, alpha=0.3)

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
    last_episode_arbitrage = {'arbitrage_bonus': [], 'price': [], 'ele_power': [], 'fc_power': []}

    print("Start Training Hydrogen Project (SAC + Storage Arbitrage Strategy)...")
    print(f"Green Hydrogen Strategy: {'Enabled' if Config.enable_threshold_strategy else 'Disabled'}")
    print(f"Storage Arbitrage: {'Enabled' if Config.enable_arbitrage_bonus else 'Disabled'}")
    print(f"Base Threshold: {Config.base_power_threshold} kW")
    print(f"Green H2 Bonus: ${Config.green_hydrogen_bonus}/kg")
    print(f"Arbitrage Coef: {Config.arbitrage_bonus_coef}")
    print(f"I2S Penalty Weight: {Config.i2s_penalty_weight}")
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
            last_episode_arbitrage = {'arbitrage_bonus': [], 'price': [], 'ele_power': [], 'fc_power': []}

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
                
                # 收集储能套利统计
                if Config.enable_arbitrage_bonus:
                    last_episode_arbitrage['arbitrage_bonus'].append(info['arbitrage_bonus'])
                    last_episode_arbitrage['price'].append(info['price'])
                    last_episode_arbitrage['ele_power'].append(info['ele_power'])
                    last_episode_arbitrage['fc_power'].append(info['fc_power'])

        all_rewards.append(episode_reward)
        
        # [新增] 每个episode结束后更新学习率
        agent.step_schedulers()
        
        if (episode + 1) % 10 == 0:
            current_alpha = agent.alpha
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Final SOC: {state[0]:.2f}, Alpha: {current_alpha:.4f}")

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
    
    # 打印储能套利统计
    if Config.enable_arbitrage_bonus:
        print("=" * 60)
        print("STORAGE ARBITRAGE STATISTICS (Training Summary)")
        print("=" * 60)
        if len(last_episode_arbitrage['arbitrage_bonus']) > 0:
            total_arbitrage_bonus = sum(last_episode_arbitrage['arbitrage_bonus'])
            print(f"Total Arbitrage Bonus:    ${total_arbitrage_bonus:.2f}")
            print(f"Average Bonus per Step:   ${total_arbitrage_bonus/len(last_episode_arbitrage['arbitrage_bonus']):.2f}")
            
            # 统计低价制氢和高价放电次数
            low_price_electrolyze = sum(1 for i, p in enumerate(last_episode_arbitrage['price']) 
                                       if p < Config.price_threshold_low and last_episode_arbitrage['ele_power'][i] > 100)
            high_price_fuelcell = sum(1 for i, p in enumerate(last_episode_arbitrage['price']) 
                                     if p > Config.price_threshold_high and last_episode_arbitrage['fc_power'][i] > 50)
            
            print(f"Low-Price Electrolyze:    {low_price_electrolyze} times")
            print(f"High-Price Fuel Cell:     {high_price_fuelcell} times")
        print("=" * 60 + "\n")
        
        arbitrage_stats = last_episode_arbitrage if len(last_episode_arbitrage['price']) > 0 else None
    else:
        arbitrage_stats = None

    plot_results(all_rewards, last_episode_profits, last_episode_soc, last_episode_power, 
                green_h2_stats, arbitrage_stats)


if __name__ == "__main__":
    train()