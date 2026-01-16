import numpy as np
import matplotlib.pyplot as plt
from env import HydrogenEnv
from TD3 import TD3, ReplayBuffer
from config import Config


def plot_results(rewards, profits, soc_history, power_balance, green_h2_stats=None, arbitrage_stats=None, battery_stats=None):
    # 设置中文显示和负号显示 (优化字体大小避免拥挤)
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9

    # 固定3行2列布局 (最多6张图) - 增加高度改善垂直间距
    fig, axs = plt.subplots(3, 2, figsize=(28, 38))

    # 图1: Training Reward (原始 + 移动平均合并)
    axs[0, 0].plot(rewards, linewidth=2, color='lightblue', alpha=0.5, label='Raw Reward')
    window_size = 10
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        axs[0, 0].plot(range(window_size-1, len(rewards)), moving_avg, color='#2ca02c', linewidth=3.5, label='MA(10)')
    axs[0, 0].set_title("Training Reward Curve", fontsize=14, fontweight='bold')
    axs[0, 0].set_xlabel("Episode", fontsize=11)
    axs[0, 0].set_ylabel("Total Reward", fontsize=11)
    axs[0, 0].legend(fontsize=10, loc='best')
    axs[0, 0].grid(True, alpha=0.3, linestyle='--', linewidth=1.0)

    # 图2: 双储能SOC对比 (氢气 vs 电池)
    if battery_stats is not None and len(battery_stats['soc']) > 0:
        battery_soc_arr = np.array(battery_stats['soc']) * 100
        h2_soc_arr = np.array(soc_history) * 100
        
        # 确保两个数组长度一致，取最小长度
        min_len = min(len(battery_soc_arr), len(h2_soc_arr))
        battery_soc_arr = battery_soc_arr[:min_len]
        h2_soc_arr = h2_soc_arr[:min_len]
        steps_soc = range(min_len)
        
        axs[0, 1].plot(steps_soc, h2_soc_arr, color='#1f77b4', label='H2 Storage', linewidth=2.5)
        axs[0, 1].plot(steps_soc, battery_soc_arr, color='#ff6b6b', label='Battery', linewidth=2.5)
        axs[0, 1].axhline(y=Config.storage_initial*100, color='blue', linestyle='--', linewidth=1.5, alpha=0.5)
        axs[0, 1].axhline(y=Config.battery_initial_soc*100, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        axs[0, 1].set_title("Dual Storage SOC (H2 + Battery)", fontsize=14, fontweight='bold')
        axs[0, 1].set_ylabel("SOC (%)", fontsize=11)
        axs[0, 1].set_xlabel("Time Step", fontsize=11)
        axs[0, 1].set_ylim(0, 100)
        axs[0, 1].legend(fontsize=10, loc='best')
        axs[0, 1].grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
    else:
        # 如果没有电池，只显示氢气SOC
        axs[0, 1].plot(soc_history, color='#ff7f0e', label='H2 SOC', linewidth=2.5)
        axs[0, 1].axhline(y=Config.storage_initial, color='#d62728', linestyle='--', linewidth=2, label='Initial Target')
        axs[0, 1].set_title("H2 Storage Level", fontsize=14, fontweight='bold')
        axs[0, 1].set_ylim(0, 1.0)
        axs[0, 1].set_ylabel("SOC", fontsize=11)
        axs[0, 1].legend(fontsize=10, loc='best')
        axs[0, 1].grid(True, alpha=0.3, linestyle='--', linewidth=1.0)

    # 图3: Energy Balance (含电池)
    steps = range(len(power_balance['re']))
    re_gen = np.array(power_balance['re'])
    grid_power = np.array(power_balance['net'])
    fc_power = np.array(power_balance['fc'])
    load_power = np.array(power_balance['load'])

    axs[1, 0].plot(steps, re_gen, label='RE Gen', alpha=0.8, linewidth=2.5, color='#2ca02c')
    axs[1, 0].plot(steps, fc_power, label='Fuel Cell', color='#9467bd', linewidth=2.5)
    axs[1, 0].plot(steps, -load_power, label='Total Load', color='#d62728', alpha=0.7, linewidth=2.5)
    
    if battery_stats is not None and len(battery_stats['soc']) > 0:
        battery_net = np.array(battery_stats['discharge_power']) - np.array(battery_stats['charge_power'])
        # 确保battery_net与其他数据长度一致
        min_len = min(len(battery_net), len(re_gen))
        battery_net = battery_net[:min_len]
        steps_battery = range(min_len)
        axs[1, 0].plot(steps_battery, battery_net, label='Battery Net', color='#ff6b6b', linewidth=2, linestyle='--')

    axs[1, 0].fill_between(steps, grid_power, 0, where=(grid_power < 0), color='gray', alpha=0.4, label='Grid Import')
    axs[1, 0].fill_between(steps, grid_power, 0, where=(grid_power > 0), color='green', alpha=0.3, label='Grid Export')

    axs[1, 0].set_title("Energy Balance (with Battery)", fontsize=14, fontweight='bold')
    axs[1, 0].set_ylabel("Power (kW)", fontsize=11)
    axs[1, 0].legend(loc='upper right', fontsize=9)
    axs[1, 0].grid(True, alpha=0.3, linestyle='--', linewidth=1.0)

    # 图4: 绿氢生产 (合并占比和来源到一张图)
    if green_h2_stats is not None and len(green_h2_stats['green_ratio']) > 0:
        steps = range(len(green_h2_stats['green_ratio']))
        power_from_re = np.array(green_h2_stats['power_from_re'])
        power_from_grid = np.array(green_h2_stats['power_from_grid'])
        green_ratio = np.array(green_h2_stats['green_ratio']) * 100
        
        # 使用堆叠面积图显示功率来源，同时在右轴显示绿氢占比
        axs[1, 1].fill_between(steps, 0, power_from_re, color='green', alpha=0.6, label='RE Power')
        axs[1, 1].fill_between(steps, power_from_re, power_from_re + power_from_grid, 
                              color='gray', alpha=0.6, label='Grid Power')
        axs[1, 1].set_ylabel("Electrolyzer Power (kW)", fontsize=11)
        axs[1, 1].set_xlabel("Time Step", fontsize=11)
        axs[1, 1].legend(loc='upper left', fontsize=10)
        
        # 右轴显示绿氢占比
        ax_green = axs[1, 1].twinx()
        ax_green.plot(steps, green_ratio, color='darkgreen', linewidth=2.5, linestyle='--', label='Green H2 %')
        ax_green.set_ylabel("Green H2 Ratio (%)", color='darkgreen', fontsize=11)
        ax_green.tick_params(axis='y', labelcolor='darkgreen', labelsize=9)
        ax_green.set_ylim(0, 105)
        ax_green.legend(loc='upper right', fontsize=10)
        
        axs[1, 1].set_title("Green Hydrogen Production", fontsize=14, fontweight='bold')
        axs[1, 1].grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
    else:
        # 没有绿氢统计，显示空白
        axs[1, 1].text(0.5, 0.5, 'Green H2 Strategy Disabled', 
                      ha='center', va='center', fontsize=12, transform=axs[1, 1].transAxes)
        axs[1, 1].set_title("Green Hydrogen Production", fontsize=14, fontweight='bold')
    
    # 图5: 储能套利行为 (价格vs功率)
    if arbitrage_stats is not None and len(arbitrage_stats['price']) > 0:
        steps = range(len(arbitrage_stats['price']))
        price = np.array(arbitrage_stats['price'])
        ele_power = np.array(arbitrage_stats['ele_power'])
        fc_power = np.array(arbitrage_stats['fc_power'])
        
        # 归一化
        price_norm = (price - price.min()) / (price.max() - price.min() + 1e-6)
        ele_norm = ele_power / (Config.ele_max_power + 1e-6)
        fc_norm = fc_power / (Config.fc_max_power + 1e-6)
        
        axs[2, 0].plot(steps, price_norm, color='red', linewidth=2.5, label='Price', alpha=0.7)
        axs[2, 0].plot(steps, ele_norm, color='blue', linewidth=2, label='Electrolyzer', alpha=0.7)
        axs[2, 0].plot(steps, fc_norm, color='purple', linewidth=2, label='Fuel Cell', alpha=0.7)
        axs[2, 0].set_title("Storage Arbitrage Behavior", fontsize=14, fontweight='bold')
        axs[2, 0].set_ylabel("Normalized Value", fontsize=11)
        axs[2, 0].set_xlabel("Time Step", fontsize=11)
        axs[2, 0].legend(loc='best', fontsize=10)
        axs[2, 0].grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
    else:
        # 没有套利统计
        axs[2, 0].text(0.5, 0.5, 'Arbitrage Strategy Disabled', 
                      ha='center', va='center', fontsize=12, transform=axs[2, 0].transAxes)
        axs[2, 0].set_title("Storage Arbitrage Behavior", fontsize=14, fontweight='bold')
    
    # 图6: 电池充放电功率 (最重要的电池指标)
    if battery_stats is not None and len(battery_stats['soc']) > 0:
        steps = range(len(battery_stats['soc']))
        charge_power = np.array(battery_stats['charge_power'])
        discharge_power = np.array(battery_stats['discharge_power'])
        battery_soc = np.array(battery_stats['soc']) * 100
        
        # 主轴：充放电功率
        axs[2, 1].fill_between(steps, 0, charge_power, where=(charge_power > 0),
                              color='#4ecdc4', alpha=0.6, label='Charge')
        axs[2, 1].fill_between(steps, 0, -discharge_power, where=(discharge_power > 0),
                              color='#ff6b6b', alpha=0.6, label='Discharge')
        axs[2, 1].axhline(y=0, color='black', linestyle='-', linewidth=1.2)
        axs[2, 1].set_ylabel("Power (kW)", fontsize=11)
        axs[2, 1].set_xlabel("Time Step", fontsize=11)
        axs[2, 1].legend(loc='upper left', fontsize=10)
        
        # 右轴：电池SOC
        ax_batt_soc = axs[2, 1].twinx()
        ax_batt_soc.plot(steps, battery_soc, color='#ff6b6b', linewidth=2.5, linestyle='--', label='Battery SOC')
        ax_batt_soc.set_ylabel("Battery SOC (%)", color='#ff6b6b', fontsize=11)
        ax_batt_soc.tick_params(axis='y', labelcolor='#ff6b6b', labelsize=9)
        ax_batt_soc.set_ylim(0, 100)
        ax_batt_soc.legend(loc='upper right', fontsize=10)
        
        axs[2, 1].set_title("Battery Charge/Discharge + SOC", fontsize=14, fontweight='bold')
        axs[2, 1].grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
    else:
        # 没有电池统计
        axs[2, 1].text(0.5, 0.5, 'Battery Storage Disabled', 
                      ha='center', va='center', fontsize=12, transform=axs[2, 1].transAxes)
        axs[2, 1].set_title("Battery Charge/Discharge + SOC", fontsize=14, fontweight='bold')

    # 调整子图间距，避免标题和坐标轴标签重叠
    plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.96, bottom=0.04)
    print("Plotting results...")
    plt.show()


def train():
    env = HydrogenEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # [v2.5] 使用TD3算法替代SAC
    agent = TD3(state_dim, action_dim)
    replay_buffer = ReplayBuffer(capacity=100000, state_dim=state_dim, action_dim=action_dim)

    num_episodes = 200
    batch_size = 64
    all_rewards = []

    last_episode_profits = []
    last_episode_soc = []
    last_episode_power = {'re': [], 'net': [], 'fc': [], 'load': []}
    last_episode_green_h2 = {'green_ratio': [], 'threshold': [], 'power_from_re': [], 'power_from_grid': []}
    last_episode_arbitrage = {'arbitrage_bonus': [], 'price': [], 'ele_power': [], 'fc_power': []}
    last_episode_battery = {'soc': [], 'charge_power': [], 'discharge_power': []}  # v2.6新增

    print("=" * 70)
    print("Start Training Hydrogen Project (TD3 + Battery + Storage Arbitrage)")
    print("=" * 70)
    print(f"Algorithm: TD3 (Twin Delayed DDPG)")
    print(f"Battery Storage: {'Enabled' if Config.enable_battery_storage else 'Disabled'} (v2.6)")
    print(f"Green Hydrogen Strategy: {'Enabled' if Config.enable_threshold_strategy else 'Disabled'}")
    print(f"Storage Arbitrage: {'Enabled' if Config.enable_arbitrage_bonus else 'Disabled'}")
    print(f"Battery Capacity: {Config.battery_capacity} kWh")
    print(f"Base Threshold: {Config.base_power_threshold} kW")
    print(f"Green H2 Bonus: ${Config.green_hydrogen_bonus}/kg")
    print(f"Arbitrage Coef: {Config.arbitrage_bonus_coef}")
    print(f"I2S Penalty Weight: {Config.i2s_penalty_weight}")
    print("-" * 70)

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
            last_episode_battery = {'soc': [], 'charge_power': [], 'discharge_power': []}  # v2.6: 电池统计

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
                
                # 收集电池储能统计 (v2.6新增)
                if Config.enable_battery_storage:
                    last_episode_battery['soc'].append(info['battery_soc'])
                    last_episode_battery['charge_power'].append(info['battery_charge_power'])
                    last_episode_battery['discharge_power'].append(info['battery_discharge_power'])

        all_rewards.append(episode_reward)
        
        # [新增] 每个episode结束后更新学习率
        agent.step_schedulers()
        
        if (episode + 1) % 10 == 0:
            # TD3不需要显示alpha (无熵调优)
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Final SOC: {state[0]:.2f}, Updates: {agent.total_it}")

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
    
    # 打印电池储能统计 (v2.6新增)
    if Config.enable_battery_storage:
        battery_stats_summary = env.battery.get_statistics()
        print("=" * 60)
        print("BATTERY ENERGY STORAGE STATISTICS (Training Summary)")
        print("=" * 60)
        print(f"Final SOC:                {battery_stats_summary['current_soc']*100:.1f}%")
        print(f"Current Energy:           {battery_stats_summary['current_energy_kwh']:.2f} kWh")
        print(f"Total Charge Energy:      {battery_stats_summary['total_charge_kwh']:.2f} kWh")
        print(f"Total Discharge Energy:   {battery_stats_summary['total_discharge_kwh']:.2f} kWh")
        print(f"Total Throughput:         {battery_stats_summary['total_throughput_kwh']:.2f} kWh")
        print(f"Roundtrip Efficiency:     {battery_stats_summary['roundtrip_efficiency_pct']:.1f}%")
        print(f"Charge Cycles:            {battery_stats_summary['charge_cycles']:.2f}")
        print(f"Battery Degradation:      {battery_stats_summary['degradation_pct']:.2f}%")
        print("=" * 60 + "\n")
        
        battery_stats = last_episode_battery if len(last_episode_battery['soc']) > 0 else None
    else:
        battery_stats = None

    plot_results(all_rewards, last_episode_profits, last_episode_soc, last_episode_power, 
                green_h2_stats, arbitrage_stats, battery_stats)


if __name__ == "__main__":
    train()