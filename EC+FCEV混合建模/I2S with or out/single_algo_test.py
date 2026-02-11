import numpy as np
import matplotlib.pyplot as plt
from env import HydrogenEnv
from PPO import PPO
from config import Config


def plot_results(rewards, profits, soc_history, power_balance, green_h2_stats=None, arbitrage_stats=None, 
                battery_stats=None, compressor_stats=None, revenue_stats=None):
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

    # 图2: 双储能SOC + 电池充放电 (合并原图2和图6)
    if battery_stats is not None and len(battery_stats['soc']) > 0:
        battery_soc_arr = np.array(battery_stats['soc']) * 100
        h2_soc_arr = np.array(soc_history) * 100
        
        # 确保两个数组长度一致，取最小长度
        min_len = min(len(battery_soc_arr), len(h2_soc_arr))
        battery_soc_arr = battery_soc_arr[:min_len]
        h2_soc_arr = h2_soc_arr[:min_len]
        steps_soc = range(min_len)
        
        # 左轴: SOC曲线
        axs[0, 1].plot(steps_soc, h2_soc_arr, color='#1f77b4', label='H2 Storage SOC', linewidth=2.5)
        axs[0, 1].plot(steps_soc, battery_soc_arr, color='#ff6b6b', label='Battery SOC', linewidth=2.5)
        axs[0, 1].axhline(y=Config.storage_initial*100, color='blue', linestyle='--', linewidth=1.5, alpha=0.5)
        axs[0, 1].axhline(y=Config.battery_initial_soc*100, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        axs[0, 1].set_ylabel("SOC (%)", fontsize=11)
        axs[0, 1].set_ylim(0, 100)
        axs[0, 1].legend(loc='upper left', fontsize=9)
        
        # 右轴: 电池充放电功率 (柱状图)
        ax_batt_power = axs[0, 1].twinx()
        charge_power = np.array(battery_stats['charge_power'][:min_len])
        discharge_power = np.array(battery_stats['discharge_power'][:min_len])
        
        ax_batt_power.bar(steps_soc, charge_power, color='#4ecdc4', alpha=0.3, 
                         width=1.0, label='Charge', edgecolor='none')
        ax_batt_power.bar(steps_soc, -discharge_power, color='#ffa07a', alpha=0.3, 
                         width=1.0, label='Discharge', edgecolor='none')
        ax_batt_power.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax_batt_power.set_ylabel("Battery Power (kW)", fontsize=11, color='#4ecdc4')
        ax_batt_power.tick_params(axis='y', labelcolor='#4ecdc4', labelsize=9)
        ax_batt_power.set_ylim(-300, 300)
        ax_batt_power.legend(loc='upper right', fontsize=9)
        
        axs[0, 1].set_title("Dual Storage SOC + Battery Power", fontsize=14, fontweight='bold')
        axs[0, 1].set_xlabel("Time Step", fontsize=11)
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
    
    # 图5: 压缩机系统功耗分布 (v3.1核心创新) - 替换原Storage Arbitrage
    if compressor_stats is not None and len(compressor_stats['c1_power']) > 0:
        steps = range(len(compressor_stats['c1_power']))
        c1_power = np.array(compressor_stats['c1_power'])
        c2_power = np.array(compressor_stats['c2_power'])
        c3_power = np.array(compressor_stats['c3_power'])
        
        # 堆叠面积图显示三级压缩机功耗
        axs[2, 0].fill_between(steps, 0, c1_power, color='#3498db', alpha=0.7, label='C1 (2→35bar)')
        axs[2, 0].fill_between(steps, c1_power, c1_power + c2_power, 
                              color='#e74c3c', alpha=0.7, label='C2 (35→500bar)')
        axs[2, 0].fill_between(steps, c1_power + c2_power, c1_power + c2_power + c3_power,
                              color='#f39c12', alpha=0.7, label='C3 (500→700bar)')
        
        # 标记旁路激活点（用散点表示）
        bypass_c1 = np.array(compressor_stats['bypass_c1'])
        bypass_c2 = np.array(compressor_stats['bypass_c2'])
        bypass_c3 = np.array(compressor_stats['bypass_c3'])
        
        # 找到旁路激活的时刻
        bypass_times_c1 = [i for i in range(len(bypass_c1)) if bypass_c1[i] == 1]
        bypass_times_c2 = [i for i in range(len(bypass_c2)) if bypass_c2[i] == 1]
        bypass_times_c3 = [i for i in range(len(bypass_c3)) if bypass_c3[i] == 1]
        
        # 在旁路时刻标记点
        if len(bypass_times_c1) > 0:
            axs[2, 0].scatter(bypass_times_c1, [10] * len(bypass_times_c1), 
                            color='#3498db', marker='x', s=50, alpha=0.8, label=f'C1 Bypass ({len(bypass_times_c1)})')
        if len(bypass_times_c2) > 0:
            axs[2, 0].scatter(bypass_times_c2, [20] * len(bypass_times_c2), 
                            color='#e74c3c', marker='x', s=50, alpha=0.8, label=f'C2 Bypass ({len(bypass_times_c2)})')
        if len(bypass_times_c3) > 0:
            axs[2, 0].scatter(bypass_times_c3, [30] * len(bypass_times_c3), 
                            color='#f39c12', marker='x', s=50, alpha=0.8, label=f'C3 Bypass ({len(bypass_times_c3)})')
        
        axs[2, 0].set_title("Compressor Power Distribution (v3.1)", fontsize=14, fontweight='bold')
        axs[2, 0].set_ylabel("Power (kW)", fontsize=11)
        axs[2, 0].set_xlabel("Time Step", fontsize=11)
        axs[2, 0].legend(loc='best', fontsize=9, ncol=2)
        axs[2, 0].grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
    else:
        # 没有压缩机统计数据
        axs[2, 0].text(0.5, 0.5, 'Compressor Data Not Available', 
                      ha='center', va='center', fontsize=12, transform=axs[2, 0].transAxes)
        axs[2, 0].set_title("Compressor Power Distribution (v3.1)", fontsize=14, fontweight='bold')
    
    # 图6: 系统收益分解 (v3.1新增 - 替换原Battery图)
    if revenue_stats is not None and len(revenue_stats['ev']) > 0:
        steps = range(len(revenue_stats['ev']))
        
        # 累积收益计算
        ev_revenue = np.cumsum(revenue_stats['ev'])
        fcev_revenue = np.cumsum(revenue_stats['fcev'])
        grid_sell = np.cumsum(revenue_stats['grid_sell'])
        green_h2_bonus = np.cumsum(revenue_stats['green_h2'])
        grid_cost = np.cumsum(revenue_stats['grid_cost'])
        
        # 堆叠面积图显示收入来源
        axs[2, 1].fill_between(steps, 0, ev_revenue, color='#3498db', alpha=0.7, label='EV Charging')
        axs[2, 1].fill_between(steps, ev_revenue, ev_revenue + fcev_revenue, 
                              color='#e74c3c', alpha=0.7, label='FCEV Refueling')
        axs[2, 1].fill_between(steps, ev_revenue + fcev_revenue, 
                              ev_revenue + fcev_revenue + grid_sell,
                              color='#2ecc71', alpha=0.7, label='Grid Sell')
        axs[2, 1].fill_between(steps, ev_revenue + fcev_revenue + grid_sell,
                              ev_revenue + fcev_revenue + grid_sell + green_h2_bonus,
                              color='#f39c12', alpha=0.7, label='Green H2 Bonus')
        
        # 成本曲线（负值）- 用更明显的红色虚线
        axs[2, 1].plot(steps, -np.array(grid_cost), color='#e74c3c', 
                      linewidth=2.5, linestyle='--', label='Grid Cost', alpha=0.8)
        
        axs[2, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axs[2, 1].set_title("System Revenue Breakdown (Cumulative)", fontsize=14, fontweight='bold')
        axs[2, 1].set_ylabel("Revenue ($)", fontsize=11)
        axs[2, 1].set_xlabel("Time Step", fontsize=11)
        axs[2, 1].legend(loc='upper left', fontsize=9, ncol=2)
        axs[2, 1].grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
        
        # 在图上添加统计信息
        final_ev = ev_revenue[-1]
        final_fcev = fcev_revenue[-1]
        final_grid_sell = grid_sell[-1]
        final_green = green_h2_bonus[-1]
        final_cost = grid_cost[-1]
        total_revenue = final_ev + final_fcev + final_grid_sell + final_green
        net_profit = total_revenue - final_cost
        
        textstr = f'Total Revenue: ${total_revenue:.0f}\n' \
                  f'Grid Cost: ${final_cost:.0f}\n' \
                  f'Net Profit: ${net_profit:.0f}\n' \
                  f'Profit Margin: {net_profit/total_revenue*100:.1f}%'
        axs[2, 1].text(0.98, 0.02, textstr, transform=axs[2, 1].transAxes,
                      verticalalignment='bottom', horizontalalignment='right',
                      fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        # 没有收益统计
        axs[2, 1].text(0.5, 0.5, 'Revenue Data Not Available', 
                      ha='center', va='center', fontsize=12, transform=axs[2, 1].transAxes)
        axs[2, 1].set_title("System Revenue Breakdown", fontsize=14, fontweight='bold')

    # 调整子图间距，避免标题和坐标轴标签重叠
    plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.96, bottom=0.04)
    print("Plotting results...")
    plt.show()


def train():
    env = HydrogenEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # [v3.0] 使用PPO算法
    agent = PPO(state_dim, action_dim, lr=3e-4)

    num_episodes = 200
    all_rewards = []

    last_episode_profits = []
    last_episode_soc = []
    last_episode_power = {'re': [], 'net': [], 'fc': [], 'load': []}
    last_episode_green_h2 = {'green_ratio': [], 'power_from_re': [], 'power_from_grid': []}
    last_episode_arbitrage = {'arbitrage_bonus': [], 'price': [], 'ele_power': [], 'fc_power': []}
    last_episode_battery = {'soc': [], 'charge_power': [], 'discharge_power': []}  # v2.6新增
    last_episode_compressor = {'c1_power': [], 'c2_power': [], 'c3_power': [], 
                               'bypass_c1': [], 'bypass_c2': [], 'bypass_c3': []}  # v3.1新增
    last_episode_revenues = {'ev': [], 'fcev': [], 'grid_sell': [], 'green_h2': [], 
                            'grid_cost': [], 'total_profit': []}  # v3.1新增：收益分解

    print("=" * 70)
    print("Start Training Hydrogen Project (PPO + Battery + Storage Arbitrage)")
    print("=" * 70)
    print(f"Algorithm: PPO (Proximal Policy Optimization)")
    print(f"Battery Storage: {'Enabled' if Config.enable_battery_storage else 'Disabled'} (v2.6)")
    print(f"Storage Arbitrage: {'Enabled' if Config.enable_arbitrage_bonus else 'Disabled'}")
    print(f"Battery Capacity: {Config.battery_capacity} kWh")
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
            last_episode_green_h2 = {'green_ratio': [], 'power_from_re': [], 'power_from_grid': []}
            last_episode_arbitrage = {'arbitrage_bonus': [], 'price': [], 'ele_power': [], 'fc_power': []}
            last_episode_battery = {'soc': [], 'charge_power': [], 'discharge_power': []}  # v2.6: 电池统计

        # PPO: on-policy，收集一个完整episode的数据
        while not done:
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, info = env.step(action)

            # PPO存储经验
            agent.store_transition(state, action, reward, done)
            
            state = next_state
            episode_reward += reward

            if episode == num_episodes - 1:
                last_episode_profits.append(info['profit'])
                last_episode_soc.append(info['soc'])
                last_episode_power['re'].append(info['re_gen'])
                last_episode_power['net'].append(info['net_power'])
                last_episode_power['fc'].append(info['fc_power'])
                last_episode_power['load'].append(info['load_power'])
                
                # 收集绿氢统计
                last_episode_green_h2['green_ratio'].append(info['green_h2_ratio'])
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
                
                # 收集压缩机统计 (v3.1新增)
                if 'comp_c1_power' in info:
                    last_episode_compressor['c1_power'].append(info.get('comp_c1_power', 0))
                    last_episode_compressor['c2_power'].append(info.get('comp_c2_power', 0))
                    last_episode_compressor['c3_power'].append(info.get('comp_c3_power', 0))
                    # 旁路状态 (功耗为0表示旁路激活)
                    last_episode_compressor['bypass_c1'].append(1 if info.get('comp_c1_power', 0) == 0 else 0)
                    last_episode_compressor['bypass_c2'].append(1 if info.get('comp_c2_power', 0) == 0 else 0)
                    last_episode_compressor['bypass_c3'].append(1 if info.get('comp_c3_power', 0) == 0 else 0)
                
                # 收集收益分解统计 (v3.1新增)
                last_episode_revenues['ev'].append(info.get('revenue_ev', 0))
                last_episode_revenues['fcev'].append(info.get('revenue_fcev', 0))
                last_episode_revenues['grid_sell'].append(info.get('revenue_grid', 0))
                last_episode_revenues['green_h2'].append(info.get('green_h2_bonus', 0))
                last_episode_revenues['grid_cost'].append(info.get('cost_grid', 0))
                last_episode_revenues['total_profit'].append(info.get('profit', 0))

        # PPO: episode结束后更新
        update_info = agent.update()
        
        all_rewards.append(episode_reward)
        
        # 每个episode结束后更新学习率
        agent.step_scheduler()
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, "
                  f"Final SOC: {state[0]:.2f}, Updates: {agent.total_updates}")

    green_h2_stats = last_episode_green_h2 if len(last_episode_green_h2['green_ratio']) > 0 else None
    
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
    
    # 打印电网交互统计 (v3.1新增: 验证高买低卖策略)
    if len(last_episode_revenues['grid_cost']) > 0 and len(last_episode_power['net']) > 0:
        print("=" * 60)
        print("GRID INTERACTION STATISTICS (v3.1 - Buy Low, Sell High)")
        print("=" * 60)
        
        net_power_arr = np.array(last_episode_power['net'])
        
        # 统计购电和售电
        purchase_steps = np.where(net_power_arr < 0)[0]
        sell_steps = np.where(net_power_arr > 0)[0]
        
        total_purchase_power = abs(net_power_arr[purchase_steps].sum()) * Config.dt if len(purchase_steps) > 0 else 0
        total_sell_power = net_power_arr[sell_steps].sum() * Config.dt if len(sell_steps) > 0 else 0
        
        print(f"Grid Purchase Events:     {len(purchase_steps)} times")
        print(f"Grid Sell Events:         {len(sell_steps)} times")
        print(f"Total Energy Purchased:   {total_purchase_power:.2f} kWh")
        print(f"Total Energy Sold:        {total_sell_power:.2f} kWh")
        print(f"Net Energy from Grid:     {total_purchase_power - total_sell_power:.2f} kWh")
        
        # 成本和收入
        total_cost = sum(last_episode_revenues['grid_cost'])
        total_sell_revenue = sum(last_episode_revenues['grid_sell'])
        
        print(f"Grid Purchase Cost:       ${total_cost:.2f}")
        print(f"Grid Sell Revenue:        ${total_sell_revenue:.2f}")
        print(f"Net Grid Cost:            ${total_cost - total_sell_revenue:.2f}")
        
        # 平均价格分析
        if len(last_episode_arbitrage['price']) > 0:
            prices = np.array(last_episode_arbitrage['price'])
            avg_price = prices.mean()
            
            if len(purchase_steps) > 0:
                avg_purchase_price = prices[purchase_steps].mean()
                print(f"Avg Purchase Price:       ${avg_purchase_price:.4f}/kWh")
            
            if len(sell_steps) > 0:
                avg_sell_price = prices[sell_steps].mean()
                print(f"Avg Sell Price:           ${avg_sell_price:.4f}/kWh")
            
            print(f"Overall Avg Price:        ${avg_price:.4f}/kWh")
            
            # 套利效果判断
            if len(purchase_steps) > 0 and len(sell_steps) > 0:
                price_spread = avg_sell_price - avg_purchase_price
                if price_spread > 0:
                    print(f"✓ Arbitrage Success:      YES (Buy Low ${avg_purchase_price:.4f}, Sell High ${avg_sell_price:.4f})")
                    print(f"  Price Spread:           ${price_spread:.4f}/kWh ({price_spread/avg_price*100:.1f}%)")
                else:
                    print(f"✗ Arbitrage Success:      NO (Need Optimization)")
                    print(f"  Price Spread:           ${price_spread:.4f}/kWh (Negative!)")
        
        print("=" * 60 + "\n")
    
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
    
    # 打印压缩机系统统计 (v3.1新增)
    comp_stats = env.comp_system.get_statistics()
    print("=" * 60)
    print("COMPRESSOR SYSTEM STATISTICS (v3.1 - Advanced Control)")
    print("=" * 60)
    print(f"[C1] Energy Consumption:  {comp_stats['c1']['total_energy_kwh']:.2f} kWh")
    print(f"[C1] VSD Savings:         {comp_stats['c1']['vsd_savings_kwh']:.2f} kWh")
    print(f"[C1] Cooling Savings:     {comp_stats['c1']['cooling_savings_kwh']:.2f} kWh")
    print(f"[C2] Energy Consumption:  {comp_stats['c2']['total_energy_kwh']:.2f} kWh")
    print(f"[C2] VSD Savings:         {comp_stats['c2']['vsd_savings_kwh']:.2f} kWh")
    print(f"[C2] Cooling Savings:     {comp_stats['c2']['cooling_savings_kwh']:.2f} kWh")
    print(f"[C3] Energy Consumption:  {comp_stats['c3']['total_energy_kwh']:.2f} kWh")
    print(f"[C3] VSD Savings:         {comp_stats['c3']['vsd_savings_kwh']:.2f} kWh")
    print(f"[C3] Cooling Savings:     {comp_stats['c3']['cooling_savings_kwh']:.2f} kWh")
    print("-" * 60)
    print(f"Total VSD Savings:        {comp_stats['total_vsd_savings_kwh']:.2f} kWh ({comp_stats['total_vsd_savings_kwh']/(comp_stats['c1']['total_energy_kwh']+comp_stats['c2']['total_energy_kwh']+comp_stats['c3']['total_energy_kwh']+1e-6)*100:.1f}%)")
    print(f"Total Cooling Savings:    {comp_stats['total_cooling_savings_kwh']:.2f} kWh")
    print(f"Bypass Savings:           {comp_stats['bypass_savings_kwh']:.2f} kWh")
    print(f"Bypass Activations:")
    print(f"  - C1: {comp_stats['bypass_activations']['c1']} times")
    print(f"  - C2: {comp_stats['bypass_activations']['c2']} times")
    print(f"  - C3: {comp_stats['bypass_activations']['c3']} times")
    total_savings = comp_stats['total_vsd_savings_kwh'] + comp_stats['total_cooling_savings_kwh'] + comp_stats['bypass_savings_kwh']
    print(f"TOTAL Compressor Savings: {total_savings:.2f} kWh")
    print("=" * 60 + "\n")

    plot_results(all_rewards, last_episode_profits, last_episode_soc, last_episode_power, 
                green_h2_stats, arbitrage_stats, battery_stats, last_episode_compressor, last_episode_revenues)


if __name__ == "__main__":
    train()
