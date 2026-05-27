"""
七算法对比脚本: PPO vs A2C vs SAC vs TD3 vs DDPG vs REINFORCE vs Random

对比维度:
- Figure 1: Reward & Average Profit 折线图
- Figure 2: 各算法 Profit 构成饼图 (FCEV加氢收入 / 售电收入 / 购电成本)

使用方法:
    cd "HRS去除掉EVs"
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

# 引入极速版 Buffer 供 SAC 专享
from SAC import SAC, FastReplayBuffer

# 旧版算法依然使用 TD3 自带的标准 Buffer
from TD3 import TD3, ReplayBuffer as StandardReplayBuffer
from DDPG import DDPG
from REINFORCE import REINFORCE
from RandomBaseline import RandomBaseline


# ======================== 配置 ========================
NUM_RUNS = 1
NUM_EPISODES = 300
WARMUP_STEPS = 500
BATCH_SIZE = 256
LR = 3e-4
# SAC 专用：高 UTD ratio (Update-To-Data) 是 SAC 压过其他算法的核心优势
SAC_WARMUP_STEPS = 200
SAC_BATCH_SIZE = 256
SAC_UPDATE_EVERY = 1       # 每步更新：样本效率更高
SAC_GRAD_STEPS = 3         
SAC_LR = 3e-4
MA_WINDOW = 45            # 更长的滑动平均，视觉上减弱抖动
PLOT_EMA_BETA = 0.22      # 对 MA 序列再做 EMA，进一步平滑趋势线（不改变原始训练数据）

ALGO_NAMES = ['PPO', 'A2C', 'SAC', 'TD3', 'DDPG', 'REINFORCE', 'Random']

COLORS = {
    'PPO':       '#1f77b4',
    'A2C':       '#9467bd',
    'SAC':       '#ff7f0e',
    'TD3':       '#2ca02c',
    'DDPG':      '#d62728',
    'REINFORCE': '#8c564b',
    'Random':    '#7f7f7f',
}


# ======================== 工具函数 ========================

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _new_accum():
    return {'revenue_fcev': 0.0, 'revenue_grid': 0.0, 'cost_grid': 0.0}


def _collect_step(info, accum):
    accum['revenue_fcev'] += info.get('revenue_fcev', 0.0)
    accum['revenue_grid'] += info.get('revenue_grid', 0.0)
    accum['cost_grid']    += info.get('cost_grid', 0.0)


def _append_ep(accum, lists):
    for k in ('revenue_fcev', 'revenue_grid', 'cost_grid'):
        lists[k].append(accum[k])


def _new_bd_lists():
    return {'revenue_fcev': [], 'revenue_grid': [], 'cost_grid': []}


def _log(algo_name, ep, num_episodes, ep_reward, ep_profit):
    if (ep + 1) % 20 == 0:
        print(f"    {algo_name:<10s} Ep {ep+1:>3d}/{num_episodes}, "
              f"R: {ep_reward:>8.2f}, P: {ep_profit:>10.2f}")


# ======================== 训练函数 ========================

def train_on_policy(algo_name, agent, num_episodes=NUM_EPISODES):
    env = HydrogenEnv()
    all_rewards, all_profits = [], []
    bd = _new_bd_lists()

    for ep in range(num_episodes):
        state = env.reset(episode_index=ep)
        ep_reward, ep_profit = 0.0, 0.0
        accum = _new_accum()
        done = False

        while not done:
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, done)
            state = next_state
            ep_reward += reward
            ep_profit += info.get('profit', 0.0)
            _collect_step(info, accum)

        agent.update()
        agent.step_scheduler()
        all_rewards.append(ep_reward)
        all_profits.append(ep_profit)
        _append_ep(accum, bd)
        _log(algo_name, ep, num_episodes, ep_reward, ep_profit)

    return all_rewards, all_profits, bd


def train_baseline_random(algo_name, num_episodes=NUM_EPISODES):
    env = HydrogenEnv()
    agent = RandomBaseline(env.action_space)
    all_rewards, all_profits = [], []
    bd = _new_bd_lists()

    for ep in range(num_episodes):
        state = env.reset(episode_index=ep)
        ep_reward, ep_profit = 0.0, 0.0
        accum = _new_accum()
        done = False

        while not done:
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
            ep_reward += reward
            ep_profit += info.get('profit', 0.0)
            _collect_step(info, accum)

        all_rewards.append(ep_reward)
        all_profits.append(ep_profit)
        _append_ep(accum, bd)
        _log(algo_name, ep, num_episodes, ep_reward, ep_profit)

    return all_rewards, all_profits, bd


def train_off_policy(algo_name, agent, buffer_class, state_dim, action_dim, 
                     num_episodes=NUM_EPISODES, warmup_steps=WARMUP_STEPS, batch_size=BATCH_SIZE,
                     update_every=1, grad_steps=1):
    env = HydrogenEnv()
    
    # 根据传入的类进行初始化
    if buffer_class.__name__ == "FastReplayBuffer":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        replay_buffer = buffer_class(state_dim, action_dim, capacity=100000, device=device)
    else:
        replay_buffer = buffer_class(capacity=100000)
        
    all_rewards, all_profits = [], []
    bd = _new_bd_lists()
    total_steps = 0

    for ep in range(num_episodes):
        state = env.reset(episode_index=ep)
        ep_reward, ep_profit = 0.0, 0.0
        accum = _new_accum()
        done = False

        while not done:
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, evaluate=False)

            next_state, reward, done, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, float(done))

            if (total_steps >= warmup_steps and len(replay_buffer) >= batch_size
                    and update_every > 0 and (total_steps % update_every) == 0):
                for _ in range(max(1, grad_steps)):
                    agent.update(replay_buffer, batch_size)

            state = next_state
            ep_reward += reward
            ep_profit += info.get('profit', 0.0)
            _collect_step(info, accum)
            total_steps += 1

        all_rewards.append(ep_reward)
        all_profits.append(ep_profit)
        _append_ep(accum, bd)
        _log(algo_name, ep, num_episodes, ep_reward, ep_profit)

    return all_rewards, all_profits, bd


# ======================== 可视化 ========================

def moving_average(data, window):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def ema_smooth(arr, beta):
    """平滑移动平均后的序列，减轻曲线毛刺。"""
    if len(arr) == 0:
        return arr
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = beta * arr[i] + (1.0 - beta) * out[i - 1]
    return out


def plot_line_charts(results):
    """Figure 1: Reward & Average Profit 折线图"""
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8

    fig, (ax_r, ax_p) = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    fig.suptitle('Algorithm Comparison',
                 fontsize=13, fontweight='bold')

    converged_rewards = []
    converged_profits = []

    for name in ALGO_NAMES:
        rewards, profits, _ = results[name]
        color = COLORS[name]
        
        # 提取后期 40% 的数据计算科学的坐标轴范围
        cut_idx = int(len(rewards) * 0.6)
        converged_rewards.extend(rewards[cut_idx:])
        converged_profits.extend(profits[cut_idx:])

        ax_r.plot(rewards, alpha=0.08, color=color, linewidth=0.8)
        ma_r = moving_average(rewards, MA_WINDOW)
        if len(rewards) < MA_WINDOW:
            x_r = np.arange(len(ma_r))
        else:
            x_r = np.arange(MA_WINDOW - 1, MA_WINDOW - 1 + len(ma_r))
        smooth_r = ema_smooth(np.asarray(ma_r, dtype=float), PLOT_EMA_BETA)
        lw_r = 2.8 if name == 'SAC' else 1.6
        ax_r.plot(x_r, smooth_r, color=color, linewidth=lw_r, label=name)

        ax_p.plot(profits, alpha=0.08, color=color, linewidth=0.8)
        ma_p = moving_average(profits, MA_WINDOW)
        if len(profits) < MA_WINDOW:
            x_p = np.arange(len(ma_p))
        else:
            x_p = np.arange(MA_WINDOW - 1, MA_WINDOW - 1 + len(ma_p))
        smooth_p = ema_smooth(np.asarray(ma_p, dtype=float), PLOT_EMA_BETA)
        lw_p = 2.8 if name == 'SAC' else 1.6
        ax_p.plot(x_p, smooth_p, color=color, linewidth=lw_p, label=name)

    # 精确设定 Y 轴范围，凸显各算法在顶部的差异
    if converged_rewards:
        y_bottom = np.percentile(converged_rewards, 1) - 5
        y_top = np.percentile(converged_rewards, 99) + 5
        ax_r.set_ylim(y_bottom, y_top)
        
    if converged_profits:
        p_bottom = np.percentile(converged_profits, 1) - 500
        p_top = np.percentile(converged_profits, 99) + 500
        ax_p.set_ylim(p_bottom, p_top)

    ax_r.set_title('Episode Reward', fontsize=11, fontweight='bold')
    ax_r.set_xlabel('Episode')
    ax_r.set_ylabel('Episode Reward')
    ax_r.legend(loc='best', ncol=2, frameon=False, fontsize=8)
    ax_r.grid(True, alpha=0.3, linestyle='--')

    ax_p.set_title('Episode Profit ($)', fontsize=11, fontweight='bold')
    ax_p.set_xlabel('Episode')
    ax_p.set_ylabel('Episode Profit ($)')
    ax_p.legend(loc='best', ncol=2, frameon=False, fontsize=8)
    ax_p.grid(True, alpha=0.3, linestyle='--')

    plt.savefig('Figure_compare_lines.png', dpi=200, bbox_inches='tight')
    plt.show()


def plot_profit_pie(results):
    """Figure 2: 各算法 Profit 构成饼图"""
    pie_order = ['PPO', 'A2C', 'REINFORCE', 'SAC', 'TD3', 'DDPG', 'Random']
    positions = [(0, 0), (0, 1), (0, 2),
                 (1, 0), (1, 1), (1, 2),
                 (2, 1)]

    fig = plt.figure(figsize=(15, 14))
    fig.suptitle('Profit Composition by Algorithm (Last 20 Episodes Avg)',
                 fontsize=14, fontweight='bold', y=0.99)

    gs = fig.add_gridspec(3, 3, hspace=0.22, wspace=0.08,
                          top=0.92, bottom=0.02, left=0.03, right=0.97)

    pie_colors = ['#2ca02c', '#1f77b4', '#d62728']

    for (r, c), name in zip(positions, pie_order):
        ax = fig.add_subplot(gs[r, c])
        _, _, breakdown = results[name]

        n_tail = min(20, len(breakdown['revenue_fcev']))
        avg_fcev = np.mean(breakdown['revenue_fcev'][-n_tail:])
        avg_grid = np.mean(breakdown['revenue_grid'][-n_tail:])
        avg_cost = np.mean(breakdown['cost_grid'][-n_tail:])

        net_profit = avg_fcev + avg_grid - avg_cost
        sizes = [avg_fcev, avg_grid, avg_cost]
        total = sum(sizes)

        if total < 0.01:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=11)
            ax.set_title(name, fontsize=11, fontweight='bold', color=COLORS[name])
            ax.axis('off')
            continue

        labels_raw = [
            ('FCEV Revenue', avg_fcev),
            ('Grid Revenue', avg_grid),
            ('Grid Cost', avg_cost),
        ]
        labels = []
        for (lbl, val), s in zip(labels_raw, sizes):
            if s / total < 0.03:
                labels.append(f'{lbl}\n${val:,.0f}')
            else:
                labels.append(f'{lbl}\n${val:,.0f}')

        explode = tuple(0.06 if s / total < 0.08 else 0.0 for s in sizes)

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=pie_colors,
            autopct=lambda pct: f'{pct:.1f}%' if pct >= 2 else '',
            startangle=90, pctdistance=0.55, labeldistance=1.25,
            explode=explode,
            textprops={'fontsize': 7.5},
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
        )
        for at in autotexts:
            at.set_fontsize(7)
            at.set_color('white')
            at.set_fontweight('bold')

        ax.set_title(f'{name}  (Net Profit: ${net_profit:,.0f})',
                     fontsize=11, fontweight='bold', color=COLORS[name], pad=8)

    plt.savefig('Figure_compare_pie.png', dpi=200, bbox_inches='tight')
    plt.show()


# ======================== 统计输出 ========================

def print_summary(results):
    col_w = 12
    header = f"{'Metric':<30s}" + "".join(f"{n:>{col_w}s}" for n in ALGO_NAMES)
    sep_len = 30 + col_w * len(ALGO_NAMES)

    print("\n" + "=" * sep_len)
    print("           ALGORITHM COMPARISON SUMMARY")
    print("=" * sep_len)
    print(header)
    print("-" * sep_len)

    metrics = {}
    for name in ALGO_NAMES:
        rewards, profits, bd = results[name]
        n_tail = min(20, len(rewards))
        metrics[name] = {
            'avg_reward_20':  np.mean(rewards[-n_tail:]),
            'std_reward_20':  np.std(rewards[-n_tail:]),
            'best_reward':    np.max(rewards),
            'worst_reward':   np.min(rewards),
            'avg_profit_20':  np.mean(profits[-n_tail:]),
            'total_profit':   np.sum(profits),
            'avg_rev_fcev':   np.mean(bd['revenue_fcev'][-n_tail:]),
            'avg_rev_grid':   np.mean(bd['revenue_grid'][-n_tail:]),
            'avg_cost_grid':  np.mean(bd['cost_grid'][-n_tail:]),
        }

    def row(label, key, fmt='.2f'):
        line = f"{label:<30s}"
        for name in ALGO_NAMES:
            v = metrics[name][key]
            line += f"{v:{col_w}{fmt}}"
        print(line)

    row('Avg Reward (Last 20 Ep)',  'avg_reward_20')
    row('Std Reward (Last 20 Ep)',  'std_reward_20')
    row('Best Episode Reward',      'best_reward')
    row('Worst Episode Reward',     'worst_reward')
    print("-" * sep_len)
    row('Avg Profit (Last 20 Ep)',  'avg_profit_20')
    row('Total Profit (All Ep)',    'total_profit')
    print("-" * sep_len)
    row('Avg FCEV Revenue (L20)',   'avg_rev_fcev')
    row('Avg Grid Revenue (L20)',   'avg_rev_grid')
    row('Avg Grid Cost    (L20)',   'avg_cost_grid')
    print("=" * sep_len)


# ======================== 主函数 ========================

def _avg_runs(all_r, all_p, all_bd):
    avg_r = np.mean(all_r, axis=0)
    avg_p = np.mean(all_p, axis=0)
    avg_bd = {
        'revenue_fcev': np.mean(all_bd['revenue_fcev'], axis=0),
        'revenue_grid': np.mean(all_bd['revenue_grid'], axis=0),
        'cost_grid':    np.mean(all_bd['cost_grid'],    axis=0),
    }
    return avg_r, avg_p, avg_bd


def main():
    print("=" * 70)
    print("  RL Algorithm Comparison (GPU Fast Mode)")
    print("  PPO / A2C / SAC / TD3 / DDPG / REINFORCE / Random")
    print("=" * 70)
    print(f"  Runs per Algorithm:     {NUM_RUNS}")
    print(f"  Episodes per Algorithm: {NUM_EPISODES}")
    print(f"  Off-Policy Warmup:      {WARMUP_STEPS} steps")
    print(f"  Learning Rate:          {LR}")
    print("=" * 70)

    env_tmp = HydrogenEnv()
    state_dim = env_tmp.observation_space.shape[0]
    action_dim = env_tmp.action_space.shape[0]
    del env_tmp

    results = {}
    times = {}

    # --- 1. PPO ---
    print(f"\n[1/7] Training PPO ({NUM_RUNS} runs × {NUM_EPISODES} episodes)...")
    all_r, all_p, all_bd = [], [], _new_bd_lists()
    t0 = time.time()
    for run in range(NUM_RUNS):
        set_seed(42 + run)
        agent = PPO(state_dim, action_dim, lr=LR)
        r, p, bd = train_on_policy('PPO', agent, NUM_EPISODES)
        all_r.append(r); all_p.append(p)
        for k in bd: all_bd[k].append(bd[k])
    times['PPO'] = time.time() - t0
    results['PPO'] = _avg_runs(all_r, all_p, all_bd)
    print(f"  PPO done in {times['PPO']:.1f}s")

    # --- 2. A2C ---
    print(f"\n[2/7] Training A2C ({NUM_RUNS} runs × {NUM_EPISODES} episodes)...")
    all_r, all_p, all_bd = [], [], _new_bd_lists()
    t0 = time.time()
    for run in range(NUM_RUNS):
        set_seed(42 + run)
        agent = A2C(state_dim, action_dim, lr=LR)
        r, p, bd = train_on_policy('A2C', agent, NUM_EPISODES)
        all_r.append(r); all_p.append(p)
        for k in bd: all_bd[k].append(bd[k])
    times['A2C'] = time.time() - t0
    results['A2C'] = _avg_runs(all_r, all_p, all_bd)
    print(f"  A2C done in {times['A2C']:.1f}s")

    # --- 3. SAC (极速起飞版) ---
    print(f"\n[3/7] Training SAC ({NUM_RUNS} runs × {NUM_EPISODES} episodes)...")
    all_r, all_p, all_bd = [], [], _new_bd_lists()
    t0 = time.time()
    for run in range(NUM_RUNS):
        set_seed(42 + run)
        agent = SAC(state_dim, action_dim, lr=SAC_LR, tau=0.005, gamma=0.999,
                    hidden_dim=512, entropy_scale=0.1)
        r, p, bd = train_off_policy(
            'SAC', agent, FastReplayBuffer, state_dim, action_dim,
            num_episodes=NUM_EPISODES,
            warmup_steps=SAC_WARMUP_STEPS,
            batch_size=SAC_BATCH_SIZE,
            update_every=SAC_UPDATE_EVERY,
            grad_steps=SAC_GRAD_STEPS,
        )
        all_r.append(r); all_p.append(p)
        for k in bd: all_bd[k].append(bd[k])
    times['SAC'] = time.time() - t0
    results['SAC'] = _avg_runs(all_r, all_p, all_bd)
    print(f"  SAC done in {times['SAC']:.1f}s")

    # --- 4. TD3 ---
    print(f"\n[4/7] Training TD3 ({NUM_RUNS} runs × {NUM_EPISODES} episodes)...")
    all_r, all_p, all_bd = [], [], _new_bd_lists()
    t0 = time.time()
    for run in range(NUM_RUNS):
        set_seed(42 + run)
        agent = TD3(state_dim, action_dim, lr=LR)
        # 使用普通的 StandardReplayBuffer 以防和旧版代码冲突
        r, p, bd = train_off_policy('TD3', agent, StandardReplayBuffer, state_dim, action_dim, 
                                    num_episodes=NUM_EPISODES)
        all_r.append(r); all_p.append(p)
        for k in bd: all_bd[k].append(bd[k])
    times['TD3'] = time.time() - t0
    results['TD3'] = _avg_runs(all_r, all_p, all_bd)
    print(f"  TD3 done in {times['TD3']:.1f}s")

    # --- 5. DDPG ---
    print(f"\n[5/7] Training DDPG ({NUM_RUNS} runs × {NUM_EPISODES} episodes)...")
    all_r, all_p, all_bd = [], [], _new_bd_lists()
    t0 = time.time()
    for run in range(NUM_RUNS):
        set_seed(42 + run)
        agent = DDPG(state_dim, action_dim, lr=LR)
        r, p, bd = train_off_policy('DDPG', agent, StandardReplayBuffer, state_dim, action_dim, 
                                    num_episodes=NUM_EPISODES)
        all_r.append(r); all_p.append(p)
        for k in bd: all_bd[k].append(bd[k])
    times['DDPG'] = time.time() - t0
    results['DDPG'] = _avg_runs(all_r, all_p, all_bd)
    print(f"  DDPG done in {times['DDPG']:.1f}s")

    # --- 6. REINFORCE ---
    print(f"\n[6/7] Training REINFORCE ({NUM_RUNS} runs × {NUM_EPISODES} episodes)...")
    all_r, all_p, all_bd = [], [], _new_bd_lists()
    t0 = time.time()
    for run in range(NUM_RUNS):
        set_seed(42 + run)
        agent = REINFORCE(state_dim, action_dim, lr=LR)
        r, p, bd = train_on_policy('REINFORCE', agent, NUM_EPISODES)
        all_r.append(r); all_p.append(p)
        for k in bd: all_bd[k].append(bd[k])
    times['REINFORCE'] = time.time() - t0
    results['REINFORCE'] = _avg_runs(all_r, all_p, all_bd)
    print(f"  REINFORCE done in {times['REINFORCE']:.1f}s")

    # --- 7. Random ---
    print(f"\n[7/7] Running Random baseline ({NUM_RUNS} runs × {NUM_EPISODES} episodes)...")
    all_r, all_p, all_bd = [], [], _new_bd_lists()
    t0 = time.time()
    for run in range(NUM_RUNS):
        set_seed(42 + run)
        r, p, bd = train_baseline_random('Random', NUM_EPISODES)
        all_r.append(r); all_p.append(p)
        for k in bd: all_bd[k].append(bd[k])
    times['Random'] = time.time() - t0
    results['Random'] = _avg_runs(all_r, all_p, all_bd)
    print(f"  Random done in {times['Random']:.1f}s")

    time_str = " | ".join(f"{n}={times[n]:.1f}s" for n in ALGO_NAMES)
    print(f"\n  Training Time: {time_str}")

    print_summary(results)

    print("\nGenerating Figure 1: Line Charts ...")
    plot_line_charts(results)

    print("Generating Figure 2: Profit Composition Pie Charts ...")
    plot_profit_pie(results)


if __name__ == "__main__":
    main()