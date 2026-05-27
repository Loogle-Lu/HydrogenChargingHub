"""
七算法对比脚本 (改进版): PPO vs A2C vs SAC vs TD3 vs DDPG vs REINFORCE vs Random

改进点 (相比原版):
1. 所有算法共享 episode_index → 面对相同外生场景 (电价/风光/到达), 消除随机噪声
2. SAC 使用 UTD=2 (每环境步 2 次梯度更新):
   - 学术依据: SAC 的熵正则化天然防止高 UTD 下的过拟合
   - 参考: Chen et al. "Randomized Ensembled Double Q-Learning" (REDQ, ICLR 2021)
   - DDPG/TD3 保持 UTD=1 (高 UTD 时确定性策略容易过拟合)
3. SAC warmup=1000 (收集更多初始数据 → Critic 初始估值更准)
4. NUM_RUNS=3 (3次独立重复 → 减少随机种子偏差)

使用方法:
    python compare.py
"""

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import time
from pathlib import Path
from tqdm.auto import tqdm
from env import HydrogenEnv
from config import Config
from PPO import PPO
from A2C import A2C

from SAC import SAC, FastReplayBuffer
from TD3 import TD3, ReplayBuffer as StandardReplayBuffer
from DDPG import DDPG
from REINFORCE import REINFORCE
from RandomBaseline import RandomBaseline


# ======================== 配置 ========================
NUM_RUNS = 5                    # ← 改: 1 → 3 (统计可靠性)
NUM_EPISODES = 1000
WARMUP_STEPS = 250              # DDPG / TD3 warmup
SAC_WARMUP_STEPS = 500         # ← 新增: SAC 独立 warmup (更多初始数据)
BATCH_SIZE = 256
LR = 3e-4
MA_WINDOW = 20
SCRIPT_DIR = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAC_BUFFER_CAPACITY = 50_000       # GPU replay buffer: keep small for low-memory school GPUs

ALGO_NAMES = [
    'PPO', 'A2C', 'SAC', 'TD3', 'DDPG', 'REINFORCE', 'Random',
    'Rule-Service', 'Rule-Profit'
]

COLORS = {
    'PPO':       '#1f77b4',
    'A2C':       '#9467bd',
    'SAC':       '#d62728',       # 红 (原与 DDPG 橙区分)
    'TD3':       '#2ca02c',
    'DDPG':      '#ff7f0e',       # 橙
    'REINFORCE': '#8c564b',
    'Random':    '#7f7f7f',
    'Rule-Service': '#1f3a5f',    # 与 exp3 保持一致
    'Rule-Profit':  '#4d4d4d',    # 与 exp3 保持一致
}


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cleanup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _new_accum():
    return {'revenue_fcev': 0.0, 'revenue_fc': 0.0, 'revenue_grid': 0.0, 'cost_grid': 0.0}


def _collect_step(info, accum):
    accum['revenue_fcev'] += info.get('revenue_fcev', 0.0)
    accum['revenue_fc']   += info.get('revenue_fc',   0.0)
    accum['revenue_grid'] += info.get('revenue_grid', 0.0)
    accum['cost_grid']    += info.get('cost_grid', 0.0)


def _append_ep(accum, lists):
    for k in ('revenue_fcev', 'revenue_fc', 'revenue_grid', 'cost_grid'):
        lists[k].append(accum[k])


def _new_bd_lists():
    return {'revenue_fcev': [], 'revenue_fc': [], 'revenue_grid': [], 'cost_grid': []}


def _log(algo_name, ep, num_episodes, ep_reward, ep_profit):
    if (ep + 1) % 50 == 0:
        tqdm.write(f"    {algo_name:<10s} Ep {ep+1:>4d}/{num_episodes}, "
                   f"R: {ep_reward:>8.2f}, P: {ep_profit:>10.2f}")


# =====================================================================
# 改动核心: env.reset(episode_index=ep) → 同 episode 号的各算法面对相同场景
# =====================================================================

def train_on_policy(algo_name, agent, num_episodes=NUM_EPISODES):
    env = HydrogenEnv()
    all_rewards, all_profits = [], []
    bd = _new_bd_lists()

    progress = tqdm(range(num_episodes), desc=f"{algo_name} episodes",
                    unit="ep", leave=False)
    for ep in progress:
        state = env.reset(episode_index=ep)     # ← 改: 传入 episode_index
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
        progress.set_postfix(reward=f"{ep_reward:.2f}", profit=f"{ep_profit:.2f}")
        _log(algo_name, ep, num_episodes, ep_reward, ep_profit)

    return all_rewards, all_profits, bd


def train_baseline_random(algo_name, num_episodes=NUM_EPISODES):
    env = HydrogenEnv()
    agent = RandomBaseline(env.action_space)
    all_rewards, all_profits = [], []
    bd = _new_bd_lists()

    progress = tqdm(range(num_episodes), desc=f"{algo_name} episodes",
                    unit="ep", leave=False)
    for ep in progress:
        state = env.reset(episode_index=ep)     # ← 改
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
        progress.set_postfix(reward=f"{ep_reward:.2f}", profit=f"{ep_profit:.2f}")
        _log(algo_name, ep, num_episodes, ep_reward, ep_profit)

    return all_rewards, all_profits, bd


def _build_rule_action(state, mode):
    """
    与 exp3_actions_ablation.py 保持一致的 6 维规则动作:
    [ele, fc, c1_cool, c2_cool, c3_pressure_bias, bypass_bias]
    """
    s = np.asarray(state, dtype=np.float32).flatten()
    price_norm = float(np.clip(s[0], 0.0, 1.5))
    t3_avg_soc = float(np.mean(s[4:7]))
    queue_pressure = float(s[7] + s[8])

    low_price_norm = Config.price_threshold_low / Config.price_max
    high_price_norm = Config.price_threshold_high / Config.price_max

    if mode == "service":
        urgent = (queue_pressure > 0.05) or (t3_avg_soc < 0.60)
        ele = 1.0 if urgent else 0.65
        fc = 0.0 if urgent else 0.20
        c1_cool = 1.0 if price_norm < low_price_norm else 0.30
        c2_cool = 1.0 if price_norm < low_price_norm else 0.30
        c3_pressure_bias = 1.0
        bypass_bias = 0.0
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


def train_rule_baseline(algo_name, mode, num_episodes=NUM_EPISODES):
    env = HydrogenEnv()
    all_rewards, all_profits = [], []
    bd = _new_bd_lists()

    progress = tqdm(range(num_episodes), desc=f"{algo_name} episodes",
                    unit="ep", leave=False)
    for ep in progress:
        state = env.reset(episode_index=ep)
        ep_reward, ep_profit = 0.0, 0.0
        accum = _new_accum()
        done = False

        while not done:
            action = _build_rule_action(state, mode)
            next_state, reward, done, info = env.step(action)
            state = next_state
            ep_reward += reward
            ep_profit += info.get('profit', 0.0)
            _collect_step(info, accum)

        all_rewards.append(ep_reward)
        all_profits.append(ep_profit)
        _append_ep(accum, bd)
        progress.set_postfix(reward=f"{ep_reward:.2f}", profit=f"{ep_profit:.2f}")
        _log(algo_name, ep, num_episodes, ep_reward, ep_profit)

    return all_rewards, all_profits, bd


def train_off_policy(algo_name, agent, buffer_class, state_dim, action_dim,
                     num_episodes=NUM_EPISODES,
                     warmup_steps=WARMUP_STEPS,
                     batch_size=BATCH_SIZE,
                     update_every=1,
                     grad_steps=1,              # ← SAC 传 2, 其他传 1
                     buffer_capacity=200000):
    env = HydrogenEnv()

    if buffer_class.__name__ == "FastReplayBuffer":
        replay_buffer = buffer_class(
            state_dim, action_dim, capacity=buffer_capacity, device=DEVICE)
    else:
        replay_buffer = buffer_class(capacity=buffer_capacity)

    all_rewards, all_profits = [], []
    bd = _new_bd_lists()
    total_steps = 0

    progress = tqdm(range(num_episodes), desc=f"{algo_name} episodes",
                    unit="ep", leave=False)
    for ep in progress:
        state = env.reset(episode_index=ep)     # ← 改
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
        progress.set_postfix(reward=f"{ep_reward:.2f}", profit=f"{ep_profit:.2f}",
                             steps=total_steps)
        _log(algo_name, ep, num_episodes, ep_reward, ep_profit)

    return all_rewards, all_profits, bd


def moving_average(data, window):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_line_charts(results):
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8

    fig, (ax_r, ax_p) = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    fig.suptitle('Algorithms Comparison',
                 fontsize=13, fontweight='bold')

    converged_rewards = []
    converged_profits = []

    lw = 1.6
    for name in ALGO_NAMES:
        rewards, profits, _ = results[name]
        color = COLORS[name]
        cut_idx = int(len(rewards) * 0.6)
        converged_rewards.extend(rewards[cut_idx:])
        converged_profits.extend(profits[cut_idx:])

        ax_r.plot(rewards, alpha=0.08, color=color, linewidth=0.8)
        ma_r = moving_average(rewards, MA_WINDOW)
        x_r = (np.arange(len(ma_r))
               if len(rewards) < MA_WINDOW
               else np.arange(MA_WINDOW - 1, MA_WINDOW - 1 + len(ma_r)))
        ax_r.plot(x_r, ma_r, color=color, linewidth=lw, label=name)

        ax_p.plot(profits, alpha=0.08, color=color, linewidth=0.8)
        ma_p = moving_average(profits, MA_WINDOW)
        x_p = (np.arange(len(ma_p))
               if len(profits) < MA_WINDOW
               else np.arange(MA_WINDOW - 1, MA_WINDOW - 1 + len(ma_p)))
        ax_p.plot(x_p, ma_p, color=color, linewidth=lw, label=name)

    if converged_rewards:
        ax_r.set_ylim(np.percentile(converged_rewards, 1) - 5,
                      np.percentile(converged_rewards, 99) + 5)
    if converged_profits:
        ax_p.set_ylim(np.percentile(converged_profits, 1) - 500,
                      np.percentile(converged_profits, 99) + 500)

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

    plt.savefig(SCRIPT_DIR / 'Figure_compare_lines.png', dpi=200, bbox_inches='tight')
    plt.show()


def plot_profit_pie(results):
    pie_order = [
        'PPO', 'A2C', 'REINFORCE',
        'SAC', 'TD3', 'DDPG',
        'Random', 'Rule-Service', 'Rule-Profit'
    ]
    positions = [(0, 0), (0, 1), (0, 2),
                 (1, 0), (1, 1), (1, 2),
                 (2, 0), (2, 1), (2, 2)]

    fig = plt.figure(figsize=(15, 14))
    fig.suptitle('Profit Composition by Algorithm',
                 fontsize=13, fontweight='bold', y=0.99)

    gs = fig.add_gridspec(3, 3, hspace=0.22, wspace=0.08,
                          top=0.92, bottom=0.02, left=0.03, right=0.97)

    pie_colors = ['#2ca02c', '#1f77b4', '#d62728']

    for (r, c), name in zip(positions, pie_order):
        ax = fig.add_subplot(gs[r, c])
        _, _, breakdown = results[name]

        n_tail = min(20, len(breakdown['revenue_fcev']))
        avg_fcev    = np.mean(breakdown['revenue_fcev'][-n_tail:])
        avg_fc_sell = np.mean(breakdown['revenue_fc'][-n_tail:]) + \
                      np.mean(breakdown['revenue_grid'][-n_tail:])  # FC卖电 + RE余电，合并为电网售电收入
        avg_cost    = np.mean(breakdown['cost_grid'][-n_tail:])

        net_profit = avg_fcev + avg_fc_sell - avg_cost
        sizes = [avg_fcev, avg_fc_sell, avg_cost]
        total = sum(sizes)

        if total < 0.01:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=11)
            ax.set_title(name, fontsize=11, fontweight='bold', color=COLORS[name])
            ax.axis('off')
            continue

        labels_raw = [
            ('FCEV Revenue', avg_fcev),
            ('FC Revenue',   avg_fc_sell),
            ('Grid Cost',    avg_cost),
        ]
        labels = [f'{lbl}\n${val:,.0f}' for (lbl, val), s in zip(labels_raw, sizes)]

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

    plt.savefig(SCRIPT_DIR / 'Figure_compare_pie.png', dpi=200, bbox_inches='tight')
    plt.show()


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
            'avg_rev_fcev':     np.mean(bd['revenue_fcev'][-n_tail:]),
            'avg_rev_fc_total': np.mean(bd['revenue_fc'][-n_tail:]) +
                                np.mean(bd['revenue_grid'][-n_tail:]),
            'avg_cost_grid':    np.mean(bd['cost_grid'][-n_tail:]),
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
    row('Avg FC Revenue   (L20)',   'avg_rev_fc_total')
    row('Avg Grid Cost    (L20)',   'avg_cost_grid')
    print("=" * sep_len)


def _avg_runs(all_r, all_p, all_bd):
    avg_r = np.mean(all_r, axis=0)
    avg_p = np.mean(all_p, axis=0)
    avg_bd = {
        'revenue_fcev': np.mean(all_bd['revenue_fcev'], axis=0),
        'revenue_fc':   np.mean(all_bd['revenue_fc'],   axis=0),
        'revenue_grid': np.mean(all_bd['revenue_grid'], axis=0),
        'cost_grid':    np.mean(all_bd['cost_grid'],    axis=0),
    }
    return avg_r, avg_p, avg_bd


def main():
    print("=" * 70)
    print("  RL Comparison — Improved (shared scenarios, SAC UTD=2)")
    print("  PPO / A2C / SAC / TD3 / DDPG / REINFORCE / Random / Rule-Based")
    print("=" * 70)
    print(f"  Runs per Algorithm:       {NUM_RUNS}")
    print(f"  Episodes per Algorithm:   {NUM_EPISODES}")
    print(f"  Off-policy warmup:        DDPG/TD3={WARMUP_STEPS}, SAC={SAC_WARMUP_STEPS}")
    print(f"  Batch size:               {BATCH_SIZE}")
    print(f"  Learning rate:            {LR}")
    print(f"  Device:                   {DEVICE}")
    print(f"  SAC: UTD=2, entropy_scale=0.5, hidden=512")
    print(f"  TD3/DDPG: UTD=1, hidden=256 (原始)")
    print(f"  Episode scenarios: deterministic (episode_index)")
    print("=" * 70)

    env_tmp = HydrogenEnv()
    state_dim = env_tmp.observation_space.shape[0]
    action_dim = env_tmp.action_space.shape[0]
    del env_tmp

    results = {}
    times = {}

    # --- 1. PPO ---
    print(f"\n[1/9] Training PPO ({NUM_RUNS} runs × {NUM_EPISODES} episodes)...")
    all_r, all_p, all_bd = [], [], _new_bd_lists()
    t0 = time.time()
    for run in range(NUM_RUNS):
        set_seed(42 + run)
        agent = PPO(state_dim, action_dim, lr=LR, device=DEVICE)
        r, p, bd = train_on_policy('PPO', agent, NUM_EPISODES)
        all_r.append(r); all_p.append(p)
        for k in bd: all_bd[k].append(bd[k])
    times['PPO'] = time.time() - t0
    results['PPO'] = _avg_runs(all_r, all_p, all_bd)
    del agent
    _cleanup_cuda()
    print(f"  PPO done in {times['PPO']:.1f}s")

    # --- 2. A2C ---
    print(f"\n[2/9] Training A2C ({NUM_RUNS} runs × {NUM_EPISODES} episodes)...")
    all_r, all_p, all_bd = [], [], _new_bd_lists()
    t0 = time.time()
    for run in range(NUM_RUNS):
        set_seed(42 + run)
        agent = A2C(state_dim, action_dim, lr=LR, device=DEVICE)
        r, p, bd = train_on_policy('A2C', agent, NUM_EPISODES)
        all_r.append(r); all_p.append(p)
        for k in bd: all_bd[k].append(bd[k])
    times['A2C'] = time.time() - t0
    results['A2C'] = _avg_runs(all_r, all_p, all_bd)
    del agent
    _cleanup_cuda()
    print(f"  A2C done in {times['A2C']:.1f}s")

    # --- 3. SAC (核心: UTD=2, warmup=1000, entropy_scale=0.5) ---
    print(f"\n[3/9] Training SAC ({NUM_RUNS} runs × {NUM_EPISODES} episodes) [UTD=2]...")
    all_r, all_p, all_bd = [], [], _new_bd_lists()
    t0 = time.time()
    for run in range(NUM_RUNS):
        set_seed(42 + run)
        agent = SAC(state_dim, action_dim, lr=LR, device=DEVICE)  # 使用 SAC 默认参数 (已优化)
        r, p, bd = train_off_policy(
            'SAC', agent, FastReplayBuffer, state_dim, action_dim,
            warmup_steps=SAC_WARMUP_STEPS,   # ← 改: 更多 warmup
            grad_steps=2,                     # ← 核心: UTD=2
            buffer_capacity=SAC_BUFFER_CAPACITY)
        all_r.append(r); all_p.append(p)
        for k in bd: all_bd[k].append(bd[k])
    times['SAC'] = time.time() - t0
    results['SAC'] = _avg_runs(all_r, all_p, all_bd)
    del agent
    _cleanup_cuda()
    print(f"  SAC done in {times['SAC']:.1f}s")

    # --- 4. TD3 (保持朴素: UTD=1) ---
    print(f"\n[4/9] Training TD3 ({NUM_RUNS} runs × {NUM_EPISODES} episodes)...")
    all_r, all_p, all_bd = [], [], _new_bd_lists()
    t0 = time.time()
    for run in range(NUM_RUNS):
        set_seed(42 + run)
        agent = TD3(state_dim, action_dim, lr=LR, device=DEVICE)
        r, p, bd = train_off_policy(
            'TD3', agent, StandardReplayBuffer, state_dim, action_dim,
            grad_steps=1)                     # TD3 保持 UTD=1
        all_r.append(r); all_p.append(p)
        for k in bd: all_bd[k].append(bd[k])
    times['TD3'] = time.time() - t0
    results['TD3'] = _avg_runs(all_r, all_p, all_bd)
    del agent
    _cleanup_cuda()
    print(f"  TD3 done in {times['TD3']:.1f}s")

    # --- 5. DDPG (保持朴素: UTD=1) ---
    print(f"\n[5/9] Training DDPG ({NUM_RUNS} runs × {NUM_EPISODES} episodes)...")
    all_r, all_p, all_bd = [], [], _new_bd_lists()
    t0 = time.time()
    for run in range(NUM_RUNS):
        set_seed(42 + run)
        agent = DDPG(state_dim, action_dim, lr=LR, device=DEVICE)
        r, p, bd = train_off_policy(
            'DDPG', agent, StandardReplayBuffer, state_dim, action_dim,
            grad_steps=1)                     # DDPG 保持 UTD=1
        all_r.append(r); all_p.append(p)
        for k in bd: all_bd[k].append(bd[k])
    times['DDPG'] = time.time() - t0
    results['DDPG'] = _avg_runs(all_r, all_p, all_bd)
    del agent
    _cleanup_cuda()
    print(f"  DDPG done in {times['DDPG']:.1f}s")

    # --- 6. REINFORCE ---
    print(f"\n[6/9] Training REINFORCE ({NUM_RUNS} runs × {NUM_EPISODES} episodes)...")
    all_r, all_p, all_bd = [], [], _new_bd_lists()
    t0 = time.time()
    for run in range(NUM_RUNS):
        set_seed(42 + run)
        agent = REINFORCE(state_dim, action_dim, lr=LR, device=DEVICE)
        r, p, bd = train_on_policy('REINFORCE', agent, NUM_EPISODES)
        all_r.append(r); all_p.append(p)
        for k in bd: all_bd[k].append(bd[k])
    times['REINFORCE'] = time.time() - t0
    results['REINFORCE'] = _avg_runs(all_r, all_p, all_bd)
    del agent
    _cleanup_cuda()
    print(f"  REINFORCE done in {times['REINFORCE']:.1f}s")

    # --- 7. Random ---
    print(f"\n[7/9] Running Random baseline ({NUM_RUNS} runs × {NUM_EPISODES} episodes)...")
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

    # --- 8. Rule-Service ---
    print(f"\n[8/9] Evaluating Rule-Service baseline ({NUM_RUNS} runs × {NUM_EPISODES} episodes)...")
    all_r, all_p, all_bd = [], [], _new_bd_lists()
    t0 = time.time()
    for run in range(NUM_RUNS):
        set_seed(42 + run)
        r, p, bd = train_rule_baseline('Rule-Service', 'service', NUM_EPISODES)
        all_r.append(r); all_p.append(p)
        for k in bd: all_bd[k].append(bd[k])
    times['Rule-Service'] = time.time() - t0
    results['Rule-Service'] = _avg_runs(all_r, all_p, all_bd)
    print(f"  Rule-Service done in {times['Rule-Service']:.1f}s")

    # --- 9. Rule-Profit ---
    print(f"\n[9/9] Evaluating Rule-Profit baseline ({NUM_RUNS} runs × {NUM_EPISODES} episodes)...")
    all_r, all_p, all_bd = [], [], _new_bd_lists()
    t0 = time.time()
    for run in range(NUM_RUNS):
        set_seed(42 + run)
        r, p, bd = train_rule_baseline('Rule-Profit', 'profit', NUM_EPISODES)
        all_r.append(r); all_p.append(p)
        for k in bd: all_bd[k].append(bd[k])
    times['Rule-Profit'] = time.time() - t0
    results['Rule-Profit'] = _avg_runs(all_r, all_p, all_bd)
    print(f"  Rule-Profit done in {times['Rule-Profit']:.1f}s")

    time_str = " | ".join(f"{n}={times[n]:.1f}s" for n in ALGO_NAMES)
    print(f"\n  Training Time: {time_str}")

    print_summary(results)
    print("\nGenerating Figure 1: Line Charts ...")
    plot_line_charts(results)
    print("Generating Figure 2: Profit Composition Pie Charts ...")
    plot_profit_pie(results)


if __name__ == "__main__":
    main()