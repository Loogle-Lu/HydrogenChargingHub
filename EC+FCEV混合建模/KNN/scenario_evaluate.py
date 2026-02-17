"""
Scenario-based Reliability Analysis (v3.7)

四条件对比: SAC Transformer/I2S 组合 × 6 场景
    1. SAC + Transformer + I2S
    2. SAC + Transformer - I2S
    3. SAC - Transformer + I2S
    4. SAC - Transformer - I2S

每个条件独立训练，然后在 6 个典型场景下评估可靠性。
输出 4 张 Dashboard 图 (每条件 1 张)。

使用方法:
    cd KNN
    python scenario_evaluate.py
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from env import HydrogenEnv
from SAC import SAC, ReplayBuffer, SequenceReplayBuffer
from scenario_manager import ScenarioManager, SCENARIO_NAMES
from data_loader import DataLoader


# ======================== 配置 ========================
TRAIN_EPISODES = 100      # 训练 Episode 数
EVAL_EPISODES = 10        # 每场景评估 Episode 数
WARMUP_STEPS = 500        # Off-Policy 热身步数
BATCH_SIZE = 256
LR = 3e-4

# 四个对比条件: (use_transformer, enable_i2s, label)
CONDITIONS = [
    (True,  True,  "SAC + Transformer + I2S"),
    (True,  False, "SAC + Transformer - I2S"),
    (False, True,  "SAC - Transformer + I2S"),
    (False, False, "SAC - Transformer - I2S"),
]

# 5 算法用于场景内对比 (每个条件下只训练 SAC，但保留多算法颜色框架)
# 这里用于 Dashboard 内显示场景名颜色
SCENE_COLORS = {
    "normal":        '#1f77b4',
    "high_price":    '#ff7f0e',
    "low_renewable": '#2ca02c',
    "high_demand":   '#d62728',
    "ideal":         '#9467bd',
    "extreme":       '#8c564b',
}


# ======================== 训练函数 ========================

def train_sac(use_transformer, enable_i2s, state_dim, action_dim,
              num_episodes=TRAIN_EPISODES):
    """训练 SAC 并返回 (agent, reward_history, profit_history)"""
    env = HydrogenEnv(enable_i2s_constraint=enable_i2s)
    agent = SAC(state_dim, action_dim, lr=LR, use_transformer=use_transformer)

    if use_transformer:
        replay_buffer = SequenceReplayBuffer(capacity=100000)
    else:
        replay_buffer = ReplayBuffer(capacity=100000)

    total_steps = 0
    all_rewards = []
    all_profits = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        ep_profit = 0

        if use_transformer:
            agent.reset_state_buffer()
            agent.append_state(state)

        while not done:
            if use_transformer:
                state_seq = agent.get_state_seq()
                if total_steps < WARMUP_STEPS:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action_from_seq(state_seq, evaluate=False)
                next_state, reward, done, info = env.step(action)
                agent.append_state(next_state)
                next_state_seq = agent.get_state_seq()
                replay_buffer.push(state_seq, action, reward, next_state_seq, float(done))
                if total_steps >= WARMUP_STEPS and len(replay_buffer) >= BATCH_SIZE:
                    agent.update(replay_buffer, BATCH_SIZE)
            else:
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
            total_steps += 1

        all_rewards.append(ep_reward)
        all_profits.append(ep_profit)

        if (ep + 1) % 20 == 0:
            tag = "T" if use_transformer else "M"
            i2s_tag = "I2S" if enable_i2s else "noI2S"
            print(f"    SAC-{tag}-{i2s_tag}  Ep {ep+1:>3d}/{num_episodes}, "
                  f"Reward: {ep_reward:>8.2f}, Profit: {ep_profit:>10.2f}")

    return agent, all_rewards, all_profits


# ======================== 场景评估函数 ========================

def evaluate_on_scenarios(agent, scenario_configs, enable_i2s,
                          num_episodes=EVAL_EPISODES):
    """
    在所有场景下评估 agent。

    Returns
    -------
    dict[scenario_name] = {"rewards": [...], "profits": [...]}
    """
    results = {}
    for scene_name, cfg in scenario_configs.items():
        env = HydrogenEnv(enable_i2s_constraint=enable_i2s)
        scene_rewards = []
        scene_profits = []

        for ep in range(num_episodes):
            state = env.reset_with_scenario(cfg)
            done = False
            ep_r, ep_p = 0.0, 0.0
            while not done:
                action = agent.select_action(state, evaluate=True)
                next_state, reward, done, info = env.step(action)
                state = next_state
                ep_r += reward
                ep_p += info.get("profit", 0.0)
            scene_rewards.append(ep_r)
            scene_profits.append(ep_p)

        results[scene_name] = {
            "rewards": scene_rewards,
            "profits": scene_profits,
        }
    return results


# ======================== Dashboard 可视化 ========================

def plot_dashboard(eval_results, condition_label):
    """
    绘制单个条件的 2x2 Dashboard:
        (0,0) Reward Heatmap  (场景 × 指标)
        (0,1) Profit Heatmap
        (1,0) Reward Radar
        (1,1) Profit Bar Chart

    因为只有 1 个算法 (SAC)，热力图退化为 1 行 ×6 列场景条。
    雷达图只有 1 条线。柱状图只有 1 组柱。
    """
    n_scene = len(SCENARIO_NAMES)

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(condition_label, fontsize=16, fontweight='bold', y=1.00)

    # =================== (0,0) Reward Heatmap ===================
    ax1 = fig.add_subplot(2, 2, 1)
    reward_vals = [np.mean(eval_results[s]["rewards"]) for s in SCENARIO_NAMES]
    matrix_r = np.array(reward_vals).reshape(1, -1)

    im1 = ax1.imshow(matrix_r, cmap='RdYlGn', aspect='auto')
    ax1.set_xticks(range(n_scene))
    ax1.set_xticklabels([s.replace("_", "\n") for s in SCENARIO_NAMES], fontsize=8)
    ax1.set_yticks([0])
    ax1.set_yticklabels(["SAC"], fontsize=10)
    for j in range(n_scene):
        ax1.text(j, 0, f'{reward_vals[j]:.1f}', ha='center', va='center',
                 fontsize=9, fontweight='bold',
                 color='white' if abs(reward_vals[j]) > np.mean(np.abs(reward_vals)) else 'black')
    cb1 = plt.colorbar(im1, ax=ax1, shrink=0.6, pad=0.02)
    cb1.ax.tick_params(labelsize=7)
    ax1.set_title('Average Reward by Scenario', fontsize=11, fontweight='bold', pad=8)

    # =================== (0,1) Profit Heatmap ===================
    ax2 = fig.add_subplot(2, 2, 2)
    profit_vals = [np.mean(eval_results[s]["profits"]) for s in SCENARIO_NAMES]
    matrix_p = np.array(profit_vals).reshape(1, -1)

    im2 = ax2.imshow(matrix_p, cmap='RdYlGn', aspect='auto')
    ax2.set_xticks(range(n_scene))
    ax2.set_xticklabels([s.replace("_", "\n") for s in SCENARIO_NAMES], fontsize=8)
    ax2.set_yticks([0])
    ax2.set_yticklabels(["SAC"], fontsize=10)
    for j in range(n_scene):
        ax2.text(j, 0, f'{profit_vals[j]:.0f}', ha='center', va='center',
                 fontsize=9, fontweight='bold',
                 color='white' if profit_vals[j] < np.mean(profit_vals) * 0.5 else 'black')
    cb2 = plt.colorbar(im2, ax=ax2, shrink=0.6, pad=0.02)
    cb2.ax.tick_params(labelsize=7)
    ax2.set_title('Average Profit by Scenario ($)', fontsize=11, fontweight='bold', pad=8)

    # =================== (1,0) Reward Radar ===================
    ax3 = fig.add_subplot(2, 2, 3, polar=True)
    angles = np.linspace(0, 2 * np.pi, n_scene, endpoint=False).tolist()
    angles += angles[:1]

    r_min = min(reward_vals)
    r_max = max(reward_vals)
    r_range = max(r_max - r_min, 1e-6)
    radar_vals = [(v - r_min) / r_range for v in reward_vals]
    radar_vals += radar_vals[:1]

    ax3.plot(angles, radar_vals, 'o-', linewidth=2.0, color='#ff7f0e', markersize=6)
    ax3.fill(angles, radar_vals, alpha=0.15, color='#ff7f0e')
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels([s.replace("_", "\n") for s in SCENARIO_NAMES], fontsize=8)
    ax3.set_ylim(0, 1.15)
    ax3.set_title('Reward Profile (Radar)', fontsize=11, fontweight='bold', pad=15)

    # 在雷达图上标注原始数值
    for i, (angle, val, raw) in enumerate(zip(angles[:-1], radar_vals[:-1], reward_vals)):
        ax3.annotate(f'{raw:.0f}', xy=(angle, val), fontsize=7,
                     ha='center', va='bottom', color='#d62728', fontweight='bold')

    # =================== (1,1) Profit Bar Chart ===================
    ax4 = fig.add_subplot(2, 2, 4)
    x = np.arange(n_scene)
    bar_colors = [SCENE_COLORS[s] for s in SCENARIO_NAMES]
    stds = [np.std(eval_results[s]["profits"]) for s in SCENARIO_NAMES]

    bars = ax4.bar(x, profit_vals, width=0.6, color=bar_colors, alpha=0.85,
                   yerr=stds, capsize=4, edgecolor='white', linewidth=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels([s.replace("_", " ").title() for s in SCENARIO_NAMES],
                        fontsize=8, rotation=15, ha='right')
    ax4.set_ylabel('Average Profit ($)', fontsize=10)
    ax4.set_title('Profit by Scenario (Bar Chart)', fontsize=11, fontweight='bold', pad=8)
    ax4.grid(True, alpha=0.3, linestyle='--', axis='y')

    # 在柱上标数值
    for bar, val in zip(bars, profit_vals):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                 f'${val:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.subplots_adjust(top=0.92, bottom=0.06, left=0.06, right=0.96,
                        hspace=0.35, wspace=0.30)
    plt.show()


# ======================== 统计表格 ========================

def print_condition_summary(condition_label, eval_results):
    """打印单个条件的场景评估汇总"""
    col_w = 14
    sep_len = 20 + col_w * len(SCENARIO_NAMES)

    print("\n" + "=" * sep_len)
    print(f"  {condition_label}")
    print("=" * sep_len)
    header = f"{'Metric':<20s}" + "".join(f"{s:>{col_w}s}" for s in SCENARIO_NAMES)
    print(header)
    print("-" * sep_len)

    # Reward 行
    line = f"{'Avg Reward':<20s}"
    for s in SCENARIO_NAMES:
        line += f"{np.mean(eval_results[s]['rewards']):{col_w}.2f}"
    print(line)

    # Reward Std 行
    line = f"{'Std Reward':<20s}"
    for s in SCENARIO_NAMES:
        line += f"{np.std(eval_results[s]['rewards']):{col_w}.2f}"
    print(line)

    print("-" * sep_len)

    # Profit 行
    line = f"{'Avg Profit ($)':<20s}"
    for s in SCENARIO_NAMES:
        line += f"{np.mean(eval_results[s]['profits']):{col_w}.1f}"
    print(line)

    # Profit Std 行
    line = f"{'Std Profit ($)':<20s}"
    for s in SCENARIO_NAMES:
        line += f"{np.std(eval_results[s]['profits']):{col_w}.1f}"
    print(line)

    # 跨场景统计
    print("-" * sep_len)
    r_means = [np.mean(eval_results[s]["rewards"]) for s in SCENARIO_NAMES]
    p_means = [np.mean(eval_results[s]["profits"]) for s in SCENARIO_NAMES]
    r_cv = np.std(r_means) / max(abs(np.mean(r_means)), 1e-6)
    p_cv = np.std(p_means) / max(abs(np.mean(p_means)), 1e-6)
    stability = "Stable" if r_cv < 0.3 else "Moderate" if r_cv < 0.6 else "Unstable"

    print(f"  Cross-Scenario Avg Reward:  {np.mean(r_means):>10.2f}")
    print(f"  Cross-Scenario Avg Profit:  {np.mean(p_means):>10.1f}")
    print(f"  Reward CV (lower=stable):   {r_cv:>10.4f}  ({stability})")
    print(f"  Profit CV (lower=stable):   {p_cv:>10.4f}")
    print("=" * sep_len)


def print_cross_condition_comparison(all_condition_results):
    """打印四条件横向对比表"""
    print("\n" + "=" * 80)
    print("  CROSS-CONDITION COMPARISON (4 SAC Variants)")
    print("=" * 80)

    col_w = 16
    header = f"{'Metric':<24s}" + "".join(f"{c[2]:>{col_w}s}" for c in CONDITIONS)
    print(header)
    print("-" * (24 + col_w * len(CONDITIONS)))

    for metric_name, metric_key, fmt in [
        ("Avg Reward (all)", "rewards", ".2f"),
        ("Avg Profit (all)", "profits", ".1f"),
    ]:
        line = f"{metric_name:<24s}"
        for _, _, label in CONDITIONS:
            res = all_condition_results[label]
            all_vals = []
            for s in SCENARIO_NAMES:
                all_vals.extend(res[s][metric_key])
            line += f"{np.mean(all_vals):{col_w}{fmt}}"
        print(line)

    # CV 行
    line = f"{'Reward CV':<24s}"
    for _, _, label in CONDITIONS:
        res = all_condition_results[label]
        r_means = [np.mean(res[s]["rewards"]) for s in SCENARIO_NAMES]
        cv = np.std(r_means) / max(abs(np.mean(r_means)), 1e-6)
        line += f"{cv:{col_w}.4f}"
    print(line)

    line = f"{'Stability':<24s}"
    for _, _, label in CONDITIONS:
        res = all_condition_results[label]
        r_means = [np.mean(res[s]["rewards"]) for s in SCENARIO_NAMES]
        cv = np.std(r_means) / max(abs(np.mean(r_means)), 1e-6)
        tag = "Stable" if cv < 0.3 else "Moderate" if cv < 0.6 else "Unstable"
        line += f"{tag:>{col_w}s}"
    print(line)
    print("=" * (24 + col_w * len(CONDITIONS)))


# ======================== 主函数 ========================

def main():
    print("=" * 70)
    print("  SCENARIO-BASED RELIABILITY ANALYSIS (v3.7)")
    print("  SAC Transformer/I2S 4-Condition x 6-Scenario Evaluation")
    print("=" * 70)
    print(f"  Conditions:     {len(CONDITIONS)}")
    for _, _, label in CONDITIONS:
        print(f"    - {label}")
    print(f"  Train Episodes: {TRAIN_EPISODES}")
    print(f"  Eval Episodes:  {EVAL_EPISODES} per scenario")
    print(f"  Scenarios:      {len(SCENARIO_NAMES)}")
    print("=" * 70)

    # --- Phase 1: 初始化场景管理器 ---
    print("\n[Phase 1] Initializing Scenario Manager (KNN feature extraction)...")
    dl = DataLoader()
    sm = ScenarioManager(dl)
    sm.print_scenario_features()
    scenario_configs = sm.get_all_scenarios()

    # 获取环境维度
    env_tmp = HydrogenEnv()
    state_dim = env_tmp.observation_space.shape[0]
    action_dim = env_tmp.action_space.shape[0]
    del env_tmp

    # --- Phase 2+3: 训练 & 评估 (每个条件) ---
    all_condition_results = {}  # label -> {scene -> {rewards, profits}}

    for idx, (use_tf, enable_i2s, label) in enumerate(CONDITIONS):
        print("\n" + "=" * 70)
        print(f"  [{idx+1}/{len(CONDITIONS)}] {label}")
        print("=" * 70)

        # 训练
        print(f"\n  Training SAC (Transformer={'Yes' if use_tf else 'No'}, "
              f"I2S={'Yes' if enable_i2s else 'No'})...")
        t0 = time.time()
        agent, train_r, train_p = train_sac(use_tf, enable_i2s,
                                             state_dim, action_dim, TRAIN_EPISODES)
        elapsed = time.time() - t0
        print(f"  Training done in {elapsed:.1f}s")
        print(f"  Final 20-ep avg reward: {np.mean(train_r[-20:]):.2f}, "
              f"profit: {np.mean(train_p[-20:]):.1f}")

        # 评估
        print(f"\n  Evaluating on {len(SCENARIO_NAMES)} scenarios "
              f"({EVAL_EPISODES} eps each)...")
        eval_results = evaluate_on_scenarios(agent, scenario_configs,
                                              enable_i2s, EVAL_EPISODES)
        all_condition_results[label] = eval_results

        for scene in SCENARIO_NAMES:
            avg_r = np.mean(eval_results[scene]["rewards"])
            avg_p = np.mean(eval_results[scene]["profits"])
            print(f"    {scene:<16s} | Reward: {avg_r:>8.2f}, Profit: {avg_p:>10.1f}")

        # 打印条件摘要
        print_condition_summary(label, eval_results)

    # --- Phase 4: 横向对比 + 可视化 ---
    print("\n[Phase 4] Cross-Condition Comparison & Visualization...")
    print_cross_condition_comparison(all_condition_results)

    # 为每个条件生成 Dashboard
    for _, _, label in CONDITIONS:
        print(f"\n  Generating Dashboard: {label}")
        plot_dashboard(all_condition_results[label], label)

    print("\n" + "=" * 70)
    print("  Scenario-based Reliability Analysis Complete!")
    print("  4 Dashboard figures generated.")
    print("=" * 70)


if __name__ == "__main__":
    main()
