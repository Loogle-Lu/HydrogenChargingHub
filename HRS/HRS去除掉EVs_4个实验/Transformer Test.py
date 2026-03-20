"""
SAC vs SAC_Transformer 对比测试 (With I2S 条件)

功能:
- 在 With I2S 条件下比较 SAC(MLP) 与 SAC(Transformer) 的训练效果
- 输出 2 张曲线图: Reward 对比 | Profit 对比

使用方法:
    cd "I2S with or out"
    python single_algo_test.py
"""

import numpy as np
import matplotlib.pyplot as plt
from env import HydrogenEnv
from SAC import SAC, ReplayBuffer
from SAC_trans import SAC_Transformer, SequenceReplayBuffer


# ======================== 配置 ========================
NUM_EPISODES = 100       # 每种算法训练的 Episode 数
WARMUP_STEPS = 500       # 随机探索步数
BATCH_SIZE = 256
LR = 3e-4
MA_WINDOW = 20


def train_sac_mlp(num_episodes=NUM_EPISODES):
    """训练 SAC (MLP 版)"""
    env = HydrogenEnv(enable_i2s_constraint=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SAC(state_dim, action_dim, lr=LR)
    replay_buffer = ReplayBuffer(capacity=100000)

    all_rewards = []
    all_profits = []
    total_steps = 0

    print(f"\nTraining SAC (MLP) [With I2S] - {num_episodes} episodes...")

    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0
        ep_profit = 0
        done = False

        while not done:
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
            print(f"  SAC(MLP) Episode {ep+1:>3d}/{num_episodes}, Reward: {ep_reward:>8.2f}, Profit: ${ep_profit:>10.2f}")

    return all_rewards, all_profits


def train_sac_transformer(num_episodes=NUM_EPISODES):
    """训练 SAC (Transformer 版)"""
    env = HydrogenEnv(enable_i2s_constraint=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SAC_Transformer(state_dim, action_dim, lr=LR)
    replay_buffer = SequenceReplayBuffer(capacity=100000)

    all_rewards = []
    all_profits = []
    total_steps = 0

    print(f"\nTraining SAC (Transformer) [With I2S] - {num_episodes} episodes...")

    for ep in range(num_episodes):
        state = env.reset()
        agent.reset_state_buffer()
        agent.append_state(state)
        ep_reward = 0
        ep_profit = 0
        done = False

        while not done:
            state_seq = agent.get_state_seq()
            if total_steps < WARMUP_STEPS:
                action = env.action_space.sample()
            else:
                action = agent._select_action_from_seq(state_seq, evaluate=False)

            next_state, reward, done, info = env.step(action)
            agent.append_state(next_state)
            next_state_seq = agent.get_state_seq()
            replay_buffer.push(state_seq, action, reward, next_state_seq, float(done))

            if total_steps >= WARMUP_STEPS and len(replay_buffer) >= BATCH_SIZE:
                agent.update(replay_buffer, BATCH_SIZE)

            state = next_state
            ep_reward += reward
            ep_profit += info.get("profit", 0.0)
            total_steps += 1

        all_rewards.append(ep_reward)
        all_profits.append(ep_profit)

        if (ep + 1) % 20 == 0:
            print(f"  SAC(Trans) Episode {ep+1:>3d}/{num_episodes}, Reward: {ep_reward:>8.2f}, Profit: ${ep_profit:>10.2f}")

    return all_rewards, all_profits


def moving_average(data, window):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_sac_comparison(rewards_mlp, profits_mlp, rewards_trans, profits_trans):
    """绘制 SAC vs SAC_Transformer 两张曲线图"""
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["axes.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9

    color_mlp = "#1f77b4"   # 蓝色 - SAC (MLP)
    color_trans = "#ff7f0e" # 橙色 - SAC (Transformer)

    # 图1: Reward 曲线
    fig1, ax1 = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax1.plot(rewards_mlp, alpha=0.15, color=color_mlp, linewidth=0.8)
    ma_mlp = moving_average(rewards_mlp, MA_WINDOW)
    ax1.plot(range(MA_WINDOW - 1, len(rewards_mlp)), ma_mlp, color=color_mlp, linewidth=2, label=f"SAC (MLP) MA{MA_WINDOW}")

    ax1.plot(rewards_trans, alpha=0.15, color=color_trans, linewidth=0.8)
    ma_trans = moving_average(rewards_trans, MA_WINDOW)
    ax1.plot(range(MA_WINDOW - 1, len(rewards_trans)), ma_trans, color=color_trans, linewidth=2, label=f"SAC (Transformer) MA{MA_WINDOW}")

    ax1.set_title("SAC vs SAC (Transformer) - Episode Reward (With I2S)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Episode Reward")
    ax1.set_xlabel("Episode")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3, linestyle="--")
    plt.show()

    # 图2: Profit 曲线
    fig2, ax2 = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax2.plot(profits_mlp, alpha=0.15, color=color_mlp, linewidth=0.8)
    ma_mlp = moving_average(profits_mlp, MA_WINDOW)
    ax2.plot(range(MA_WINDOW - 1, len(profits_mlp)), ma_mlp, color=color_mlp, linewidth=2, label=f"SAC (MLP) MA{MA_WINDOW}")

    ax2.plot(profits_trans, alpha=0.15, color=color_trans, linewidth=0.8)
    ma_trans = moving_average(profits_trans, MA_WINDOW)
    ax2.plot(range(MA_WINDOW - 1, len(profits_trans)), ma_trans, color=color_trans, linewidth=2, label=f"SAC (Transformer) MA{MA_WINDOW}")

    ax2.set_title("SAC vs SAC (Transformer) - Episode Profit (With I2S)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Episode Profit ($)")
    ax2.set_xlabel("Episode")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3, linestyle="--")
    plt.show()


def print_summary(rewards_mlp, profits_mlp, rewards_trans, profits_trans):
    """打印对比摘要"""
    n = min(20, len(profits_mlp), len(profits_trans))
    avg_rew_mlp = np.mean(rewards_mlp[-n:]) if len(rewards_mlp) >= n else np.mean(rewards_mlp)
    avg_rew_trans = np.mean(rewards_trans[-n:]) if len(rewards_trans) >= n else np.mean(rewards_trans)
    avg_prof_mlp = np.mean(profits_mlp[-n:]) if len(profits_mlp) >= n else np.mean(profits_mlp)
    avg_prof_trans = np.mean(profits_trans[-n:]) if len(profits_trans) >= n else np.mean(profits_trans)

    print("\n" + "=" * 60)
    print("  SAC vs SAC (Transformer) Summary (With I2S)")
    print("=" * 60)
    print(f"{'Metric':<30} {'SAC (MLP)':>15} {'SAC (Transformer)':>18}")
    print("-" * 60)
    print(f"{'Avg Reward (Last 20 Ep)':<30} {avg_rew_mlp:>15.2f} {avg_rew_trans:>18.2f}")
    print(f"{'Avg Profit (Last 20 Ep)':<30} {avg_prof_mlp:>15.2f} {avg_prof_trans:>18.2f}")
    print("=" * 60)


def main():
    print("=" * 70)
    print("  SAC vs SAC (Transformer) - With I2S Comparison")
    print("=" * 70)
    print(f"  Episodes per algorithm: {NUM_EPISODES}")
    print(f"  Warmup steps:           {WARMUP_STEPS}")
    print(f"  Learning rate:           {LR}")
    env_tmp = HydrogenEnv()
    print(f"  Action dimension:        {env_tmp.action_space.shape[0]}")
    del env_tmp
    print("=" * 70)

    # 1. 训练 SAC (MLP)
    rewards_mlp, profits_mlp = train_sac_mlp(NUM_EPISODES)

    # 2. 训练 SAC (Transformer)
    rewards_trans, profits_trans = train_sac_transformer(NUM_EPISODES)

    # 3. 打印摘要
    print_summary(rewards_mlp, profits_mlp, rewards_trans, profits_trans)

    # 4. 绘图 (两张曲线图)
    print("\nGenerating comparison plots...")
    plot_sac_comparison(rewards_mlp, profits_mlp, rewards_trans, profits_trans)


if __name__ == "__main__":
    main()
