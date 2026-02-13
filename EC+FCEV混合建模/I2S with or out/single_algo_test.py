"""
SAC (无 Transformer) 算法 - 有/无 I2S 约束对比测试

功能:
- 使用 SAC 算法 (use_transformer=False)，对比 I2S 开启与关闭两种条件下的训练效果
- 输出 2×2 图: With I2S Reward/Profit | Without I2S Reward/Profit
- 可生成论文用对比图

使用方法:
    cd "I2S with or out"
    python single_algo_test.py
"""

import numpy as np
import matplotlib.pyplot as plt
from env import HydrogenEnv
from SAC import SAC
from SAC import ReplayBuffer
from config import Config


# ======================== 配置 ========================
NUM_EPISODES = 100       # 每种条件下的训练 Episode 数
WARMUP_STEPS = 500      # 随机探索步数
BATCH_SIZE = 256
LR = 3e-4
MA_WINDOW = 20


def train_sac_single(enable_i2s, num_episodes=NUM_EPISODES):
    """在指定 I2S 条件下训练 SAC (无 Transformer)"""
    env = HydrogenEnv(enable_i2s_constraint=enable_i2s)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SAC(
        state_dim, action_dim,
        lr=LR,
        use_transformer=False,  # 不使用 Transformer
    )
    replay_buffer = ReplayBuffer(capacity=100000)

    all_rewards = []
    all_profits = []
    total_steps = 0

    label = "With I2S" if enable_i2s else "Without I2S"
    print(f"\nTraining SAC (no Transformer) [{label}] - {num_episodes} episodes...")

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
            print(f"  Episode {ep+1:>3d}/{num_episodes}, Reward: {ep_reward:>8.2f}, Profit: ${ep_profit:>10.2f}")

    return all_rewards, all_profits


def moving_average(data, window):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_sac_i2s_comparison(results_with, results_without):
    """绘制 SAC 有/无 I2S 对比图"""
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["axes.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9

    fig, axs = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("SAC (no Transformer) Comparison: With vs Without I2S Constraint",
                 fontsize=12, fontweight="bold")

    rewards_with, profits_with = results_with
    rewards_wo, profits_wo = results_without

    color_with = "#2ca02c"   # 绿色 - With I2S
    color_wo = "#d62728"     # 红色 - Without I2S

    # With I2S: Reward
    ax = axs[0, 0]
    ax.plot(rewards_with, alpha=0.15, color=color_with, linewidth=0.8)
    ma = moving_average(rewards_with, MA_WINDOW)
    ax.plot(range(MA_WINDOW - 1, len(rewards_with)), ma, color=color_with, linewidth=2, label=f"With I2S (MA{MA_WINDOW})")
    ax.set_title("With I2S: Episode Reward")
    ax.set_ylabel("Episode Reward")
    ax.set_xlabel("Episode")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")

    # With I2S: Profit
    ax = axs[0, 1]
    ax.plot(profits_with, alpha=0.15, color=color_with, linewidth=0.8)
    ma = moving_average(profits_with, MA_WINDOW)
    ax.plot(range(MA_WINDOW - 1, len(profits_with)), ma, color=color_with, linewidth=2, label=f"With I2S (MA{MA_WINDOW})")
    ax.set_title("With I2S: Episode Profit")
    ax.set_ylabel("Episode Profit ($)")
    ax.set_xlabel("Episode")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")

    # Without I2S: Reward
    ax = axs[1, 0]
    ax.plot(rewards_wo, alpha=0.15, color=color_wo, linewidth=0.8)
    ma = moving_average(rewards_wo, MA_WINDOW)
    ax.plot(range(MA_WINDOW - 1, len(rewards_wo)), ma, color=color_wo, linewidth=2, label=f"Without I2S (MA{MA_WINDOW})")
    ax.set_title("Without I2S: Episode Reward")
    ax.set_ylabel("Episode Reward")
    ax.set_xlabel("Episode")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")

    # Without I2S: Profit
    ax = axs[1, 1]
    ax.plot(profits_wo, alpha=0.15, color=color_wo, linewidth=0.8)
    ma = moving_average(profits_wo, MA_WINDOW)
    ax.plot(range(MA_WINDOW - 1, len(profits_wo)), ma, color=color_wo, linewidth=2, label=f"Without I2S (MA{MA_WINDOW})")
    ax.set_title("Without I2S: Episode Profit")
    ax.set_ylabel("Episode Profit ($)")
    ax.set_xlabel("Episode")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.show()


def print_summary(results_with, results_without):
    """打印对比摘要"""
    rewards_with, profits_with = results_with
    rewards_wo, profits_wo = results_without

    n = min(20, len(profits_with), len(profits_wo))
    avg_profit_with = np.mean(profits_with[-n:]) if len(profits_with) >= n else np.mean(profits_with)
    avg_profit_wo = np.mean(profits_wo[-n:]) if len(profits_wo) >= n else np.mean(profits_wo)
    avg_reward_with = np.mean(rewards_with[-n:]) if len(rewards_with) >= n else np.mean(rewards_with)
    avg_reward_wo = np.mean(rewards_wo[-n:]) if len(rewards_wo) >= n else np.mean(rewards_wo)

    print("\n" + "=" * 60)
    print("  SAC (no Transformer) I2S Comparison Summary")
    print("=" * 60)
    print(f"{'Metric':<30} {'With I2S':>15} {'Without I2S':>15}")
    print("-" * 60)
    print(f"{'Avg Reward (Last 20 Ep)':<30} {avg_reward_with:>15.2f} {avg_reward_wo:>15.2f}")
    print(f"{'Avg Profit (Last 20 Ep)':<30} {avg_profit_with:>15.2f} {avg_profit_wo:>15.2f}")
    print("=" * 60)


def main():
    print("=" * 70)
    print("  SAC (no Transformer) - I2S Constraint Comparison Test")
    print("=" * 70)
    print(f"  Episodes per condition: {NUM_EPISODES}")
    print(f"  Warmup steps:            {WARMUP_STEPS}")
    print(f"  Learning rate:           {LR}")
    env_tmp = HydrogenEnv()
    print(f"  Action dimension:        {env_tmp.action_space.shape[0]} (v3.6: ele, fc, comp_load, cooling, battery, bypass, c3_pressure, chiller)")
    del env_tmp
    print("=" * 70)

    # 1. With I2S
    results_with = train_sac_single(enable_i2s=True, num_episodes=NUM_EPISODES)

    # 2. Without I2S
    results_without = train_sac_single(enable_i2s=False, num_episodes=NUM_EPISODES)

    # 3. 打印摘要
    print_summary(results_with, results_without)

    # 4. 绘图
    print("\nGenerating comparison plot...")
    plot_sac_i2s_comparison(results_with, results_without)


if __name__ == "__main__":
    main()
