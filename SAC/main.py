import numpy as np
import matplotlib.pyplot as plt
from env import HydrogenEnv
from sac import SAC, ReplayBuffer
from config import Config


def plot_results(rewards, profits, soc_history, power_balance):
    plt.rcParams['axes.unicode_minus'] = False

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Training Reward
    axs[0, 0].plot(rewards)
    axs[0, 0].set_title("Training Reward Curve (Clipped Gradient)")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Total Reward")

    # 2. Cumulative Profit
    cum_profit = np.cumsum(profits)
    axs[0, 1].plot(cum_profit, color='green')
    axs[0, 1].set_title("Cumulative Profit (Last Episode)")
    axs[0, 1].set_ylabel("Profit ($)")

    # 3. SOC History
    axs[1, 0].plot(soc_history, color='orange', label='SOC')
    axs[1, 0].axhline(y=Config.storage_initial, color='r', linestyle='--', label='Initial Target')
    axs[1, 0].set_title("H2 Storage Level (Should Match Target)")
    axs[1, 0].set_ylim(0, 1.0)
    axs[1, 0].legend()

    # 4. Energy Balance
    steps = range(len(power_balance['re']))
    re_gen = np.array(power_balance['re'])
    grid_power = np.array(power_balance['net'])
    fc_power = np.array(power_balance['fc'])
    load_power = np.array(power_balance['load'])

    axs[1, 1].plot(steps, re_gen, label='RE Gen (+)', alpha=0.5)
    axs[1, 1].plot(steps, fc_power, label='Fuel Cell (+)', color='purple', linewidth=2)
    axs[1, 1].plot(steps, -load_power, label='Total Load (-)', color='red', alpha=0.5)

    axs[1, 1].fill_between(steps, grid_power, 0, where=(grid_power < 0), color='gray', alpha=0.3,
                           label='Grid Import (-)')
    axs[1, 1].fill_between(steps, grid_power, 0, where=(grid_power > 0), color='green', alpha=0.3,
                           label='Grid Export (+)')

    axs[1, 1].set_title("Energy Balance (Prod+Comp+Chill)")
    axs[1, 1].legend(loc='upper right', fontsize='small')

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

    print("Start Training Hydrogen Project (SAC + Gradient Clipping)...")

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        if episode == num_episodes - 1:
            last_episode_profits = []
            last_episode_soc = [state[0]]
            last_episode_power = {'re': [], 'net': [], 'fc': [], 'load': []}

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

        all_rewards.append(episode_reward)
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Final SOC: {state[0]:.2f}")

    plot_results(all_rewards, last_episode_profits, last_episode_soc, last_episode_power)


if __name__ == "__main__":
    train()