import numpy as np
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from starting_case_env import StartingCaseEnv


def train():
    print("=== 开始训练 Case 1 (7 Days) ===")

    # 使用 8 个并行环境加速
    env = make_vec_env(StartingCaseEnv, n_envs=8, vec_env_cls=DummyVecEnv, seed=42)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        tensorboard_log="./ppo_case1_tensorboard/"
    )

    # 增加训练步数，因为周期变长了 (7天)
    # 建议至少 50万 - 100万步
    model.learn(total_timesteps=800000)

    model.save("ppo_case1_model")
    print("模型已保存。")
    return model


def test(model=None):
    print("\n=== 测试模型 (7 Days) ===")

    if model is None:
        try:
            model = PPO.load("ppo_case1_model")
        except:
            print("未找到模型文件。")
            return

    test_env = StartingCaseEnv()
    obs, _ = test_env.reset(seed=100)

    h_levels = []
    prices = []
    actions = []
    init_h = test_env.initial_level

    done = False
    total_cost = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated

        h_levels.append(info['h_level'])
        prices.append(info['price'])
        actions.append(info['g_t'])
        total_cost += info['cost']

    print(f"初始储量: {init_h:.2f} kg")
    print(f"最终储量: {h_levels[-1]:.2f} kg")
    print(f"总电费成本: {total_cost:.2f}")

    plot_results(h_levels, prices, actions, init_h)


def plot_results(h_levels, prices, actions, init_h):
    t = np.arange(len(h_levels))
    fig, ax1 = plt.subplots(figsize=(14, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Time Step (15 mins/Step)')
    ax1.set_ylabel('Hydrogen Level (kg)', color=color)
    ax1.plot(t, h_levels, color=color, linewidth=1.5, label='H2 Level')
    ax1.axhline(y=init_h, color='green', linestyle='--', linewidth=2, label='Target (Daily I2S)')

    # 画出每天的分界线 (每96步)
    for day in range(1, 8):
        ax1.axvline(x=day * 96, color='gray', linestyle=':', alpha=0.3)
        if day <= 7:
            ax1.text(day * 96 - 48, max(h_levels) * 0.95, f'Day {day}', ha='center', color='gray', fontsize=8)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Electricity Price ($)', color=color)
    # Price 设为透明一点，以免喧宾夺主
    ax2.plot(t, prices, color=color, linestyle='-', alpha=0.3, linewidth=1, label='Price')

    # 绘制产氢动作 (背景柱状图)
    ylim = ax1.get_ylim()
    ax1.bar(t, actions, color='orange', alpha=0.3, width=1.0, label='Production')
    ax1.set_ylim(ylim)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', ncol=3)

    plt.title('Case 1: 7-Day Operation with Real Price Data')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 确保pkl文件在当前目录下
    if not os.path.exists("price_after_MAD_96.pkl"):
        print("Warning: 'price_after_MAD_96.pkl' not found. Using synthetic data.")

    trained_model = train()

    test(trained_model)
