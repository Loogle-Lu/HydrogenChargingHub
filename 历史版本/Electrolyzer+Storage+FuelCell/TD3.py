import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from google.colab import drive

# ==========================================
# 1. 自动挂载 Drive 并读取数据
# ==========================================
print("正在初始化... 尝试挂载 Google Drive 读取价格数据...")

try:
    drive.mount('/content/drive')

    # 定义可能的文件路径 (根据你的描述)
    possible_paths = [
        "/content/drive/MyDrive/price_after_MAD_96.pkl",
        "/content/drive/MyDrive/RL_environment/price_after_MAD_96.pkl",
        "/content/drive/MyDrive/RL_environment/electrolyzer-storage-fuel-cell/price_after_MAD_96.pkl",
        "/content/drive/MyDrive/data/price_after_MAD_96.pkl",
        "./price_after_MAD_96.pkl" # 或者是当前目录
    ]

    PRICE_DATA = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ 成功找到文件: {path}")
            try:
                with open(path, 'rb') as f:
                    PRICE_DATA = np.array(pickle.load(f))
                print(f"   -> 数据加载完毕，长度: {len(PRICE_DATA)}")
                break
            except Exception as e:
                print(f"   -> 读取出错: {e}")

    if PRICE_DATA is None:
        print("⚠️ 警告: 未在常见路径找到文件，将使用合成的正弦波数据。")
        t = np.linspace(0, 2*np.pi, 96)
        PRICE_DATA = np.clip(20 + 10*np.sin(t), 1, 50)

except Exception as e:
    print(f"无法挂载 Drive 或读取出错: {e}")
    print("使用合成数据继续...")
    t = np.linspace(0, 2*np.pi, 96)
    PRICE_DATA = np.clip(20 + 10*np.sin(t), 1, 50)

# ==========================================
# 2. 组件模型 (Components)
# ==========================================

class Electrolyzer:
    def __init__(self, g_max, eta):
        self.g_max = g_max
        self.eta = eta

    def produce(self, power_input):
        g_t = np.clip(power_input, 0, self.g_max)
        h2_produced = self.eta * g_t
        return h2_produced, g_t

class FuelCell:
    def __init__(self, f_max, eta):
        self.f_max = f_max
        self.eta = eta

    def generate(self, power_request):
        f_t = np.clip(power_request, 0, self.f_max)
        h2_consumed = self.eta * f_t
        return h2_consumed, f_t

class HydrogenStorage:
    def __init__(self, h_max):
        self.h_max = h_max
        self.level = 0.0

    def update(self, inflow, outflow, dt=1.0):
        delta_h = (inflow - outflow) * dt
        new_level = self.level + delta_h
        violation_overflow = max(0, new_level - self.h_max)
        violation_underflow = max(0, -new_level)
        self.level = np.clip(new_level, 0, self.h_max)
        return self.level, violation_overflow, violation_underflow

    def check_demand_satisfaction(self, demand):
        if self.level < demand:
            return False, demand - self.level
        return True, 0.0

# ==========================================
# 3. Gym 环境 (Environment)
# ==========================================

class StartingCaseEnv(gym.Env):
    def __init__(self, config=None):
        super(StartingCaseEnv, self).__init__()

        self.cfg = {
            'n_days': 7,
            'dt': 0.25,
            'g_max': 670.0,
            'ele_eta': 0.02,
            'f_max': 200.0,
            'fc_eta': 0.06,
            'h_max': 450.0,
            'init_soc': 0.5,
            'terminal_penalty_coeff': 10000.0,
            'step_penalty_coeff': 20.0,
            'reward_scale': 0.001
        }
        if config:
            self.cfg.update(config)

        self.total_steps = 96 * self.cfg['n_days']

        self.electrolyzer = Electrolyzer(self.cfg['g_max'], self.cfg['ele_eta'])
        self.fuel_cell = FuelCell(self.cfg['f_max'], self.cfg['fc_eta'])
        self.storage = HydrogenStorage(self.cfg['h_max'])

        # Action: [G_t, F_t]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation
        low_obs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high_obs = np.array([self.cfg['h_max'], 200.0, 20.0, 1000.0, self.cfg['h_max']], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        self.time_step = 0
        self.initial_level = 0.0

        # 直接使用开头加载的全局变量
        self.raw_price_data = PRICE_DATA

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time_step = 0
        self.initial_level = self.cfg['init_soc'] * self.cfg['h_max']
        self.storage.level = self.initial_level
        self._generate_scenario_data()
        return self._get_obs(), {}

    def _generate_scenario_data(self):
        # 价格
        base_prices = np.tile(self.raw_price_data, self.cfg['n_days'])
        noise = self.np_random.normal(0, 0.5, len(base_prices))
        self.prices = np.clip(base_prices + noise, 0.1, 100.0)

        # 氢气需求
        self.demands_h2 = self.np_random.uniform(2.0, 8.0, self.total_steps)

        # 电力负荷
        t = np.linspace(0, 2*np.pi * self.cfg['n_days'], self.total_steps)
        base_load = 200.0
        load_profile = 100.0 * np.sin(t - np.pi/2)
        self.demands_ele = np.clip(base_load + load_profile + np.random.normal(0, 30, self.total_steps), 0, 500)

    def _get_obs(self):
        h_t = self.storage.level
        p_t = self.prices[self.time_step]
        d_h2_t = self.demands_h2[self.time_step]
        d_ele_t = self.demands_ele[self.time_step]
        return np.array([h_t, p_t, d_h2_t, d_ele_t, self.initial_level], dtype=np.float32)

    def step(self, action):
        act_ele = float(action[0])
        act_fc = float(action[1])

        g_t_target = (act_ele + 1) / 2 * self.cfg['g_max']
        f_t_target = (act_fc + 1) / 2 * self.cfg['f_max']

        current_p = self.prices[self.time_step]
        current_d_h2 = self.demands_h2[self.time_step]
        current_d_ele = self.demands_ele[self.time_step]

        # 物理流
        h2_prod, real_g_t = self.electrolyzer.produce(g_t_target)
        h2_cons, real_f_t = self.fuel_cell.generate(f_t_target)

        total_outflow_qty = (current_d_h2 * self.cfg['dt']) + h2_cons
        inflow_qty = h2_prod

        shortage = 0.0
        if self.storage.level < total_outflow_qty:
            shortage = total_outflow_qty - self.storage.level

        new_level, overflow, underflow = self.storage.update(
            inflow_qty / self.cfg['dt'],
            total_outflow_qty / self.cfg['dt'],
            self.cfg['dt']
        )

        # 经济计算 (包含FC发电抵消)
        net_grid_power = (real_g_t + current_d_ele) - real_f_t
        cost = current_p * net_grid_power * self.cfg['dt']

        # 惩罚
        step_penalty = (shortage * 20.0) + (overflow * 1.0) + (underflow * 20.0)

        daily_penalty = 0.0
        self.time_step += 1
        if self.time_step % 96 == 0:
            diff = self.initial_level - new_level
            if diff > 0:
                daily_penalty = diff * self.cfg['terminal_penalty_coeff']

        terminated = self.time_step >= self.total_steps

        raw_reward = -cost - (step_penalty * self.cfg['step_penalty_coeff']) - daily_penalty
        scaled_reward = raw_reward * self.cfg['reward_scale']

        fc_revenue = real_f_t * current_p * self.cfg['dt']

        info = {
            'cost': cost,
            'fc_revenue': fc_revenue,
            'h_level': new_level,
            'g_t': real_g_t,
            'f_t': real_f_t,
            'price': current_p,
            'ele_load': current_d_ele,
            'raw_reward': raw_reward
        }

        if terminated:
             obs = np.array([new_level, 0.0, 0.0, 0.0, self.initial_level], dtype=np.float32)
        else:
             obs = self._get_obs()

        return obs, scaled_reward, terminated, False, info

# ==========================================
# 4. TD3 训练与测试逻辑 (核心修改)
# ==========================================

def plot_results(h_levels, prices, prod_actions, fc_actions, ele_loads, init_h, algorithm_name="TD3"):
    t = np.arange(len(h_levels))
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    # 子图1: 储量与电价
    color = 'tab:blue'
    ax1.set_ylabel('Hydrogen Level (kg)', color=color, fontsize=12)
    ax1.plot(t, h_levels, color=color, linewidth=2, label='H2 Level')
    ax1.axhline(y=init_h, color='green', linestyle='--', linewidth=2, label='Target (I2S)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    for day in range(1, 8):
        ax1.axvline(x=day*96, color='gray', linestyle=':', alpha=0.5)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Electricity Price ($)', color=color, fontsize=12)
    ax2.plot(t, prices, color=color, linestyle='-', alpha=0.5, linewidth=1, label='Price')
    ax2.tick_params(axis='y', labelcolor=color)
    ax1.set_title(f'{algorithm_name} Control: Hydrogen Storage & Price', fontsize=14)

    # 子图2: 供需平衡
    ax3.set_ylabel('Power (kW)', fontsize=12)
    ax3.set_xlabel('Time Step (15 mins/Step)', fontsize=12)
    ax3.fill_between(t, 0, ele_loads, color='gray', alpha=0.2, label='EV Load (Demand)')
    ax3.bar(t, prod_actions, bottom=ele_loads, color='orange', alpha=0.6, width=1.0, label='Electrolyzer (Buy)')
    offset = [min(l, f) for l, f in zip(ele_loads, fc_actions)]
    ax3.bar(t, offset, color='purple', alpha=0.8, width=1.0, label='FC Generation (Offset)')
    ax3.legend(loc='upper right')
    ax3.set_title('Power Balance: EV Load vs. Hydrogen Actions', fontsize=12)

    plt.tight_layout()
    plt.show()

def train_td3():
    print("=== 开始训练 TD3 Agent (7 Days, 50k Steps) ===")

    env = make_vec_env(StartingCaseEnv, n_envs=1, vec_env_cls=DummyVecEnv, seed=42)

    # TD3 需要手动添加 Action Noise 来鼓励探索
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        # TD3 特有参数: 策略延迟更新频率
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
    )

    model.learn(total_timesteps=50000)
    return model

def test_td3(model):
    print("\n=== 测试 TD3 模型 ===")
    test_env = StartingCaseEnv()
    obs, _ = test_env.reset(seed=100)

    h_levels = []
    prices = []
    prod_actions = []
    fc_actions = []
    ele_loads = []

    done = False
    total_net_cost = 0
    total_fc_value = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated

        h_levels.append(info['h_level'])
        prices.append(info['price'])
        prod_actions.append(info['g_t'])
        fc_actions.append(info['f_t'])
        ele_loads.append(info['ele_load'])

        total_net_cost += info['cost']
        total_fc_value += info['fc_revenue']

    print(f"TD3 - 最终净电费成本: ${total_net_cost:.2f}")
    print(f"TD3 - 燃料电池创造价值: ${total_fc_value:.2f}")

    plot_results(h_levels, prices, prod_actions, fc_actions, ele_loads, test_env.initial_level, algorithm_name="TD3")

# ==========================================
# 主执行入口
# ==========================================
if __name__ == "__main__":
    td3_model = train_td3()
    test_td3(td3_model)
