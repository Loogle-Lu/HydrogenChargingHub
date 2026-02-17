# ==========================================
# 0. 安装依赖 (Colab 专用)
# ==========================================
try:
    import gymnasium
    import stable_baselines3
except ImportError:
    print("正在安装依赖库，请稍候...")
    !pip install gymnasium stable-baselines3 shimmy matplotlib pandas > /dev/null 2>&1
    print("依赖安装完成！")

# ==========================================
# 1. 导入库
# ==========================================
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd

# ==========================================
# 2. 物理组件模型 (Components)
# ==========================================
class Electrolyzer:
    def __init__(self, max_power, efficiency=0.02):
        self.max_power = max_power # kW
        self.efficiency = efficiency # kg/kWh (约 50kWh/kg)

    def run(self, action_norm):
        # 动作 [-1, 1] -> [0, max_power]
        power = (action_norm + 1) / 2 * self.max_power
        h2_produced = power * self.efficiency # kg
        return power, h2_produced

class FuelCell:
    def __init__(self, max_power, efficiency=0.06): # 假设效率较高用于演示
        self.max_power = max_power # kW
        self.efficiency = efficiency # kg/kWh (消耗)

    def run(self, action_norm):
        # 动作 [-1, 1] -> [0, max_power]
        power = (action_norm + 1) / 2 * self.max_power
        h2_consumed = power * self.efficiency # kg
        return power, h2_consumed

class Compressor:
    def __init__(self, max_flow_rate, energy_cost=1.5):
        self.max_flow = max_flow_rate # kg/step
        self.energy_cost = energy_cost # kWh/kg

    def run(self, action_norm, lp_level):
        # 动作 [-1, 1] -> [0, max_flow]
        target_flow = (action_norm + 1) / 2 * self.max_flow
        # 物理限制：不能抽取超过低压罐存量的气体
        real_flow = min(target_flow, lp_level)
        power_consumed = real_flow * self.energy_cost
        return power_consumed, real_flow

class HydrogenTank:
    def __init__(self, capacity, init_soc=0.5):
        self.capacity = capacity # kg
        self.level = capacity * init_soc

    def update(self, inflow, outflow):
        self.level += (inflow - outflow)
        # 溢出处理 (浪费)
        overflow = max(0, self.level - self.capacity)
        self.level = min(self.level, self.capacity)
        # 欠压处理 (未满足需求)
        underflow = max(0, 0 - self.level)
        self.level = max(0, self.level)
        return self.level, overflow, underflow

# ==========================================
# 3. 核心环境 (Hybrid Hub Environment)
# ==========================================
class HybridChargingHubEnv(gym.Env):
    def __init__(self):
        super(HybridChargingHubEnv, self).__init__()

        # --- 配置参数 ---
        self.steps_per_day = 96  # 15分钟一个步长
        self.days_per_episode = 3
        self.max_steps = self.steps_per_day * self.days_per_episode
        self.dt = 0.25 # hours

        # 经济参数 (核心差异点)
        self.params = {
            'grid_price_base': 0.15,    # $/kWh 基础电价
            'ev_service_fee': 0.10,     # $/kWh 充电服务费 (利润来源1)
            'h2_selling_price': 10.0,   # $/kg 氢气售价 (利润来源2)
            'penalty_I2S': 50.0,        # I2S 约束惩罚系数
            'penalty_missed': 20.0,     # 需求未满足惩罚
        }

        # 组件规格
        self.ely = Electrolyzer(max_power=300.0)    # 制氢机 300kW
        self.fc = FuelCell(max_power=100.0)         # 燃料电池 100kW
        self.comp = Compressor(max_flow_rate=5.0)   # 压缩机

        self.lp_tank = HydrogenTank(capacity=20.0, init_soc=0.0) # 低压缓冲罐
        self.hp_tank = HydrogenTank(capacity=200.0, init_soc=0.5) # 高压储氢罐 (I2S约束对象)
        self.init_hp_level = self.hp_tank.level

        # 动作空间: [制氢功率, 压缩速率, 发电功率] -> 都是 [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # 观察空间: [HP_SOC, LP_SOC, 电价, H2需求, EV需求, 光伏, 风电]
        self.observation_space = spaces.Box(low=0, high=1000, shape=(7,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.lp_tank.level = 0
        self.hp_tank.level = self.init_hp_level
        self.cumulative_profit = 0

        # 生成场景数据 (模拟真实世界)
        self._generate_scenario()

        return self._get_obs(), {}

    def _generate_scenario(self):
        # 简单的合成数据生成器
        t = np.linspace(0, self.days_per_episode * 24, self.max_steps)

        # 1. 电价 (鸭子曲线：晚高峰贵)
        base_price = 0.15
        peak_adder = 0.20 * np.exp(-((t % 24 - 19)**2) / 10) # 晚上7点高峰
        self.data_price = np.clip(base_price + peak_adder + np.random.normal(0, 0.01, len(t)), 0.05, 0.5)

        # 2. 光伏 (正午高峰)
        self.data_pv = 300.0 * np.maximum(0, np.sin((t % 24 - 6) * np.pi / 12)) # 6点-18点有光
        self.data_pv *= np.random.uniform(0.8, 1.2, len(t)) # 随机云遮挡

        # 3. 风电 (随机性强)
        self.data_wind = 100.0 * np.abs(np.sin(t/5) + np.random.normal(0, 0.3, len(t)))

        # 4. EV 需求 (早晚高峰)
        self.data_ev = 200.0 * (np.exp(-((t % 24 - 9)**2)/5) + np.exp(-((t % 24 - 18)**2)/5))
        self.data_ev = np.clip(self.data_ev, 20, 400)

        # 5. H2 需求 (随机到达)
        self.data_h2 = np.random.choice([0, 5.0], p=[0.9, 0.1], size=len(t)) # 10%概率有车来加氢

    def _get_obs(self):
        # === 修复：增加索引边界检查 ===
        # 当 current_step 增加到 max_steps 时 (episode 结束)，防止索引越界
        idx = min(self.current_step, self.max_steps - 1)

        return np.array([
            self.hp_tank.level,
            self.lp_tank.level,
            self.data_price[idx],
            self.data_h2[idx],
            self.data_ev[idx],
            self.data_pv[idx],
            self.data_wind[idx]
        ], dtype=np.float32)

    def step(self, action):
        idx = self.current_step
        grid_price = self.data_price[idx]
        h2_demand = self.data_h2[idx]
        ev_load = self.data_ev[idx]
        pv_gen = self.data_pv[idx]
        wind_gen = self.data_wind[idx]

        # --- 1. 物理流计算 ---
        # A. 燃料电池 (HP -> Power)
        # 只有当HP有气时才能发电
        fc_power_target, h2_fc_cons_target = self.fc.run(action[2])
        actual_h2_for_fc = min(h2_fc_cons_target, self.hp_tank.level)
        # 反算实际发电量 (线性近似)
        fc_power_real = fc_power_target * (actual_h2_for_fc / (h2_fc_cons_target + 1e-6))

        # B. 制氢机 (Power -> LP)
        ely_power, h2_prod = self.ely.run(action[0])

        # C. 压缩机 (LP -> HP)
        # 从LP抽气，如果LP不够就只抽一部分
        comp_power, h2_moved = self.comp.run(action[1], self.lp_tank.level + h2_prod) # 加上刚产出的

        # --- 2. 储罐更新 ---
        # LP罐: 进=制氢, 出=压缩
        self.lp_tank.update(inflow=h2_prod, outflow=h2_moved)

        # HP罐: 进=压缩, 出=FC消耗+外部加氢需求
        # 优先满足外部加氢需求 (赚钱!)
        h2_sales_succ = min(h2_demand, self.hp_tank.level - actual_h2_for_fc) # 简单逻辑
        if h2_sales_succ < 0: h2_sales_succ = 0

        total_hp_outflow = actual_h2_for_fc + h2_sales_succ
        self.hp_tank.update(inflow=h2_moved, outflow=total_hp_outflow)

        # --- 3. 能量平衡与经济结算 (核心部分) ---
        # 负载端
        total_load = ev_load + ely_power + comp_power
        # 供给端
        total_renewables = pv_gen + wind_gen + fc_power_real

        # 净负荷 (如果<0，说明有多余电，假设被浪费或以低价上网，这里简化为0)
        net_grid_load = max(0, total_load - total_renewables)

        # === 财务计算 ===
        # 1. 成本: 向电网买电
        cost_grid = net_grid_load * grid_price * self.dt

        # 2. 收入:
        #    EV充电费 = 负荷 * (基础电价 + 服务费)
        revenue_ev = ev_load * (grid_price + self.params['ev_service_fee']) * self.dt
        #    H2销售 = 卖出的量 * 单价
        revenue_h2 = h2_sales_succ * self.params['h2_selling_price']

        # 3. 利润
        step_profit = (revenue_ev + revenue_h2) - cost_grid
        self.cumulative_profit += step_profit

        # --- 4. 奖励函数设计 ---
        reward = step_profit * 0.1 # 缩放系数

        # 惩罚1: 未满足的加氢需求
        missed_h2 = h2_demand - h2_sales_succ
        reward -= missed_h2 * self.params['penalty_missed']

        # 惩罚2: I2S 周期约束 (每天结束时检查)
        self.current_step += 1
        done = self.current_step >= self.max_steps

        if self.current_step % self.steps_per_day == 0:
            # 每天结束，希望 SOC 回归初始值
            soc_diff = abs(self.hp_tank.level - self.init_hp_level)
            penalty_i2s = soc_diff * self.params['penalty_I2S']
            reward -= penalty_i2s

        # 信息记录
        info = {
            'profit': step_profit,
            'hp_level': self.hp_tank.level,
            'grid_power': net_grid_load,
            'renewable_used': min(total_load, total_renewables),
            'ev_load': ev_load,
            'price': grid_price
        }

        return self._get_obs(), reward, done, False, info

# ==========================================
# 4. 训练与可视化工具
# ==========================================

def train_and_visualize():
    print(">>> 正在初始化混合能源枢纽环境...")
    env = DummyVecEnv([lambda: HybridChargingHubEnv()])

    print(">>> 开始 PPO 训练 (这可能需要 1-2 分钟)...")
    # 为了演示，步数设得较少，实际效果好需要更多步数
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=1e-3)
    model.learn(total_timesteps=30000)
    print(">>> 训练完成!")

    print(">>> 正在进行测试与数据收集...")
    # 测试
    test_env = HybridChargingHubEnv()
    obs, _ = test_env.reset()

    history = {k: [] for k in ['profit', 'cum_profit', 'hp_level', 'grid_power', 'renewable_used', 'price', 'ev_load']}

    done = False
    cum_profit = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, info = test_env.step(action)

        cum_profit += info['profit']
        history['profit'].append(info['profit'])
        history['cum_profit'].append(cum_profit)
        history['hp_level'].append(info['hp_level'])
        history['grid_power'].append(info['grid_power'])
        history['renewable_used'].append(info['renewable_used'])
        history['price'].append(info['price'])
        history['ev_load'].append(info['ev_load'])

    # === 可视化绘图 ===
    plot_results(history)

def plot_results(data):
    steps = range(len(data['profit']))
    fig, ax = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    # 1. 累计利润 (最重要的指标)
    ax[0].plot(steps, data['cum_profit'], color='green', linewidth=2, label='Cumulative Profit ($)')
    ax[0].set_title('Financial Performance (Cumulative Profit)', fontsize=14, fontweight='bold')
    ax[0].set_ylabel('Profit ($)')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc='upper left')

    # 2. 储氢罐状态 (I2S 约束)
    ax[1].plot(steps, data['hp_level'], color='blue', label='H2 Tank Level (kg)')
    ax[1].axhline(y=100, color='red', linestyle='--', label='Initial Target')
    ax[1].set_title('Hydrogen Storage System', fontsize=12)
    ax[1].set_ylabel('Level (kg)')
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    # 3. 功率平衡 (新能源 vs 电网)
    ax[2].stackplot(steps, data['renewable_used'], data['grid_power'],
                     labels=['Renewable Consumed', 'Grid Power Purchased'],
                     colors=['#90EE90', '#FF6347'], alpha=0.8)
    ax[2].plot(steps, data['ev_load'], color='black', linestyle='--', linewidth=1, label='EV Demand')
    ax[2].set_title('Energy Balance Source', fontsize=12)
    ax[2].set_ylabel('Power (kW)')
    ax[2].legend(loc='upper right')

    # 4. 电价趋势
    ax[3].plot(steps, data['price'], color='purple', linestyle='-')
    ax[3].set_title('Grid Electricity Price', fontsize=12)
    ax[3].set_ylabel('Price ($/kWh)')
    ax[3].set_xlabel('Time Steps (15 min intervals)')
    ax[3].grid(True)

    plt.tight_layout()
    plt.show()

# ==========================================
# 5. 执行主程序
# ==========================================
if __name__ == "__main__":
    train_and_visualize()
