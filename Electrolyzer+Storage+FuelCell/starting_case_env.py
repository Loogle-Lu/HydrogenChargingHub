import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pickle
import os


# ==========================================
# 1. 组件模型
# ==========================================

class Electrolyzer:
    def __init__(self, g_max, eta):
        self.g_max = g_max  # kW
        self.eta = eta  # kg/kWh (产氢效率)

    def produce(self, power_input):
        # 动作截断
        g_t = np.clip(power_input, 0, self.g_max)
        h2_produced = self.eta * g_t
        return h2_produced, g_t


class FuelCell:
    def __init__(self, f_max, eta):
        self.f_max = f_max  # kW
        self.eta = eta  # kg/kWh (耗氢系数: 发1kWh电需要多少kg氢)
        # 注意: 有时候 eta 定义为效率 %, 这里为了方便计算直接用 kg/kWh
        # 假设燃料电池效率 50%, 氢热值 33.3 kWh/kg -> 发 1kWh 电需要 1/(33.3*0.5) = 0.06 kg
        # 论文代码 hydro_sys.py 中 HFC 类似乎有具体参数，这里先给一个合理估计值

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
# 2. Gym 环境 (集成燃料电池)
# ==========================================

class StartingCaseEnv(gym.Env):
    def __init__(self, config=None):
        super(StartingCaseEnv, self).__init__()

        self.cfg = {
            'n_days': 7,
            'dt': 0.25,  # 15 mins

            # === Electrolyzer ===
            'g_max': 670.0,
            'ele_eta': 0.02,  # kg/kWh

            # === Fuel Cell (新增) ===
            'f_max': 200.0,  # kW (假设值，可根据论文 Case 2 调整为 100 或其他)
            'fc_eta': 0.06,  # kg/kWh (发 1kW 电消耗 0.06kg 氢)

            # === Storage ===
            'h_max': 450.0,
            'init_soc': 0.5,

            # === Penalties ===
            'terminal_penalty_coeff': 10000.0,
            'step_penalty_coeff': 20.0,
            'reward_scale': 0.001
        }
        if config:
            self.cfg.update(config)

        self.total_steps = 96 * self.cfg['n_days']

        # 组件
        self.electrolyzer = Electrolyzer(self.cfg['g_max'], self.cfg['ele_eta'])
        self.fuel_cell = FuelCell(self.cfg['f_max'], self.cfg['fc_eta'])
        self.storage = HydrogenStorage(self.cfg['h_max'])

        # 动作空间: [g_t, f_t] -> 2维
        # 范围 [-1, 1]，分别映射到 [0, g_max] 和 [0, f_max]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 观察空间: [h, p, d, h_target] (暂时保持不变，FC不增加额外状态，只增加决策维度)
        low_obs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high_obs = np.array([self.cfg['h_max'], 200.0, 20.0, self.cfg['h_max']], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        self.time_step = 0
        self.initial_level = 0.0
        self.raw_price_data = self._load_price_data()

    def _load_price_data(self):
        filename = 'price_after_MAD_96.pkl'
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, (list, np.ndarray)):
                    return np.array(data)
            except:
                pass
        t = np.linspace(0, 2 * np.pi, 96)
        return np.clip(20 + 10 * np.sin(t), 1, 50)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time_step = 0
        self.initial_level = self.cfg['init_soc'] * self.cfg['h_max']
        self.storage.level = self.initial_level
        self._generate_scenario_data()
        return self._get_obs(), {}

    def _generate_scenario_data(self):
        base_prices = np.tile(self.raw_price_data, self.cfg['n_days'])
        noise = self.np_random.normal(0, 1.0, len(base_prices))
        self.prices = np.clip(base_prices + noise, 0.1, 100.0)
        self.demands = self.np_random.uniform(2.0, 8.0, self.total_steps)

    def _get_obs(self):
        h_t = self.storage.level
        p_t = self.prices[self.time_step]
        d_t = self.demands[self.time_step]
        return np.array([h_t, p_t, d_t, self.initial_level], dtype=np.float32)

    def step(self, action):
        # 解析动作
        act_ele = float(action[0])  # Electrolyzer [-1, 1]
        act_fc = float(action[1])  # Fuel Cell [-1, 1]

        g_t_target = (act_ele + 1) / 2 * self.cfg['g_max']
        f_t_target = (act_fc + 1) / 2 * self.cfg['f_max']

        # 互斥逻辑 (可选): 也可以允许同时跑，但物理上不划算
        # 这里暂不强制互斥，让 RL 自己学

        current_p = self.prices[self.time_step]
        current_d = self.demands[self.time_step]

        # 1. 计算产/耗
        h2_prod, real_g_t = self.electrolyzer.produce(g_t_target)
        h2_cons, real_f_t = self.fuel_cell.generate(f_t_target)

        # 2. 检查储量是否足够支持发电和需求
        # 总消耗需求 = 用户需求 + 燃料电池消耗
        total_h2_demand = current_d + (h2_cons / self.cfg['dt'])  # 注意单位: h2_cons是量，d是速率
        # update函数里是以量为单位，这里简化处理：
        # update(inflow_rate, outflow_rate)

        inflow_rate = h2_prod / self.cfg['dt']  # kg/h
        outflow_rate = current_d + (h2_cons / self.cfg['dt'])  # kg/h

        # 3. 更新储罐
        # 注意: update 内部 delta = (in - out) * dt
        # 所以这里的 inflow/outflow 应该是速率
        # h2_prod 是本次step的总量? 不，produce返回的是量 = eta * power * dt?
        # 等等，之前的代码 Electrolyzer.produce 返回的是 h2_produced (量)。
        # 而 update 接收的是 inflow, outflow。之前的 update 实现是 delta = (in-out)*dt。
        # 这意味着之前的 update 假设输入是速率。
        # 让我们修正一下逻辑，统一用量 (kg)。

        # Electrolyzer: eta (kg/kWh) * Power (kW) * dt (h) = kg
        h2_prod_qty = self.cfg['ele_eta'] * real_g_t * self.cfg['dt']

        # FuelCell: eta (kg/kWh) * Power (kW) * dt (h) = kg
        h2_cons_qty = self.cfg['fc_eta'] * real_f_t * self.cfg['dt']

        # External Demand: d (kg/h) * dt (h) = kg
        demand_qty = current_d * self.cfg['dt']

        total_outflow_qty = h2_cons_qty + demand_qty

        # 检查当前储量是否足够支付 outflow
        shortage = 0.0
        if self.storage.level < total_outflow_qty:
            shortage = total_outflow_qty - self.storage.level
            # 物理限制: 不能透支。实际消耗量被截断
            # 但这里简化处理，允许透支并给予惩罚(Underflow penalty)

        # 更新储量 (直接传量，update内部逻辑稍微修改一下或者传 量/dt)
        # 之前的 update: delta_h = (inflow - outflow) * dt
        # 所以传入: inflow = h2_prod_qty / dt, outflow = total_outflow_qty / dt
        new_level, overflow, underflow = self.storage.update(
            h2_prod_qty / self.cfg['dt'],
            total_outflow_qty / self.cfg['dt'],
            self.cfg['dt']
        )

        # 4. 经济计算
        # 网购电费 = (制氢耗电 - 电池发电) * 电价
        # 如果 电池发电 > 制氢耗电，则 cost 为负 (赚钱)
        net_power = real_g_t - real_f_t
        cost = current_p * net_power * self.cfg['dt']

        # 惩罚
        # 需求缺口惩罚 (包括无法满足用户需求 + 无法满足发电需求)
        step_penalty = (shortage * 20.0) + (overflow * 1.0) + (underflow * 20.0)

        # 每日 I2S
        daily_penalty = 0.0
        self.time_step += 1
        if self.time_step % 96 == 0:
            diff = self.initial_level - new_level
            if diff > 0:
                daily_penalty = diff * self.cfg['terminal_penalty_coeff']

        terminated = self.time_step >= self.total_steps

        raw_reward = -cost - (step_penalty * self.cfg['step_penalty_coeff']) - daily_penalty
        scaled_reward = raw_reward * self.cfg['reward_scale']

        info = {
            'cost': cost,
            'h_level': new_level,
            'g_t': real_g_t,
            'f_t': real_f_t,  # 记录发电功率
            'price': current_p,
            'raw_reward': raw_reward
        }

        if terminated:
            obs = np.array([new_level, 0.0, 0.0, self.initial_level], dtype=np.float32)
        else:
            obs = self._get_obs()

        return obs, scaled_reward, terminated, False, info
