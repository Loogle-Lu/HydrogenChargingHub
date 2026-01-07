import gym
import numpy as np
from gym import spaces
from config import Config
from components import Electrolyzer, CascadeCompressor, CascadeStorage, FuelCell, Chiller
from data_loader import DataLoader


class HydrogenEnv(gym.Env):
    def __init__(self):
        super(HydrogenEnv, self).__init__()
        self.ele = Electrolyzer()
        self.comp = CascadeCompressor()
        self.storage = CascadeStorage()
        self.fc = FuelCell()
        self.chiller = Chiller()
        self.data_loader = DataLoader()

        self.current_step = 0
        # Action: [Ele_Ratio, FC_Ratio]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        # Obs: [SOC, Price, RE_Power, Demand]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.state = None
        self.current_data = None

    def reset(self):
        self.current_step = 0
        self.data_loader.reset()
        self.storage = CascadeStorage()
        self.current_data = self.data_loader.get_step_data(self.current_step)

        re_power = self.current_data["wind"] + self.current_data["pv"]
        self.state = np.array([
            self.storage.level / self.storage.capacity,
            self.current_data["price"],
            re_power,
            self.current_data["demand"]
        ], dtype=np.float32)
        return self.state

    def step(self, action):
        ele_ratio = np.clip(action[0], 0, 1)
        fc_ratio = np.clip(action[1], 0, 1)

        price = self.current_data["price"]
        re_gen = self.current_data["wind"] + self.current_data["pv"]
        h2_demand = self.current_data["demand"]

        # 1. 物理过程
        # 制氢
        ele_power_target = ele_ratio * Config.ele_max_power
        h2_produced, ele_actual_power = self.ele.compute(ele_power_target)
        comp_power, comp_heat = self.comp.compute_power(h2_produced)
        chiller_power = self.chiller.compute_power(comp_heat)

        total_prod_load = ele_actual_power + comp_power + chiller_power

        # FC 耗氢需求
        fc_power_target = fc_ratio * Config.fc_max_power
        h2_needed_for_fc = fc_power_target / self.fc.efficiency

        total_outflow = h2_demand + h2_needed_for_fc

        # 储氢
        soc, excess_h2, shortage_h2 = self.storage.step(h2_produced, total_outflow)

        # 分配缺氢量
        supply_ratio = 1.0
        if total_outflow > 0 and shortage_h2 > 0:
            supply_ratio = max(0, 1 - shortage_h2 / total_outflow)

        real_h2_sold = h2_demand * supply_ratio
        real_fc_h2_used = h2_needed_for_fc * supply_ratio

        # 计算实际 FC 发电
        fc_actual_power, _ = self.fc.compute(real_fc_h2_used)

        # 2. 能量与财务
        # Net Power > 0 (Sell), < 0 (Buy)
        net_power = (fc_actual_power + re_gen) - total_prod_load

        revenue_h2 = real_h2_sold * Config.hydrogen_price

        cost_ele = 0
        revenue_ele = 0
        if net_power < 0:
            cost_ele = abs(net_power) * price
        else:
            revenue_ele = net_power * price * Config.electricity_price_sell_coef

        # 惩罚项
        penalty = 0
        # 缺氢惩罚 (防止 SOC 掉到底)
        if shortage_h2 > 0:
            penalty += shortage_h2 * Config.penalty_unmet_demand
        if excess_h2 > 0:
            penalty += excess_h2 * 10

        step_reward = revenue_h2 + revenue_ele - cost_ele - penalty
        step_profit = revenue_h2 + revenue_ele - cost_ele

        # 3. 状态更新
        self.current_step += 1
        done = self.current_step >= Config.episode_length

        # I2S 约束
        if done and Config.enable_i2s_constraint:
            final_soc = self.storage.level / self.storage.capacity
            init_soc = Config.storage_initial
            i2s_penalty = abs(final_soc - init_soc) * Config.i2s_penalty_weight
            step_reward -= i2s_penalty

        self.current_data = self.data_loader.get_step_data(self.current_step)
        next_re = self.current_data["wind"] + self.current_data["pv"]

        self.state = np.array([
            self.storage.level / self.storage.capacity,
            self.current_data["price"],
            next_re,
            self.current_data["demand"]
        ], dtype=np.float32)

        info = {
            "soc": soc,
            "profit": step_profit,
            "net_power": net_power,
            "re_gen": re_gen,
            "fc_power": fc_actual_power,
            "load_power": total_prod_load,
            "h2_sold": real_h2_sold
        }

        return self.state, step_reward / 100.0, done, info