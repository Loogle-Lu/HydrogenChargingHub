import gym
import numpy as np
from gym import spaces
from config import Config
from components import Electrolyzer, MultiStageCompressorSystem, MultiTankStorage, FuelCell, NonlinearChiller
from data_loader import DataLoader


class HydrogenEnv(gym.Env):
    def __init__(self):
        super(HydrogenEnv, self).__init__()
        # 初始化组件
        self.ele = Electrolyzer()
        self.comp_system = MultiStageCompressorSystem()
        self.chiller = NonlinearChiller()
        self.storage = MultiTankStorage()
        self.fc = FuelCell()
        self.data_loader = DataLoader()

        self.current_step = 0
        # 动作空间扩展: [ele_ratio, c1_ratio, c2_ratio, c3_ratio, fc_ratio]
        # 简化版: 保持原有[ele_ratio, fc_ratio], 压缩机自动协调
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        # 观察空间扩展: 添加T1, T2的SOC
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.state = None
        self.current_data = None

    def reset(self):
        self.current_step = 0
        self.data_loader.reset()
        self.storage = MultiTankStorage()  # 重置到初始 SOC
        self.chiller.reset()  # 重置冷却机状态
        self.current_data = self.data_loader.get_step_data(self.current_step)

        re_power = self.current_data["wind"] + self.current_data["pv"]
        total_soc = self.storage.get_total_soc()
        self.state = np.array([
            total_soc,  # 总体SOC
            self.storage.t1.get_soc(),  # T1 SOC
            self.storage.t2.get_soc(),  # T2 SOC
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

        # --- 1. 制氢 (Electrolyzer -> T1) ---
        ele_power_target = ele_ratio * Config.ele_max_power
        h2_produced, ele_actual_power = self.ele.compute(ele_power_target)
        
        # --- 2. 多级压缩链 ---
        # 简化策略: C1自动压缩T1中的部分氢气到T2
        # C2自动从T2压缩到T3系统
        # C3根据需求从T3压缩到T4
        
        # C1流量: 取决于T1的SOC和T2的容量
        t1_soc = self.storage.t1.get_soc()
        t2_soc = self.storage.t2.get_soc()
        # 简单控制逻辑: T1 SOC高时增加C1流量
        c1_flow = h2_produced * min(1.0, max(0.5, t1_soc))
        c1_power, c1_heat = self.comp_system.compute_c1(c1_flow)
        
        # C2流量: T2 SOC高时增加C2流量
        c2_flow = c1_flow * min(1.0, max(0.3, t2_soc))
        c2_power, c2_heat = self.comp_system.compute_c2(c2_flow)
        
        # C3流量: 根据高压需求（简化，这里设为需求的一部分）
        c3_flow = h2_demand * 0.1  # 10%需求来自超高压
        c3_power, c3_heat = self.comp_system.compute_c3(c3_flow)
        
        total_comp_heat = c1_heat + c2_heat + c3_heat
        
        # --- 3. 非线性Chiller ---
        chiller_power = self.chiller.compute_power(total_comp_heat)
        
        total_prod_load = ele_actual_power + c1_power + c2_power + c3_power + chiller_power

        # --- 4. 多储罐系统更新 ---
        # 主要供氢来自T3系统
        soc, excess_h2, shortage_h2 = self.storage.step_all(
            h2_from_ele=h2_produced,
            h2_to_c1=c1_flow,
            h2_from_c1=c1_flow,  # 假设无损
            h2_to_c2=c2_flow,
            h2_from_c2=c2_flow,  # 假设无损
            h2_demand=h2_demand,
            h2_to_c3=c3_flow,
            h2_from_c3=c3_flow
        )

        # --- 5. 燃料电池 (从T2或T3取氢) ---
        fc_power_target = fc_ratio * Config.fc_max_power
        h2_needed_for_fc = fc_power_target / self.fc.efficiency
        
        # 处理缺氢
        supply_ratio = 1.0
        if shortage_h2 > 0:
            available_h2 = h2_demand + h2_needed_for_fc - shortage_h2
            if (h2_demand + h2_needed_for_fc) > 0:
                supply_ratio = max(0, available_h2 / (h2_demand + h2_needed_for_fc))

        real_h2_sold = h2_demand * supply_ratio
        real_fc_h2_used = h2_needed_for_fc * supply_ratio
        fc_actual_power, _ = self.fc.compute(real_fc_h2_used)

        # --- 6. 财务与奖励 ---
        net_power = (fc_actual_power + re_gen) - total_prod_load

        revenue_h2 = real_h2_sold * Config.hydrogen_price

        cost_ele = 0
        revenue_ele = 0
        if net_power < 0:
            cost_ele = abs(net_power) * price
        else:
            revenue_ele = net_power * price * Config.electricity_price_sell_coef

        penalty = 0
        if shortage_h2 > 0: 
            penalty += shortage_h2 * Config.penalty_unmet_demand
        if excess_h2 > 0: 
            penalty += excess_h2 * 10

        step_reward = revenue_h2 + revenue_ele - cost_ele - penalty
        step_profit = revenue_h2 + revenue_ele - cost_ele

        self.current_step += 1
        done = self.current_step >= Config.episode_length

        # --- I2S 约束 (终端惩罚) ---
        if done and Config.enable_i2s_constraint:
            final_soc = self.storage.get_total_soc()
            init_soc = Config.storage_initial
            i2s_penalty = abs(final_soc - init_soc) * Config.i2s_penalty_weight
            step_reward -= i2s_penalty

        self.current_data = self.data_loader.get_step_data(self.current_step)
        next_re = self.current_data["wind"] + self.current_data["pv"]

        self.state = np.array([
            self.storage.get_total_soc(),
            self.storage.t1.get_soc(),
            self.storage.t2.get_soc(),
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
            "h2_sold": real_h2_sold,
            "comp_power": c1_power + c2_power + c3_power,
            "chiller_power": chiller_power,
            "tank_socs": self.storage.get_tank_socs()
        }

        return self.state, step_reward / 100.0, done, info