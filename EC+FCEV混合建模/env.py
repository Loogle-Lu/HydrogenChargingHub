import gym
import numpy as np
from gym import spaces
from config import Config
from components import (
    Electrolyzer, MultiStageCompressorSystem, MultiTankStorage, FuelCell, NonlinearChiller,
    MixedDemandGenerator, IntegratedServiceStation
)
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
        
        # 新增: EV/FCEV需求建模
        self.demand_generator = MixedDemandGenerator()
        self.service_station = IntegratedServiceStation()

        self.current_step = 0
        # 动作空间: [ele_ratio, fc_ratio]
        # Agent控制电解槽和燃料电池，压缩机和服务站自动运行
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 观察空间扩展: 
        # [总SOC, T1_SOC, T2_SOC, 电价, 可再生能源, EV队列长度, FCEV队列长度, 时段]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.state = None
        self.current_data = None

    def reset(self):
        self.current_step = 0
        self.data_loader.reset()
        self.storage = MultiTankStorage()  # 重置到初始 SOC
        self.chiller.reset()  # 重置冷却机状态
        self.service_station = IntegratedServiceStation()  # 重置服务站
        self.demand_generator = MixedDemandGenerator()  # 重置需求生成器
        
        self.current_data = self.data_loader.get_step_data(self.current_step)

        re_power = self.current_data["wind"] + self.current_data["pv"]
        total_soc = self.storage.get_total_soc()
        hour_of_day = int((self.current_step * Config.dt) % 24)
        
        self.state = np.array([
            total_soc,  # 总体SOC
            self.storage.t1.get_soc(),  # T1 SOC
            self.storage.t2.get_soc(),  # T2 SOC
            self.current_data["price"],  # 电价
            re_power,  # 可再生能源
            0.0,  # EV队列长度 (初始为0)
            0.0,  # FCEV队列长度 (初始为0)
            hour_of_day / 24.0  # 时段归一化 (0-1)
        ], dtype=np.float32)
        return self.state

    def step(self, action):
        ele_ratio = np.clip(action[0], 0, 1)
        fc_ratio = np.clip(action[1], 0, 1)

        price = self.current_data["price"]
        re_gen = self.current_data["wind"] + self.current_data["pv"]
        hour_of_day = int((self.current_step * Config.dt) % 24)

        # --- 1. 生成车辆到达 (EV/FCEV) ---
        ev_arrivals, fcev_arrivals = self.demand_generator.generate_vehicles(self.current_step)
        self.service_station.add_vehicles(ev_arrivals, fcev_arrivals)

        # --- 2. 制氢 (Electrolyzer -> T1) ---
        ele_power_target = ele_ratio * Config.ele_max_power
        h2_produced, ele_actual_power = self.ele.compute(ele_power_target)
        
        # --- 3. 计算可用氢气 (用于FCEV加氢) ---
        # 从T3系统获取可用氢气量
        t3_available_h2 = (self.storage.t3_1.level + self.storage.t3_2.level + 
                          self.storage.t3_3.level) * 0.8  # 80%可用于加氢
        
        # --- 4. 服务EV/FCEV需求 ---
        ev_power_demand, fcev_h2_demand, ev_revenue, fcev_revenue, service_penalty = \
            self.service_station.step(price, t3_available_h2, Config.dt)
        
        # --- 5. 多级压缩链 ---
        t1_soc = self.storage.t1.get_soc()
        t2_soc = self.storage.t2.get_soc()
        
        # C1流量: 基于T1 SOC自适应
        c1_flow = h2_produced * min(1.0, max(0.5, t1_soc))
        c1_power, c1_heat = self.comp_system.compute_c1(c1_flow)
        
        # C2流量: 基于T2 SOC和FCEV需求
        c2_flow = c1_flow * min(1.0, max(0.4, t2_soc)) + fcev_h2_demand * 0.5
        c2_power, c2_heat = self.comp_system.compute_c2(c2_flow)
        
        # C3流量: 主要为700bar FCEV快充
        c3_flow = fcev_h2_demand * 0.3  # 30%通过C3超高压
        c3_power, c3_heat = self.comp_system.compute_c3(c3_flow)
        
        total_comp_heat = c1_heat + c2_heat + c3_heat
        
        # --- 6. 非线性Chiller ---
        chiller_power = self.chiller.compute_power(total_comp_heat)
        
        # --- 7. 总电力负荷 ---
        # 制氢链负荷
        h2_production_load = ele_actual_power + c1_power + c2_power + c3_power + chiller_power
        # EV充电负荷
        total_load = h2_production_load + ev_power_demand

        # --- 8. 多储罐系统更新 ---
        soc, excess_h2, shortage_h2 = self.storage.step_all(
            h2_from_ele=h2_produced,
            h2_to_c1=c1_flow,
            h2_from_c1=c1_flow,
            h2_to_c2=c2_flow,
            h2_from_c2=c2_flow,
            h2_demand=fcev_h2_demand,  # FCEV加氢需求
            h2_to_c3=c3_flow,
            h2_from_c3=c3_flow
        )

        # --- 9. 燃料电池发电 (应对高负荷或高电价) ---
        fc_power_target = fc_ratio * Config.fc_max_power
        h2_needed_for_fc = fc_power_target / self.fc.efficiency
        
        # 处理缺氢
        supply_ratio = 1.0
        if shortage_h2 > 0:
            available_h2 = fcev_h2_demand + h2_needed_for_fc - shortage_h2
            if (fcev_h2_demand + h2_needed_for_fc) > 0:
                supply_ratio = max(0, available_h2 / (fcev_h2_demand + h2_needed_for_fc))

        real_fcev_h2_supplied = fcev_h2_demand * supply_ratio
        real_fc_h2_used = h2_needed_for_fc * supply_ratio
        fc_actual_power, _ = self.fc.compute(real_fc_h2_used)

        # --- 10. 电网互动与财务 ---
        # 电力平衡: 供应 - 需求
        net_power = (fc_actual_power + re_gen) - total_load

        # 收入来源
        revenue_ev = ev_revenue  # EV充电收入
        revenue_fcev = fcev_revenue  # FCEV加氢收入
        
        # 电力成本/收入
        cost_grid = 0
        revenue_grid = 0
        if net_power < 0:
            # 从电网购电
            cost_grid = abs(net_power) * price
        else:
            # 向电网售电
            revenue_grid = net_power * price * Config.electricity_price_sell_coef

        # 惩罚
        penalty = service_penalty  # 未满足服务惩罚
        if shortage_h2 > 0: 
            penalty += shortage_h2 * Config.penalty_unmet_h2_demand
        if excess_h2 > 0: 
            penalty += excess_h2 * 10

        # 总收益
        step_reward = revenue_ev + revenue_fcev + revenue_grid - cost_grid - penalty
        step_profit = revenue_ev + revenue_fcev + revenue_grid - cost_grid

        self.current_step += 1
        done = self.current_step >= Config.episode_length

        # --- I2S 约束 (终端惩罚) ---
        if done and Config.enable_i2s_constraint:
            final_soc = self.storage.get_total_soc()
            init_soc = Config.storage_initial
            i2s_penalty = abs(final_soc - init_soc) * Config.i2s_penalty_weight
            step_reward -= i2s_penalty

        # --- 更新状态 ---
        self.current_data = self.data_loader.get_step_data(self.current_step)
        next_re = self.current_data["wind"] + self.current_data["pv"]
        next_hour = int((self.current_step * Config.dt) % 24)
        
        station_stats = self.service_station.get_statistics()

        self.state = np.array([
            self.storage.get_total_soc(),
            self.storage.t1.get_soc(),
            self.storage.t2.get_soc(),
            self.current_data["price"],
            next_re,
            min(station_stats['ev_queue_length'] / 10.0, 1.0),  # 归一化EV队列
            min(station_stats['fcev_queue_length'] / 5.0, 1.0),  # 归一化FCEV队列
            next_hour / 24.0  # 时段
        ], dtype=np.float32)

        info = {
            "soc": soc,
            "profit": step_profit,
            "net_power": net_power,
            "re_gen": re_gen,
            "fc_power": fc_actual_power,
            "load_power": total_load,  # 兼容main.py
            "h2_production_load": h2_production_load,
            "ev_charging_load": ev_power_demand,
            "total_load": total_load,
            "comp_power": c1_power + c2_power + c3_power,
            "chiller_power": chiller_power,
            "tank_socs": self.storage.get_tank_socs(),
            # EV/FCEV统计
            "ev_served": station_stats['ev_served'],
            "fcev_served": station_stats['fcev_served'],
            "ev_revenue": ev_revenue,
            "fcev_revenue": fcev_revenue,
            "ev_queue": station_stats['ev_queue_length'],
            "fcev_queue": station_stats['fcev_queue_length'],
            "vehicles_delayed": station_stats['vehicles_delayed'],
            # 兼容旧版
            "h2_sold": fcev_h2_demand * supply_ratio  # 近似值
        }

        return self.state, step_reward / 100.0, done, info