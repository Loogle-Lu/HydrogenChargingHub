import gym
import numpy as np
from gym import spaces
from config import Config
from components import (
    Electrolyzer, MultiStageCompressorSystem, MultiTankStorage, FuelCell, LinearChiller,
    BatteryEnergyStorage, MixedDemandGenerator, IntegratedServiceStation
)
from data_loader import DataLoader


class HydrogenEnv(gym.Env):
    def __init__(self, enable_i2s_constraint=None):
        super(HydrogenEnv, self).__init__()
        # 初始化组件
        self.ele = Electrolyzer()
        self.comp_system = MultiStageCompressorSystem()
        self.chiller = LinearChiller()
        self.storage = MultiTankStorage()
        self.fc = FuelCell()
        self.battery = BatteryEnergyStorage()  # v2.6新增: 电池储能系统
        self.data_loader = DataLoader()
        
        # 新增: EV/FCEV需求建模
        self.demand_generator = MixedDemandGenerator()
        self.service_station = IntegratedServiceStation()

        self.current_step = 0
        # 动作空间: [ele_ratio, fc_ratio]
        # Agent控制电解槽和燃料电池，压缩机和服务站自动运行
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 观察空间扩展: 
        # [总SOC, T1_SOC, T2_SOC, 电池SOC, 电价, 可再生能源, EV队列长度, FCEV队列长度, 时段]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.state = None
        self.current_data = None

        if enable_i2s_constraint is None:
            self.enable_i2s_constraint = Config.enable_i2s_constraint
        else:
            self.enable_i2s_constraint = bool(enable_i2s_constraint)

    def reset(self):
        self.current_step = 0
        self.data_loader.reset()
        self.storage = MultiTankStorage()  # 重置到初始 SOC
        self.chiller.reset()  # 重置冷却机状态
        self.battery.reset()  # v2.6: 重置电池状态
        self.service_station = IntegratedServiceStation()  # 重置服务站
        self.demand_generator = MixedDemandGenerator()  # 重置需求生成器
        self.ele.reset()  # 重置电解槽统计
        self.comp_system.reset()  # v3.1: 重置压缩机统计
        
        self.current_data = self.data_loader.get_step_data(self.current_step)

        re_power = self.current_data["wind"] + self.current_data["pv"]
        total_soc = self.storage.get_total_soc()
        hour_of_day = int((self.current_step * Config.dt) % 24)
        
        self.state = np.array([
            total_soc,  # 总体氢气SOC
            self.storage.t1.get_soc(),  # T1 SOC
            self.storage.t2.get_soc(),  # T2 SOC
            self.battery.get_soc(),  # 电池SOC (v2.6新增)
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
        current_soc = self.storage.get_total_soc()
        ele_power_target = ele_ratio * Config.ele_max_power
        h2_produced, ele_actual_power, green_h2_ratio, power_from_re, power_from_grid = \
            self.ele.compute(ele_power_target, re_gen)
        
        # --- 4. 计算可用氢气 (用于FCEV加氢) ---
        # 从T3系统获取可用氢气量
        t3_available_h2 = (self.storage.t3_1.level + self.storage.t3_2.level + 
                          self.storage.t3_3.level) * 0.8  # 80%可用于加氢
        
        # --- 5. 服务EV/FCEV需求 ---
        ev_power_demand, fcev_h2_demand, ev_revenue, fcev_revenue, service_penalty = \
            self.service_station.step(price, t3_available_h2, Config.dt)
        
        # --- 6. 多级压缩链 (v3.1: 智能旁路+自适应压力+动态冷却) ---
        t1_soc = self.storage.t1.get_soc()
        t2_soc = self.storage.t2.get_soc()
        t3_avg_pressure = (self.storage.t3_1.max_pressure * self.storage.t3_1.get_soc() +
                           self.storage.t3_2.max_pressure * self.storage.t3_2.get_soc() +
                           self.storage.t3_3.max_pressure * self.storage.t3_3.get_soc()) / 3.0
        t4_pressure = self.storage.t4.max_pressure * self.storage.t4.get_soc()
        
        # 获取当前FCEV平均SOG (用于自适应压力控制)
        avg_fcev_sog = self.service_station.current_fcev_avg_sog
        
        # C1流量: 基于T1 SOC自适应
        c1_flow = h2_produced * min(1.0, max(0.5, t1_soc))
        t2_pressure = Config.c1_output_pressure * t2_soc
        c1_power, c1_heat = self.comp_system.compute_c1(
            c1_flow, 
            tank_pressure=t2_pressure,
            electricity_price=price
        )
        
        # C2流量: 基于T2 SOC和FCEV需求
        c2_flow = c1_flow * min(1.0, max(0.4, t2_soc)) + fcev_h2_demand * 0.5
        c2_power, c2_heat = self.comp_system.compute_c2(
            c2_flow,
            tank_pressure=t3_avg_pressure,
            electricity_price=price
        )
        
        # C3流量: 主要为FCEV快充（自适应压力）
        c3_flow = fcev_h2_demand * 0.3  # 30%通过C3超高压
        c3_power, c3_heat = self.comp_system.compute_c3(
            c3_flow,
            avg_fcev_sog=avg_fcev_sog,
            tank_pressure=t4_pressure,
            electricity_price=price
        )
        
        total_comp_heat = c1_heat + c2_heat + c3_heat
        
        # --- 7. 线性Chiller ---
        chiller_power = self.chiller.compute_power(total_comp_heat)
        
        # --- 8. 总电力负荷 ---
        # 制氢链负荷
        h2_production_load = ele_actual_power + c1_power + c2_power + c3_power + chiller_power
        # EV充电负荷
        total_load = h2_production_load + ev_power_demand

        # --- 9. 多储罐系统更新 ---
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

        # --- 10. 燃料电池发电 (应对高负荷或高电价) ---
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

        # --- 11. 电池储能系统 (v2.6新增: 快速调峰与功率平滑) ---
        battery_charge_power = 0.0
        battery_discharge_power = 0.0
        battery_net_power = 0.0  # 电池净输出功率 (放电为正，充电为负)
        
        if Config.enable_battery_storage:
            # 初步计算电力缺口 (不含电池)
            available_re_for_grid = re_gen - power_from_re
            preliminary_net = (fc_actual_power + available_re_for_grid) - (total_load - ele_actual_power)
            
            # 电池充放电策略:
            # 1. 当有功率剩余 (preliminary_net > 0): 充电存储
            # 2. 当有功率缺口 (preliminary_net < 0): 放电补充
            # 3. 优先级: 电池 > 电网 (电池响应快，效率高)
            
            if preliminary_net > 50:  # 有剩余功率，充电
                battery_charge_power, _ = self.battery.charge(preliminary_net)
                battery_net_power = -battery_charge_power  # 充电为负
            elif preliminary_net < -50:  # 功率缺口，放电
                battery_discharge_power, _ = self.battery.discharge(-preliminary_net)
                battery_net_power = battery_discharge_power  # 放电为正
        
        # --- 12. 电网互动与财务 (含储能套利奖励 + 电池) ---
        # 电力平衡: 供应 - 需求
        # 注意: 电解槽消耗的RE已经在power_from_re中体现，不重复计算
        available_re_for_grid = re_gen - power_from_re  # 剩余可再生能源
        net_power = (fc_actual_power + available_re_for_grid + battery_net_power) - (total_load - ele_actual_power)

        # 收入来源
        revenue_ev = ev_revenue  # EV充电收入
        revenue_fcev = fcev_revenue  # FCEV加氢收入
        
        # 储能套利奖励 (新增: 鼓励低价制氢、高价放电)
        arbitrage_bonus = 0.0
        if Config.enable_arbitrage_bonus:
            # 低价时制氢奖励
            if price < Config.price_threshold_low and ele_actual_power > 0:
                price_advantage = (Config.price_threshold_low - price) / Config.price_threshold_low
                arbitrage_bonus += ele_actual_power * Config.dt * price_advantage * Config.arbitrage_bonus_coef
            
            # 高价时放电奖励
            if price > Config.price_threshold_high and fc_actual_power > 0:
                price_advantage = (price - Config.price_threshold_high) / Config.price_threshold_high
                arbitrage_bonus += fc_actual_power * Config.dt * price_advantage * Config.arbitrage_bonus_coef
            
            # SOC健康度奖励（鼓励维持在合理范围）
            soc_health_bonus = 0.0
            if 0.4 <= current_soc <= 0.6:
                soc_health_bonus = 50.0  # SOC在健康范围内给予奖励
            arbitrage_bonus += soc_health_bonus
        
        # 电力成本/收入计算 (v3.1修正: 更真实的成本核算)
        # 
        # 核心理念：
        # 1. 制氢生产链用电（电解槽+压缩机+冷却器）应计入成本，无论电力来源
        #    - 如果用RE：机会成本（本可以卖给电网）
        #    - 如果用Grid：直接成本
        # 2. EV充电直接向用户收费，不计入Grid Cost
        # 3. 燃料电池发电是收入
        
        # 制氢生产链总负载
        h2_production_load_total = ele_actual_power + c1_power + c2_power + c3_power + chiller_power
        
        # 制氢生产链的电力成本（按当前电价计算）
        # 这反映了：用RE制氢的机会成本 + 从电网购电的直接成本
        cost_h2_production = h2_production_load_total * price * Config.dt  # kW × $/kWh × h = $
        
        # 电网交互成本/收入（净购售电）
        cost_grid_purchase = 0
        revenue_grid = 0
        if net_power < 0:
            # 从电网购电
            cost_grid_purchase = abs(net_power) * price * Config.dt
        else:
            # 向电网售电
            revenue_grid = net_power * price * Config.electricity_price_sell_coef * Config.dt
        
        # 总电网成本 = 制氢生产成本 + 净购电成本（如果有的话）
        # 注意：制氢生产成本已经包含了所有用电，net_power是在此之后的净交互
        # 所以总成本应该是制氢生产成本
        cost_grid = cost_h2_production  # 主要成本
        
        # 如果还需要从电网额外购电（net_power < 0），说明RE+FC不够
        # 这部分已经包含在制氢生产成本中了，不需要重复计算

        # 惩罚
        penalty = service_penalty  # 未满足服务惩罚
        if shortage_h2 > 0: 
            penalty += shortage_h2 * Config.penalty_unmet_h2_demand
        if excess_h2 > 0: 
            penalty += excess_h2 * 10

        # 总收益 (包含储能套利奖励)
        step_reward = revenue_ev + revenue_fcev + revenue_grid + arbitrage_bonus - cost_grid - penalty
        step_profit = revenue_ev + revenue_fcev + revenue_grid + arbitrage_bonus - cost_grid

        self.current_step += 1
        done = self.current_step >= Config.episode_length

        # --- I2S 约束 (终端惩罚) ---
        if done and self.enable_i2s_constraint:
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
            self.battery.get_soc(),  # v2.6: 电池SOC
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
            "comp_c1_power": c1_power,  # v3.1: 单独追踪C1功耗
            "comp_c2_power": c2_power,  # v3.1: 单独追踪C2功耗
            "comp_c3_power": c3_power,  # v3.1: 单独追踪C3功耗
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
            # 绿氢统计
            "green_h2_ratio": green_h2_ratio,
            "power_from_re": power_from_re,
            "power_from_grid": power_from_grid,
            "ele_power": ele_actual_power,
            # 储能套利统计
            "arbitrage_bonus": arbitrage_bonus,
            "price": price,
            # 电池储能统计 (v2.6新增)
            "battery_soc": self.battery.get_soc(),
            "battery_charge_power": battery_charge_power,
            "battery_discharge_power": battery_discharge_power,
            "battery_net_power": battery_net_power,
            # v3.1: 收益分解
            "revenue_ev": ev_revenue,
            "revenue_fcev": fcev_revenue,
            "revenue_grid": revenue_grid,
            "cost_grid": cost_grid,
            # 兼容旧版
            "h2_sold": fcev_h2_demand * supply_ratio  # 近似值
        }

        return self.state, step_reward / 100.0, done, info
