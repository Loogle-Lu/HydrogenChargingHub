import gym
import numpy as np
from gym import spaces
from config import Config
from components import (
    Electrolyzer, MultiStageCompressorSystem, MultiTankStorage, FuelCell, LinearChiller,
    FCEVDemandGenerator, FCEVServiceStation
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
        self.data_loader = DataLoader()
        
        # FCEV需求建模 (已移除EV)
        self.demand_generator = FCEVDemandGenerator()
        self.service_station = FCEVServiceStation()

        self.current_step = 0
        # 动作空间 (6维):
        #   [ele_ratio, fc_ratio, c1_cool, c2_cool, c3_pressure_bias, bypass_bias]
        #   ele_ratio:        电解槽功率比        — 低价时多制氢、高价时少制氢
        #   fc_ratio:         FC 功率比           — 高价时 H₂→电卖网、FCEV 不足时保持 0
        #   c1_cool:          C1 级间冷却强度     — 0=轻度(省冷却电) 1=深度(省压缩功)
        #   c2_cool:          C2 级间冷却强度     — 同上
        #   c3_pressure_bias: C3 APC 偏置        — 700-bar 充装压力精细控制
        #   bypass_bias:      旁路积极程度        — 减少空压浪费
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)

        # 观察空间 (11维):
        #   [price_norm, re_ratio,
        #    T1_soc, T2_soc, T3_1_soc, T3_2_soc, T3_3_soc,
        #    queue_350, queue_700,
        #    hour_sin, hour_cos]
        #
        #   price_norm : 当前电价/price_max         ← 低买高卖的核心信号
        #   re_ratio   : 可再生能源/电解槽额定功率  ← 区分绿电(边际成本≈0)与网电
        #   T3_1~3_soc : 350/700-bar 服务罐备货状态 ← FCEV 能否立即被服务
        #   queue_350/700: 两类 FCEV 排队量         ← 当前服务压力，决定是否紧急补气
        #   hour_sin/cos : 时段循环编码              ← 捕捉早/晚高峰需求规律
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
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
        self.service_station = FCEVServiceStation()  # 重置服务站
        self.demand_generator = FCEVDemandGenerator()  # 重置需求生成器
        self.ele.reset()  # 重置电解槽统计
        self.comp_system.reset()  # v3.1: 重置压缩机统计
        self._prev_fcev_served = 0  # v4.4: 吞吐量激励追踪
        
        self.current_data = self.data_loader.get_step_data(self.current_step)

        re_power = self.current_data["wind"] + self.current_data["pv"]
        hour_of_day = int((self.current_step * Config.dt) % 24)

        self.state = np.array([
            # --- 市场信号 ---
            self.current_data["price"] / Config.price_max,              # 电价归一化 [0,1+]
            re_power / Config.ele_max_power,                            # 可再生能源比例 [0,1+]
            # --- 储罐状态 ---
            self.storage.t1.get_soc(),                                  # T1 SOC
            self.storage.t2.get_soc(),                                  # T2 SOC
            self.storage.t3_1.get_soc(),                                # T3₁ SOC (200 bar)
            self.storage.t3_2.get_soc(),                                # T3₂ SOC (350 bar)
            self.storage.t3_3.get_soc(),                                # T3₃ SOC (500 bar)
            # --- FCEV 需求压力 ---
            0.0,                                                        # 350-bar 排队 (初始为0)
            0.0,                                                        # 700-bar 排队 (初始为0)
            # --- 时段循环编码 ---
            np.sin(2 * np.pi * hour_of_day / 24.0),                    # hour_sin
            np.cos(2 * np.pi * hour_of_day / 24.0),                    # hour_cos
        ], dtype=np.float32)
        return self.state
    
    def _compute_comp_block(self, h2_produced, fcev_h2_demand_700,
                             c1_load, c2_load, bypass_bias, c3_pressure_bias, price):
        """
        计算三级压缩链的功耗与流量。

        设计原则: 流量由储罐需求驱动(最大化吞吐量), RL 控制 VSD 工作点 / 旁路 / APC
        - c1_load: VSD 效率映射参数 (传给 compute_c1 作为 cooling_intensity 代理)
        - c2_load: VSD 效率映射参数
        - bypass_bias: 旁路积极程度
        - c3_pressure_bias: C3 APC 偏置

        返回: (c1_power, c2_power, c3_power,
                c1_heat,  c2_heat,  c3_heat,
                c1_flow,  c2_flow,  c3_flow)
        """
        t1_soc = self.storage.t1.get_soc()
        t2_soc = self.storage.t2.get_soc()
        t3_avg_pressure = (self.storage.t3_1.max_pressure * self.storage.t3_1.get_soc() +
                           self.storage.t3_2.max_pressure * self.storage.t3_2.get_soc() +
                           self.storage.t3_3.max_pressure * self.storage.t3_3.get_soc()) / 3.0
        t3_3_pressure = self.storage.t3_3.max_pressure * self.storage.t3_3.get_soc()
        avg_fcev_700_sog = self.service_station.current_fcev_700_avg_sog

        # C1: 流量由制氢量和 T1 SOC 驱动 (尽快转运, 防 T1 溢出)
        # c1_load 映射为级间冷却强度 (0=轻度, 1=深度), 不影响流量
        c1_flow = h2_produced * min(1.0, max(0.5, t1_soc))
        c1_flow = min(c1_flow, Config.c1_max_flow)
        t2_pressure = Config.c1_output_pressure * t2_soc
        c1_power, c1_heat = self.comp_system.compute_c1(
            c1_flow, tank_pressure=t2_pressure, electricity_price=price,
            cooling_intensity=c1_load, bypass_bias=bypass_bias)

        # C2: 流量由 T2 SOC 和 T3 deficit 驱动 (尽快备货到服务罐)
        # c2_load 映射为级间冷却强度
        t3_avg_soc = (self.storage.t3_1.get_soc() + self.storage.t3_2.get_soc() +
                      self.storage.t3_3.get_soc()) / 3.0
        t3_deficit = max(0.0, 0.9 - t3_avg_soc)
        c2_flow = c1_flow * min(1.0, max(0.4, t2_soc)) * min(1.0, 0.5 + t3_deficit)
        c2_flow = min(c2_flow, Config.c2_max_flow)
        c2_power, c2_heat = self.comp_system.compute_c2(
            c2_flow, tank_pressure=t3_avg_pressure, electricity_price=price,
            cooling_intensity=c2_load, bypass_bias=bypass_bias)

        # C3: 需求驱动, 仅从 T3₃ 取气, 服务所有 700-bar 车型
        c3_flow = min(fcev_h2_demand_700, Config.c3_max_flow)
        c3_power, c3_heat = self.comp_system.compute_c3(
            c3_flow, avg_fcev_sog=avg_fcev_700_sog, tank_pressure=t3_3_pressure,
            electricity_price=price, bypass_bias=bypass_bias,
            c3_pressure_bias=c3_pressure_bias)

        return (c1_power, c2_power, c3_power,
                c1_heat,  c2_heat,  c3_heat,
                c1_flow,  c2_flow,  c3_flow)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).flatten()
        # 6维动作: [ele_ratio, fc_ratio, c1_cool, c2_cool, c3_pressure_bias, bypass_bias]
        defaults = [0.7, 0.5, 0.5, 0.5, 0.5, 0.5]
        if len(action) < 6:
            action = np.concatenate([action, np.array(defaults[len(action):6])])
        action = action[:6]

        ele_ratio        = np.clip(action[0], 0, 1)
        fc_ratio         = np.clip(action[1], 0, 1)
        c1_load          = np.clip(action[2], 0, 1)   # C1 级间冷却强度 (0=轻度, 1=深度)
        c2_load          = np.clip(action[3], 0, 1)   # C2 级间冷却强度
        c3_pressure_bias = np.clip(action[4], 0, 1)   # C3 APC 偏置 (0=降压, 1=升压)
        bypass_bias      = np.clip(action[5], 0, 1)   # 旁路积极程度

        price = self.current_data["price"]
        re_gen = self.current_data["wind"] + self.current_data["pv"]
        hour_of_day = int((self.current_step * Config.dt) % 24)

        # --- 1. 生成FCEV到达 ---
        fcev_arrivals = self.demand_generator.generate_vehicles(self.current_step)
        self.service_station.add_vehicles(fcev_arrivals)

        # --- 2. 制氢 (Electrolyzer -> T1) ---
        current_soc = self.storage.get_total_soc()
        ele_power_target = ele_ratio * Config.ele_max_power
        h2_produced, ele_actual_power, green_h2_ratio, power_from_re, power_from_grid = \
            self.ele.compute(ele_power_target, re_gen)

        # --- 3. 燃料电池需要的H₂量 (只从 T3₁ 取气) ---
        fc_power_target = fc_ratio * Config.fc_max_power
        h2_needed_for_fc = fc_power_target / self.fc.efficiency

        # --- 4. 计算各压力级别的可用氢气 ---
        # 350-bar 车: T3₁ + T3₂ (T3₂ 优先, T3₁ 补充)
        h2_avail_350 = (self.storage.t3_1.level + self.storage.t3_2.level) * 0.8
        # 700-bar 车: T3₃ + T3₂ + T3₁ 级联 (T3₃→T3₂→T3₁ 经 C3 升压)
        h2_avail_700 = (self.storage.t3_1.level + self.storage.t3_2.level + self.storage.t3_3.level) * 0.8
        
        # --- 5. 服务 FCEV 需求 (350-bar 和 700-bar 分离) ---
        fcev_h2_demand_350, fcev_h2_demand_700, fcev_revenue, service_penalty = \
            self.service_station.step(h2_avail_350, h2_avail_700, Config.dt)
        
        # --- 6. 多级压缩链 ---
        (c1_power, c2_power, c3_power,
         c1_heat, c2_heat, c3_heat,
         c1_flow, c2_flow, c3_flow) = self._compute_comp_block(
            h2_produced, fcev_h2_demand_700,
            c1_load, c2_load, bypass_bias, c3_pressure_bias, price
        )
        total_comp_heat = c1_heat + c2_heat + c3_heat
        
        # --- 7. Chiller (固定 0.8 制冷强度) ---
        chiller_power = self.chiller.compute_power(total_comp_heat, chiller_ratio=0.8)
        
        # --- 8. 总电力负荷 ---
        h2_production_load = ele_actual_power + c1_power + c2_power + c3_power + chiller_power
        total_load = h2_production_load

        # --- 9. 多储罐系统更新 ---
        soc, excess_h2, shortage_h2 = self.storage.step_all(
            h2_from_ele=h2_produced,
            h2_to_c1=c1_flow,
            h2_from_c1=c1_flow,
            h2_to_c2=c2_flow,
            h2_from_c2=c2_flow,
            h2_demand_350=fcev_h2_demand_350,  # 350-bar: T3₂→T3₁ 优先
            h2_for_fc=h2_needed_for_fc,         # FC: T3₁ only
            h2_to_c3=c3_flow,                   # C3 取气: T3₃→T3₂→T3₁ 级联
            h2_from_c3=c3_flow
        )

        # --- 10. 燃料电池实际发电 (T3₁ 供气, 缺氢时按比例降低) ---
        supply_ratio = 1.0
        if shortage_h2 > 0:
            total_demand = fcev_h2_demand_350 + fcev_h2_demand_700 + h2_needed_for_fc
            if total_demand > 0:
                supply_ratio = max(0.0, (total_demand - shortage_h2) / total_demand)

        real_fc_h2_used = h2_needed_for_fc * supply_ratio
        fc_actual_power, _ = self.fc.compute(real_fc_h2_used)
        fcev_h2_demand = fcev_h2_demand_350 + fcev_h2_demand_700  # 合并用于统计

        # --- 11. 电网互动与财务 (已移除EV) ---
        # 供应: FC(H2转电) + 剩余可再生能源 | 需求: 制氢链负荷
        available_re_for_grid = re_gen - power_from_re
        own_supply = fc_actual_power + available_re_for_grid  # 自有供电 (FC+RE)
        load_non_ele = total_load - ele_actual_power  # 制氢链(除电解槽)

        # 自有电优先满足制氢链，余电可售
        h2_load_from_own = min(h2_production_load, own_supply)
        h2_load_from_grid = max(0.0, h2_production_load - h2_load_from_own)
        grid_purchase_bus = h2_load_from_grid  # 母线侧购电
        net_power = own_supply - load_non_ele  # 正=余电可售，负=需购电

        # 电网成本: 电解槽购电 power_from_grid + 母线侧购电
        cost_grid = (power_from_grid * Config.dt + grid_purchase_bus * Config.dt) * price

        # 收入来源 (已移除EV收入)
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
            if 0.4 <= current_soc <= 0.6:
                arbitrage_bonus += Config.soc_health_bonus
        
        # 电网交互成本/收入 (v4.0: 支持 zero_export / local_grid 两种模式)
        revenue_grid_raw = 0
        if net_power > 0:
            if Config.grid_export_mode == "local_grid":
                # 卖电给地电网：收购价通常低于零售价
                revenue_grid_raw = net_power * price * Config.local_grid_feedin_ratio * Config.dt
            elif Config.grid_export_mode == "zero_export":
                revenue_grid_raw = 0  # 弃电，不产生收益
            else:
                revenue_grid_raw = net_power * price * Config.electricity_price_sell_coef * Config.dt
        # v4.0 防 exploit: 售电收入上限，避免"只卖电不卖氢"刷高 Profit
        service_revenue = revenue_fcev
        revenue_grid_cap = service_revenue * Config.max_grid_revenue_ratio
        revenue_grid = min(revenue_grid_raw, revenue_grid_cap)
        
        # 惩罚
        penalty = service_penalty  # 未满足服务惩罚 (FCEV 离开队列未被服务)
        if shortage_h2 > 0:
            # 缺氢惩罚 >> FCEV 服务单价，强制 agent 优先保证 T3 备货
            penalty += shortage_h2 * Config.penalty_unmet_h2_demand
        if excess_h2 > 0:
            penalty += excess_h2 * 10
        # FCEV 排队等待惩罚: 促使 agent 提前补气, 700-bar 权重 1.5x (收益更高)
        # 此惩罚仅在有排队时触发，正常服务时为 0，不抑制吞吐
        cur_stats = self.service_station.get_statistics()
        penalty += cur_stats['fcev_queue_350'] * Config.penalty_fcev_wait
        penalty += cur_stats['fcev_queue_700'] * Config.penalty_fcev_wait * 1.5

        # 压缩机效率激励 (v4.4 重设计): 奖励 = 节能量 × 电价 (真实经济价值)
        # 流量已解耦, 此激励奖励 "同等吞吐下省电", 不再与吞吐量对抗
        comp_efficiency_bonus = 0.0
        if Config.enable_comp_eff_bonus:
            total_h2_compressed_step = (c1_flow + c2_flow + c3_flow) * Config.dt  # kg
            total_comp_kw = c1_power + c2_power + c3_power
            if total_h2_compressed_step > 0.01:
                actual_kWh_per_kg = total_comp_kw * Config.dt / total_h2_compressed_step
                energy_saved_per_kg = max(0.0, Config.comp_eff_ref_kWh_per_kg - actual_kWh_per_kg)
                comp_efficiency_bonus = (energy_saved_per_kg * total_h2_compressed_step
                                         * price * Config.comp_eff_bonus_coef)

        # FCEV 服务吞吐量激励: 直接奖励服务更多车辆
        throughput_bonus = 0.0
        fcev_served_this_step = cur_stats['fcev_served'] - getattr(self, '_prev_fcev_served', 0)
        if fcev_served_this_step > 0:
            throughput_bonus = fcev_served_this_step * Config.fcev_throughput_bonus
        self._prev_fcev_served = cur_stats['fcev_served']

        # =========================================================================
        # 区分 "真实利润(Profit)" 和 "强化学习奖励(Reward)"
        # =========================================================================
        # 1. 真实经济利润 (绘图/评估): 真金白银，不含人工奖励
        #    Profit = FCEV加氢收入 + FC卖电收入 - 购电成本
        step_profit = revenue_fcev + revenue_grid - cost_grid

        # 2. 强化学习奖励 (训练): Profit + 套利引导 + 效率引导 + 吞吐激励 - 惩罚
        #    收益优先级: FCEV(18/12$/kg) >> FC电 >> 压缩效率 >> 套利
        step_reward = (step_profit + arbitrage_bonus + comp_efficiency_bonus
                       + throughput_bonus - penalty)
        # =========================================================================

        self.current_step += 1
        done = self.current_step >= Config.episode_length

        # --- I2S 约束 (终端惩罚) ---
        if done and self.enable_i2s_constraint:
            final_soc = self.storage.get_total_soc()
            init_soc = Config.storage_initial
            i2s_penalty = abs(final_soc - init_soc) * Config.i2s_penalty_weight
            step_reward -= i2s_penalty

        # --- 奖励裁剪，抑制极端值以稳定训练 ---
        step_reward = np.clip(step_reward, -5000.0, 5000.0)

        # --- 更新状态 ---
        self.current_data = self.data_loader.get_step_data(self.current_step)
        next_re = self.current_data["wind"] + self.current_data["pv"]
        next_hour = int((self.current_step * Config.dt) % 24)

        station_stats = cur_stats  # 复用刚取到的统计，避免重复调用
        q350 = station_stats['fcev_queue_350']
        q700 = station_stats['fcev_queue_700']

        self.state = np.array([
            # --- 市场信号 (下一步, 已提前知道) ---
            self.current_data["price"] / Config.price_max,              # 电价归一化
            next_re / Config.ele_max_power,                             # 可再生能源比例
            # --- 储罐状态 ---
            self.storage.t1.get_soc(),                                  # T1 SOC
            self.storage.t2.get_soc(),                                  # T2 SOC
            self.storage.t3_1.get_soc(),                                # T3₁ SOC (200 bar)
            self.storage.t3_2.get_soc(),                                # T3₂ SOC (350 bar)
            self.storage.t3_3.get_soc(),                                # T3₃ SOC (500 bar)
            # --- FCEV 需求压力 (分类，高价值 700-bar 可见) ---
            min(q350 / Config.queue_max, 1.0),                          # 350-bar 排队归一化
            min(q700 / Config.queue_max, 1.0),                          # 700-bar 排队归一化
            # --- 时段循环编码 ---
            np.sin(2 * np.pi * next_hour / 24.0),                       # hour_sin
            np.cos(2 * np.pi * next_hour / 24.0),                       # hour_cos
        ], dtype=np.float32)

        info = {
            "soc": soc,
            "profit": step_profit,
            "reward": step_reward,
            "net_power": net_power,
            "re_gen": re_gen,
            "fc_power": fc_actual_power,
            "h2_production_load": h2_production_load,
            "total_load": total_load,
            "comp_power": c1_power + c2_power + c3_power,
            "comp_c1_power": c1_power,
            "comp_c2_power": c2_power,
            "comp_c3_power": c3_power,
            "chiller_power": chiller_power,
            "bypass_activations": self.comp_system.bypass_activations.copy(),
            "tank_socs": self.storage.get_tank_socs(),
            # FCEV需求分解
            "fcev_served": station_stats['fcev_served'],
            "fcev_revenue": fcev_revenue,
            "fcev_h2_demand_350": fcev_h2_demand_350,
            "fcev_h2_demand_700": fcev_h2_demand_700,
            "fcev_queue": station_stats['fcev_queue_length'],
            "fcev_queue_350": station_stats['fcev_queue_350'],
            "fcev_queue_700": station_stats['fcev_queue_700'],
            "vehicles_delayed": station_stats['vehicles_delayed'],
            # 绿氢统计
            "green_h2_ratio": green_h2_ratio,
            "power_from_re": power_from_re,
            "power_from_grid": power_from_grid,
            "ele_power": ele_actual_power,
            # 套利 & 压缩效率激励
            "arbitrage_bonus": arbitrage_bonus,
            "comp_efficiency_bonus": comp_efficiency_bonus,
            "price": price,
            # 收益分解
            "revenue_fcev": revenue_fcev,
            "revenue_grid": revenue_grid,
            "revenue_grid_raw": revenue_grid_raw,
            "cost_grid": cost_grid,
            # 控制动作 (6维)
            "c1_load": c1_load,
            "c2_load": c2_load,
            "bypass_bias": bypass_bias,
            "c3_pressure_bias": c3_pressure_bias,
            "h2_sold": fcev_h2_demand * supply_ratio
        }

        # v3.5: 奖励缩放 (100) 保持梯度尺度适中
        return self.state, step_reward / 100.0, done, info