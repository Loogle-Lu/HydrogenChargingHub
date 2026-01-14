import numpy as np
from config import Config


class Electrolyzer:
    """
    线性电解槽: 输入功率 -> 产氢量
    支持可变功率阈值策略，区分绿氢和灰氢生产
    """

    def __init__(self):
        self.max_power = Config.ele_max_power
        self.efficiency = Config.ele_efficiency
        
        # 绿氢生产统计
        self.total_green_h2 = 0.0  # kg
        self.total_grid_h2 = 0.0   # kg
        self.total_green_energy = 0.0  # kWh
        self.total_grid_energy = 0.0   # kWh

    def compute(self, power_input, re_available=0.0, power_threshold=None):
        """
        计算产氢量，并区分绿氢和灰氢
        
        参数:
        - power_input: 目标输入功率 (kW)
        - re_available: 可用可再生能源功率 (kW)
        - power_threshold: 功率阈值 (kW)，超过此值优先使用RE
        
        返回:
        - h2_flow_kg: 产氢量 (kg/h)
        - actual_power: 实际功率 (kW)
        - green_h2_ratio: 绿氢占比 (0-1)
        - power_from_re: 来自可再生能源的功率 (kW)
        - power_from_grid: 来自电网的功率 (kW)
        """
        # 限制在最大功率范围内
        power = np.clip(power_input, 0, self.max_power)
        
        if not Config.enable_threshold_strategy or power_threshold is None:
            # 不启用阈值策略，正常运行
            h2_flow_kg = power / self.efficiency
            return h2_flow_kg, power, 0.0, 0.0, power
        
        # 应用阈值策略
        if re_available >= power_threshold:
            # 可再生能源充足，优先使用RE生产绿氢
            power_from_re = min(power, re_available)
            power_from_grid = max(0, power - power_from_re)
        else:
            # 可再生能源不足阈值，可以使用电网能源
            # 但仍优先使用可用的RE
            power_from_re = min(power, re_available)
            power_from_grid = power - power_from_re
        
        # 计算绿氢和灰氢产量
        green_h2_flow = power_from_re / self.efficiency  # kg/h
        grid_h2_flow = power_from_grid / self.efficiency  # kg/h
        total_h2_flow = green_h2_flow + grid_h2_flow
        
        # 计算绿氢占比
        green_h2_ratio = power_from_re / power if power > 0 else 0.0
        
        # 累积统计 (假设dt=Config.dt)
        self.total_green_h2 += green_h2_flow * Config.dt
        self.total_grid_h2 += grid_h2_flow * Config.dt
        self.total_green_energy += power_from_re * Config.dt
        self.total_grid_energy += power_from_grid * Config.dt
        
        return total_h2_flow, power, green_h2_ratio, power_from_re, power_from_grid
    
    def get_statistics(self):
        """获取绿氢生产统计"""
        total_h2 = self.total_green_h2 + self.total_grid_h2
        green_h2_percentage = (self.total_green_h2 / total_h2 * 100) if total_h2 > 0 else 0.0
        
        return {
            'total_green_h2_kg': self.total_green_h2,
            'total_grid_h2_kg': self.total_grid_h2,
            'total_h2_kg': total_h2,
            'green_h2_percentage': green_h2_percentage,
            'total_green_energy_kwh': self.total_green_energy,
            'total_grid_energy_kwh': self.total_grid_energy
        }
    
    def reset(self):
        """重置统计数据"""
        self.total_green_h2 = 0.0
        self.total_grid_h2 = 0.0
        self.total_green_energy = 0.0
        self.total_grid_energy = 0.0


class Compressor:
    """
    通用压缩机类 - 单级或两级压缩
    基于等熵压缩公式实现
    """
    def __init__(self, p_in, p_out, max_flow, efficiency=None, name="Compressor"):
        self.name = name
        self.p_in = p_in  # bar
        self.p_out = p_out  # bar
        self.max_flow = max_flow  # kg/h
        self.eta = efficiency or Config.comp_efficiency
        self.gamma = Config.H2_gamma
        self.R = Config.H2_R
        self.T_in = Config.T_in
        self.cp = self.gamma * self.R / (self.gamma - 1)
        self.exponent = (self.gamma - 1) / self.gamma
        
        # 确定是否需要两级压缩 (压比>3时)
        self.pressure_ratio = p_out / p_in
        self.use_two_stage = self.pressure_ratio > 3.0
        
        if self.use_two_stage:
            # 最佳中间压力
            self.p_mid = np.sqrt(p_in * p_out)
        
    def compute_power(self, mass_flow_kg_h):
        """计算压缩功耗和产生的热量"""
        if mass_flow_kg_h <= 0:
            return 0.0, 0.0
        
        # 限制最大流量
        actual_flow = min(mass_flow_kg_h, self.max_flow)
        m_dot = actual_flow / 3600.0  # kg/s
        
        if self.use_two_stage:
            # 两级压缩
            # 第一级: p_in -> p_mid
            term1 = (self.p_mid / self.p_in) ** self.exponent - 1
            work_stage1 = self.cp * self.T_in * term1 / self.eta
            t_out_1 = self.T_in * (1 + term1)
            
            # 第二级: p_mid -> p_out (假设级间冷却回T_in)
            term2 = (self.p_out / self.p_mid) ** self.exponent - 1
            work_stage2 = self.cp * self.T_in * term2 / self.eta
            t_out_2 = self.T_in * (1 + term2)
            
            total_work_j_kg = work_stage1 + work_stage2
            
            # 热负荷 (两级的热量累加)
            heat_1 = m_dot * self.cp * (t_out_1 - Config.target_temp)
            heat_2 = m_dot * self.cp * (t_out_2 - Config.target_temp)
            total_heat_kw = (heat_1 + heat_2) / 1000.0
            
        else:
            # 单级压缩
            term = (self.p_out / self.p_in) ** self.exponent - 1
            total_work_j_kg = self.cp * self.T_in * term / self.eta
            t_out = self.T_in * (1 + term)
            
            heat = m_dot * self.cp * (t_out - Config.target_temp)
            total_heat_kw = heat / 1000.0
        
        power_kw = m_dot * total_work_j_kg / 1000.0
        return power_kw, max(0, total_heat_kw)


class MultiStageCompressorSystem:
    """
    多级级联压缩机系统 (根据HRS架构图)
    C1: 2 bar -> 35 bar
    C2: 35 bar -> 500 bar (级联充装)
    C3: 500 bar -> 700 bar (LDFV快充)
    """
    def __init__(self):
        self.c1 = Compressor(
            Config.c1_input_pressure,
            Config.c1_output_pressure,
            Config.c1_max_flow,
            name="C1"
        )
        self.c2 = Compressor(
            Config.c2_input_pressure,
            Config.c2_output_pressure,
            Config.c2_max_flow,
            name="C2"
        )
        self.c3 = Compressor(
            Config.c3_input_pressure,
            Config.c3_output_pressure,
            Config.c3_max_flow,
            name="C3"
        )
    
    def compute_c1(self, mass_flow):
        """C1压缩: T1(2bar) -> T2(35bar)"""
        return self.c1.compute_power(mass_flow)
    
    def compute_c2(self, mass_flow):
        """C2压缩: T2(35bar) -> T3系统(500bar)"""
        return self.c2.compute_power(mass_flow)
    
    def compute_c3(self, mass_flow):
        """C3压缩: T3/T4 -> D2 LDFV(700bar)"""
        return self.c3.compute_power(mass_flow)
    
    def compute_total_power(self, flow_c1, flow_c2, flow_c3):
        """计算所有压缩机的总功耗和热负荷"""
        p1, h1 = self.compute_c1(flow_c1)
        p2, h2 = self.compute_c2(flow_c2)
        p3, h3 = self.compute_c3(flow_c3)
        return (p1 + p2 + p3), (h1 + h2 + h3)


class NonlinearChiller:
    """
    复杂非线性冷却机模型
    考虑:
    1. 部分负荷率(PLR)对COP的非线性影响
    2. 环境温度对性能的影响
    3. 最小负荷限制和启停惩罚
    4. 真实的性能曲线
    """
    def __init__(self):
        self.rated_capacity = Config.chiller_rated_capacity  # kW
        self.rated_cop = Config.chiller_rated_cop
        self.min_plr = Config.chiller_min_plr
        
        # 性能曲线系数
        self.cop_plr_coef = Config.chiller_cop_plr_coef
        self.temp_coef = Config.chiller_temp_coef
        
        # 运行状态
        self.is_running = False
        self.runtime = 0.0  # hours
        self.startup_energy_cost = 0.0
        
    def _compute_cop_plr(self, plr):
        """
        计算部分负荷率下的COP修正系数
        基于三次多项式拟合实际冷水机组性能曲线
        COP_factor = a0 + a1*PLR + a2*PLR^2 + a3*PLR^3
        """
        plr = np.clip(plr, self.min_plr, 1.0)
        a0, a1, a2, a3 = self.cop_plr_coef
        cop_factor = a0 + a1*plr + a2*(plr**2) + a3*(plr**3)
        # 确保COP因子在合理范围内
        return np.clip(cop_factor, 0.3, 1.2)
    
    def _compute_cop_temp(self, ambient_temp):
        """
        计算环境温度对COP的影响
        COP_temp_factor = b0 + b1*T + b2*T^2
        """
        b0, b1, b2 = self.temp_coef
        temp_factor = b0 + b1*ambient_temp + b2*(ambient_temp**2)
        return np.clip(temp_factor, 0.5, 1.5)
    
    def compute_power(self, heat_load_kw, ambient_temp=None, dt=None):
        """
        计算冷却所需功率(非线性)
        
        参数:
        - heat_load_kw: 需要移除的热负荷 (kW)
        - ambient_temp: 环境温度 (K), 默认使用名义温度
        - dt: 时间步长 (hours), 用于计算启停
        
        返回:
        - power_kw: 冷却机功耗 (kW)
        """
        if heat_load_kw <= 0:
            self.is_running = False
            self.runtime = 0.0
            return 0.0
        
        # 使用默认环境温度
        if ambient_temp is None:
            ambient_temp = Config.ambient_temp_nominal
        if dt is None:
            dt = Config.dt
        
        # 计算部分负荷率
        plr = heat_load_kw / self.rated_capacity
        
        # 如果低于最小负荷，按最小负荷运行
        if plr < self.min_plr:
            actual_plr = self.min_plr
            # 实际只需要处理部分热量，但机组按最小负荷运行
        else:
            actual_plr = min(plr, 1.0)
        
        # 计算实际COP
        cop_plr_factor = self._compute_cop_plr(actual_plr)
        cop_temp_factor = self._compute_cop_temp(ambient_temp)
        actual_cop = self.rated_cop * cop_plr_factor * cop_temp_factor
        
        # 基础功耗
        base_power = (actual_plr * self.rated_capacity) / actual_cop
        
        # 启停惩罚
        startup_penalty = 0.0
        if not self.is_running:
            # 首次启动，产生启动能耗
            startup_penalty = Config.chiller_startup_energy / dt  # kW
            self.is_running = True
            self.runtime = 0.0
        
        self.runtime += dt
        
        total_power = base_power + startup_penalty
        return total_power
    
    def reset(self):
        """重置冷却机状态"""
        self.is_running = False
        self.runtime = 0.0
        self.startup_energy_cost = 0.0


class HydrogenTank:
    """单个储氢罐"""
    def __init__(self, capacity_kg, max_pressure, initial_soc, name="Tank"):
        self.name = name
        self.capacity = capacity_kg
        self.max_pressure = max_pressure  # bar
        self.level = initial_soc * capacity_kg
        self.min_lvl = Config.storage_min_level * capacity_kg
        self.max_lvl = Config.storage_max_level * capacity_kg
    
    def step(self, inflow_kg_h, outflow_kg_h, dt=None):
        """
        更新储罐状态
        参数为 kg/h 的流量
        """
        if dt is None:
            dt = Config.dt
        
        delta = (inflow_kg_h - outflow_kg_h) * dt
        new_level = self.level + delta
        excess = 0.0
        shortage = 0.0
        
        if new_level > self.max_lvl:
            excess = new_level - self.max_lvl
            new_level = self.max_lvl
        elif new_level < self.min_lvl:
            shortage = self.min_lvl - new_level
            new_level = self.min_lvl
        
        self.level = new_level
        soc = self.level / self.capacity
        return soc, excess, shortage
    
    def get_soc(self):
        return self.level / self.capacity


class MultiTankStorage:
    """
    多储罐系统 (根据HRS架构)
    T1: 低压缓冲罐 (2 bar)
    T2: 中压储罐 (35 bar)
    T3_1, T3_2, T3_3: 级联高压储罐组 (500 bar)
    T4: 超高压缓冲罐 (900 bar, for LDFV)
    """
    def __init__(self):
        self.t1 = HydrogenTank(
            Config.t1_capacity_kg,
            Config.t1_max_pressure,
            Config.t1_initial_soc,
            name="T1"
        )
        self.t2 = HydrogenTank(
            Config.t2_capacity_kg,
            Config.t2_max_pressure,
            Config.t2_initial_soc,
            name="T2"
        )
        # T3级联系统
        self.t3_1 = HydrogenTank(
            Config.t3_1_capacity_kg,
            Config.t3_max_pressure,
            Config.t3_initial_soc,
            name="T3_1"
        )
        self.t3_2 = HydrogenTank(
            Config.t3_2_capacity_kg,
            Config.t3_max_pressure,
            Config.t3_initial_soc,
            name="T3_2"
        )
        self.t3_3 = HydrogenTank(
            Config.t3_3_capacity_kg,
            Config.t3_max_pressure,
            Config.t3_initial_soc,
            name="T3_3"
        )
        self.t4 = HydrogenTank(
            Config.t4_capacity_kg,
            Config.t4_max_pressure,
            Config.t4_initial_soc,
            name="T4"
        )
    
    def step_all(self, h2_from_ele, h2_to_c1, h2_from_c1, h2_to_c2, 
                 h2_from_c2, h2_demand, h2_to_c3, h2_from_c3):
        """
        更新所有储罐状态
        简化的物质流动:
        - T1: 接收电解槽产氢, 输出到C1
        - T2: 接收C1输出, 输出到C2
        - T3系统: 接收C2输出(级联充装), 输出到需求和C3
        - T4: 接收C3输出, 输出到LDFV快充需求
        """
        # T1步进
        soc_t1, excess_t1, short_t1 = self.t1.step(h2_from_ele, h2_to_c1)
        
        # T2步进
        soc_t2, excess_t2, short_t2 = self.t2.step(h2_from_c1, h2_to_c2)
        
        # T3级联系统 - 均匀分配充装
        h2_to_t3_each = h2_from_c2 / 3.0
        soc_t3_1, excess_t3_1, short_t3_1 = self.t3_1.step(h2_to_t3_each, h2_demand/3.0 + h2_to_c3/3.0)
        soc_t3_2, excess_t3_2, short_t3_2 = self.t3_2.step(h2_to_t3_each, h2_demand/3.0 + h2_to_c3/3.0)
        soc_t3_3, excess_t3_3, short_t3_3 = self.t3_3.step(h2_to_t3_each, h2_demand/3.0 + h2_to_c3/3.0)
        
        # T4步进 (接收C3输出，供应高压需求)
        soc_t4, excess_t4, short_t4 = self.t4.step(h2_from_c3, 0)
        
        # 汇总
        total_excess = excess_t1 + excess_t2 + excess_t3_1 + excess_t3_2 + excess_t3_3 + excess_t4
        total_shortage = short_t1 + short_t2 + short_t3_1 + short_t3_2 + short_t3_3 + short_t4
        
        # 计算总体SOC (加权平均)
        total_capacity = (self.t1.capacity + self.t2.capacity + 
                         self.t3_1.capacity + self.t3_2.capacity + self.t3_3.capacity +
                         self.t4.capacity)
        total_mass = (self.t1.level + self.t2.level +
                     self.t3_1.level + self.t3_2.level + self.t3_3.level +
                     self.t4.level)
        overall_soc = total_mass / total_capacity
        
        return overall_soc, total_excess, total_shortage
    
    def get_total_soc(self):
        """获取系统总SOC"""
        total_capacity = (self.t1.capacity + self.t2.capacity + 
                         self.t3_1.capacity + self.t3_2.capacity + self.t3_3.capacity +
                         self.t4.capacity)
        total_mass = (self.t1.level + self.t2.level +
                     self.t3_1.level + self.t3_2.level + self.t3_3.level +
                     self.t4.level)
        return total_mass / total_capacity
    
    def get_tank_socs(self):
        """获取所有储罐的SOC"""
        return {
            't1': self.t1.get_soc(),
            't2': self.t2.get_soc(),
            't3_1': self.t3_1.get_soc(),
            't3_2': self.t3_2.get_soc(),
            't3_3': self.t3_3.get_soc(),
            't4': self.t4.get_soc()
        }


class FuelCell:
    def __init__(self):
        self.max_power = Config.fc_max_power
        self.efficiency = Config.fc_efficiency

    def compute(self, h2_input_kg):
        power_generated = h2_input_kg * self.efficiency
        actual_power = min(power_generated, self.max_power)
        consumed_h2 = actual_power / self.efficiency
        return actual_power, consumed_h2


# ==================== EV/FCEV 需求建模 ====================

class Vehicle:
    """车辆基类"""
    def __init__(self, vehicle_id, arrival_time, vehicle_type):
        self.id = vehicle_id
        self.arrival_time = arrival_time
        self.vehicle_type = vehicle_type  # 'EV' or 'FCEV'
        self.service_start_time = None
        self.service_end_time = None
        self.is_being_served = False
        self.is_completed = False


class EVehicle(Vehicle):
    """
    电动车 (Electric Vehicle)
    支持快充/慢充，可参与需求响应
    """
    def __init__(self, vehicle_id, arrival_time, battery_capacity, soc_initial, 
                 soc_target, charge_mode, dr_flexibility):
        super().__init__(vehicle_id, arrival_time, 'EV')
        self.battery_capacity = battery_capacity  # kWh
        self.soc_initial = soc_initial
        self.soc_target = soc_target
        self.charge_mode = charge_mode  # 'fast', 'slow', 'ultra_fast'
        self.dr_flexibility = dr_flexibility  # 0-1, 需求响应灵活性
        
        # 计算所需能量
        self.energy_needed = (soc_target - soc_initial) * battery_capacity  # kWh
        
        # 根据充电模式确定功率和时间
        if charge_mode == 'ultra_fast':
            self.charge_power = Config.ev_ultra_fast_charge_power
        elif charge_mode == 'fast':
            self.charge_power = Config.ev_fast_charge_power
        else:
            self.charge_power = Config.ev_slow_charge_power
        
        self.estimated_charge_time = self.energy_needed / self.charge_power  # hours
        self.can_be_delayed = False
        
    def check_delay_eligibility(self, current_price):
        """检查是否可以延迟充电 (需求响应)"""
        if not Config.enable_demand_response:
            return False
        
        if current_price > Config.dr_price_threshold_high:
            # 高电价，根据灵活性决定
            if np.random.random() < self.dr_flexibility:
                self.can_be_delayed = True
                return True
        return False
    
    def get_charging_revenue(self):
        """计算充电收入"""
        return self.energy_needed * Config.ev_service_price


class FCEVehicle(Vehicle):
    """
    燃料电池车 (Fuel Cell Electric Vehicle)
    基于SAE J2601协议，快速加氢（3-5分钟）
    """
    def __init__(self, vehicle_id, arrival_time, tank_capacity, sog_initial, sog_target):
        super().__init__(vehicle_id, arrival_time, 'FCEV')
        self.tank_capacity = tank_capacity  # kg H2
        self.sog_initial = sog_initial  # State of Gas (0-1)
        self.sog_target = sog_target
        
        # 计算所需氢气量
        self.h2_needed = (sog_target - sog_initial) * tank_capacity  # kg
        
        # 加氢时间 (分钟)
        self.fill_time_minutes = np.clip(
            Config.fcev_target_fill_time,
            Config.fcev_min_fill_time,
            Config.fcev_max_fill_time
        )
        self.fill_time_hours = self.fill_time_minutes / 60.0
        
        # 加氢流量 (kg/h)
        self.h2_flow_rate = self.h2_needed / self.fill_time_hours
        
        # 需求响应灵活性很低
        self.dr_flexibility = Config.fcev_dr_flexibility
        self.can_be_delayed = False
    
    def check_delay_eligibility(self, h2_availability):
        """FCEV很难延迟，但如果氢气严重短缺可短暂等待"""
        if h2_availability < self.h2_needed * 0.5:
            if np.random.random() < self.dr_flexibility:
                self.can_be_delayed = True
                return True
        return False
    
    def get_refueling_revenue(self):
        """计算加氢收入"""
        return self.h2_needed * Config.fcev_service_price


class MixedDemandGenerator:
    """
    混合需求生成器 (EV + FCEV)
    模拟真实车辆到达模式，考虑时段变化
    """
    def __init__(self):
        self.vehicle_id_counter = 0
        self.ev_fcev_ratio = Config.ev_fcev_ratio
        
    def _get_time_multiplier(self, hour_of_day):
        """根据时段获取到达率倍数"""
        if hour_of_day in Config.peak_morning_hours or hour_of_day in Config.peak_evening_hours:
            return Config.peak_arrival_multiplier
        elif hour_of_day in Config.midday_hours:
            return Config.midday_multiplier
        elif hour_of_day in Config.offpeak_hours:
            return Config.offpeak_multiplier
        else:
            return 1.0
    
    def generate_vehicles(self, current_step):
        """
        生成当前时段到达的车辆
        
        返回:
        - ev_list: EV列表
        - fcev_list: FCEV列表
        """
        # 计算当前时刻
        hour_of_day = int((current_step * Config.dt) % 24)
        
        # 调整到达率
        time_multiplier = self._get_time_multiplier(hour_of_day)
        adjusted_arrival_rate = Config.base_vehicle_arrival_rate * time_multiplier
        
        # 泊松分布生成到达车辆数
        num_arrivals = np.random.poisson(adjusted_arrival_rate * Config.dt)
        
        ev_list = []
        fcev_list = []
        
        for _ in range(num_arrivals):
            self.vehicle_id_counter += 1
            
            # 决定车辆类型
            if np.random.random() < self.ev_fcev_ratio:
                # 生成EV
                ev = self._generate_ev(self.vehicle_id_counter, current_step)
                ev_list.append(ev)
            else:
                # 生成FCEV
                fcev = self._generate_fcev(self.vehicle_id_counter, current_step)
                fcev_list.append(fcev)
        
        return ev_list, fcev_list
    
    def _generate_ev(self, vehicle_id, arrival_time):
        """生成单个EV"""
        # 电池容量
        battery_capacity = np.clip(
            np.random.normal(Config.ev_battery_capacity_mean, Config.ev_battery_capacity_std),
            40.0, 100.0
        )
        
        # 到站SOC
        soc_initial = np.clip(
            np.random.normal(Config.ev_soc_arrival_mean, Config.ev_soc_arrival_std),
            0.05, 0.80
        )
        
        # 充电模式选择
        mode_rand = np.random.random()
        if mode_rand < Config.ev_fast_charge_ratio:
            charge_mode = 'fast'
            soc_target = Config.ev_soc_target_fast
            dr_flex = Config.ev_dr_flexibility_fast
        elif mode_rand < Config.ev_fast_charge_ratio + Config.ev_ultra_fast_ratio:
            charge_mode = 'ultra_fast'
            soc_target = Config.ev_soc_target_fast
            dr_flex = Config.ev_dr_flexibility_fast * 0.5  # 超快充用户更不愿等待
        else:
            charge_mode = 'slow'
            soc_target = Config.ev_soc_target_slow
            dr_flex = Config.ev_dr_flexibility_slow
        
        return EVehicle(
            vehicle_id, arrival_time, battery_capacity,
            soc_initial, soc_target, charge_mode, dr_flex
        )
    
    def _generate_fcev(self, vehicle_id, arrival_time):
        """生成单个FCEV"""
        # 储氢罐容量 (基于Hyundai Nexo)
        tank_capacity = Config.fcev_tank_capacity
        
        # 到站SOG (State of Gas)
        sog_initial = np.clip(
            np.random.normal(Config.fcev_sog_arrival_mean, Config.fcev_sog_arrival_std),
            0.05, 0.50
        )
        
        sog_target = Config.fcev_sog_target
        
        return FCEVehicle(
            vehicle_id, arrival_time, tank_capacity,
            sog_initial, sog_target
        )


class IntegratedServiceStation:
    """
    集成服务站 (EV充电 + FCEV加氢)
    管理车辆队列、服务调度、需求响应
    """
    def __init__(self):
        # EV充电设施
        self.ev_fast_chargers = Config.max_concurrent_ev_fast
        self.ev_slow_chargers = Config.max_concurrent_ev_slow
        
        # FCEV加氢设施
        self.fcev_dispensers = Config.max_concurrent_fcev
        
        # 队列
        self.ev_queue = []
        self.fcev_queue = []
        self.ev_being_served = []
        self.fcev_being_served = []
        
        # 统计
        self.total_ev_served = 0
        self.total_fcev_served = 0
        self.total_ev_revenue = 0.0
        self.total_fcev_revenue = 0.0
        self.total_vehicles_delayed = 0
    
    def add_vehicles(self, ev_list, fcev_list):
        """添加新到达车辆"""
        self.ev_queue.extend(ev_list)
        self.fcev_queue.extend(fcev_list)
    
    def step(self, current_price, h2_available_kg, dt=None):
        """
        执行一个时间步
        
        返回:
        - ev_power_demand: EV充电功率需求 (kW)
        - fcev_h2_demand: FCEV氢气需求 (kg/h)
        - ev_revenue: EV充电收入
        - fcev_revenue: 加氢收入
        - unmet_demand_penalty: 未满足需求惩罚
        """
        if dt is None:
            dt = Config.dt
        
        ev_power_demand = 0.0
        fcev_h2_demand = 0.0
        ev_revenue = 0.0
        fcev_revenue = 0.0
        unmet_penalty = 0.0
        
        # 1. 更新正在服务的车辆
        self._update_serving_vehicles(dt)
        
        # 2. 处理EV充电需求
        ev_power, ev_rev, ev_penalty = self._serve_ev_queue(current_price)
        ev_power_demand += ev_power
        ev_revenue += ev_rev
        unmet_penalty += ev_penalty
        
        # 3. 处理FCEV加氢需求
        fcev_flow, fcev_rev, fcev_penalty = self._serve_fcev_queue(h2_available_kg, dt)
        fcev_h2_demand += fcev_flow
        fcev_revenue += fcev_rev
        unmet_penalty += fcev_penalty
        
        return ev_power_demand, fcev_h2_demand, ev_revenue, fcev_revenue, unmet_penalty
    
    def _update_serving_vehicles(self, dt):
        """更新正在服务的车辆状态"""
        # 更新EV
        for ev in self.ev_being_served[:]:
            if ev.service_start_time is not None:
                elapsed = dt
                if elapsed >= ev.estimated_charge_time:
                    # 充电完成
                    ev.is_completed = True
                    self.ev_being_served.remove(ev)
                    self.total_ev_served += 1
                    self.total_ev_revenue += ev.get_charging_revenue()
        
        # 更新FCEV
        for fcev in self.fcev_being_served[:]:
            if fcev.service_start_time is not None:
                elapsed = dt
                if elapsed >= fcev.fill_time_hours:
                    # 加氢完成
                    fcev.is_completed = True
                    self.fcev_being_served.remove(fcev)
                    self.total_fcev_served += 1
                    self.total_fcev_revenue += fcev.get_refueling_revenue()
    
    def _serve_ev_queue(self, current_price):
        """服务EV队列"""
        power_demand = 0.0
        revenue = 0.0
        penalty = 0.0
        
        # 计算可用充电桩
        available_fast = self.ev_fast_chargers - len([v for v in self.ev_being_served if v.charge_mode in ['fast', 'ultra_fast']])
        available_slow = self.ev_slow_chargers - len([v for v in self.ev_being_served if v.charge_mode == 'slow'])
        
        for ev in self.ev_queue[:]:
            # 检查需求响应
            if ev.check_delay_eligibility(current_price):
                self.total_vehicles_delayed += 1
                continue
            
            # 尝试分配充电桩
            if ev.charge_mode in ['fast', 'ultra_fast'] and available_fast > 0:
                self.ev_being_served.append(ev)
                self.ev_queue.remove(ev)
                ev.service_start_time = 0
                ev.is_being_served = True
                power_demand += ev.charge_power
                revenue += ev.get_charging_revenue()
                available_fast -= 1
            elif ev.charge_mode == 'slow' and available_slow > 0:
                self.ev_being_served.append(ev)
                self.ev_queue.remove(ev)
                ev.service_start_time = 0
                ev.is_being_served = True
                power_demand += ev.charge_power
                revenue += ev.get_charging_revenue()
                available_slow -= 1
            else:
                # 无法服务，产生等待惩罚
                penalty += Config.penalty_vehicle_waiting * Config.dt
        
        return power_demand, revenue, penalty
    
    def _serve_fcev_queue(self, h2_available, dt):
        """服务FCEV队列"""
        h2_flow_demand = 0.0
        revenue = 0.0
        penalty = 0.0
        
        # 计算可用加氢枪
        available_dispensers = self.fcev_dispensers - len(self.fcev_being_served)
        
        # 计算本时段可用氢气 (kg)
        h2_available_this_step = h2_available * dt  # 转换为本时段的量
        
        for fcev in self.fcev_queue[:]:
            # 检查氢气可用性
            if fcev.check_delay_eligibility(h2_available):
                self.total_vehicles_delayed += 1
                penalty += Config.penalty_vehicle_waiting * dt * 2  # FCEV等待惩罚更高
                continue
            
            # 检查是否有足够氢气和设备
            if fcev.h2_needed <= h2_available_this_step and available_dispensers > 0:
                self.fcev_being_served.append(fcev)
                self.fcev_queue.remove(fcev)
                fcev.service_start_time = 0
                fcev.is_being_served = True
                h2_flow_demand += fcev.h2_flow_rate
                revenue += fcev.get_refueling_revenue()
                h2_available_this_step -= fcev.h2_needed
                available_dispensers -= 1
            else:
                # 氢气不足或设备不足
                penalty += Config.penalty_unmet_h2_demand * fcev.h2_needed * 0.1
        
        return h2_flow_demand, revenue, penalty
    
    def get_statistics(self):
        """获取统计信息"""
        return {
            'ev_served': self.total_ev_served,
            'fcev_served': self.total_fcev_served,
            'ev_revenue': self.total_ev_revenue,
            'fcev_revenue': self.total_fcev_revenue,
            'vehicles_delayed': self.total_vehicles_delayed,
            'ev_queue_length': len(self.ev_queue),
            'fcev_queue_length': len(self.fcev_queue),
            'ev_being_served': len(self.ev_being_served),
            'fcev_being_served': len(self.fcev_being_served)
        }