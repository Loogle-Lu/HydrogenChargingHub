import numpy as np
from config import Config


class Electrolyzer:
    """线性电解槽: 输入功率 -> 产氢量"""

    def __init__(self):
        self.max_power = Config.ele_max_power
        self.efficiency = Config.ele_efficiency

    def compute(self, power_input):
        power = np.clip(power_input, 0, self.max_power)
        h2_flow_kg = power / self.efficiency  # kg/h
        return h2_flow_kg, power


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