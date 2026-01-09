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


class CascadeCompressor:
    """
    两级压缩机 (Two-stage Compressor)
    基于论文 Eq. (10) 实现
    """

    def __init__(self):
        self.eta = Config.comp_efficiency
        self.gamma = Config.H2_gamma
        self.R = Config.H2_R
        self.T = Config.T_in

        # 压力参数
        self.P1 = Config.comp_input_pressure
        self.P3 = Config.comp_output_pressure
        # 最佳中间压力 P2 = sqrt(P1 * P3)
        self.P2 = np.sqrt(self.P1 * self.P3)

        # 预计算常数部分
        self.exponent = (self.gamma - 1) / self.gamma
        self.cp = self.gamma * self.R / (self.gamma - 1)

    def compute_power(self, mass_flow_kg_h):
        """
        计算压缩功耗和产生的热量
        """
        if mass_flow_kg_h <= 0: return 0.0, 0.0

        m_dot = mass_flow_kg_h / 3600.0  # kg/s

        # 第一级压缩 P1 -> P2
        term1 = (self.P2 / self.P1) ** self.exponent - 1
        work_stage1 = self.cp * self.T * term1 / self.eta

        # 第二级压缩 P2 -> P3
        term2 = (self.P3 / self.P2) ** self.exponent - 1
        work_stage2 = self.cp * self.T * term2 / self.eta

        # 总功耗 (kW)
        total_work_j_kg = work_stage1 + work_stage2
        power_kw = m_dot * total_work_j_kg / 1000.0

        # 热负荷计算 (用于 Chiller)
        # 计算出口温度带来的显热增加
        t_out_1 = self.T * (1 + term1 / self.eta)
        t_out_2 = self.T * (1 + term2 / self.eta)

        heat_1 = m_dot * self.cp * (t_out_1 - Config.target_temp)
        heat_2 = m_dot * self.cp * (t_out_2 - Config.target_temp)

        total_heat_kw = (heat_1 + heat_2) / 1000.0

        return power_kw, max(0, total_heat_kw)


class Chiller:
    """
    冷却机: 根据 COP 计算移除热量所需的电能
    """

    def __init__(self):
        self.cop = Config.chiller_cop

    def compute_power(self, heat_load_kw):
        if heat_load_kw <= 0: return 0.0
        return heat_load_kw / self.cop


class CascadeStorage:
    def __init__(self):
        self.capacity = Config.storage_capacity_kg
        self.level = Config.storage_initial * self.capacity
        self.min_lvl = Config.storage_min_level * self.capacity
        self.max_lvl = Config.storage_max_level * self.capacity

    def step(self, inflow, outflow):
        delta = (inflow - outflow) * Config.dt
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


class FuelCell:
    def __init__(self):
        self.max_power = Config.fc_max_power
        self.efficiency = Config.fc_efficiency

    def compute(self, h2_input_kg):
        power_generated = h2_input_kg * self.efficiency
        actual_power = min(power_generated, self.max_power)
        consumed_h2 = actual_power / self.efficiency
        return actual_power, consumed_h2