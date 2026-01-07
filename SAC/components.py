import numpy as np
from config import Config


class Electrolyzer:
    def __init__(self):
        self.max_power = Config.ele_max_power
        self.efficiency = Config.ele_efficiency

    def compute(self, power_input):
        power = np.clip(power_input, 0, self.max_power)
        # kWh/kg -> kg/h:  Power(kW) / Efficiency(kWh/kg) = kg/h
        h2_flow_kg = power / self.efficiency
        return h2_flow_kg, power


class CascadeCompressor:
    def __init__(self):
        self.eta = Config.comp_efficiency
        self.cp = Config.H2_gamma / (Config.H2_gamma - 1) * Config.H2_R

    def compute_power(self, mass_flow):
        if mass_flow <= 0: return 0.0, 0.0
        p_ratio = Config.comp_output_pressure / Config.comp_input_pressure
        exponent = (Config.H2_gamma - 1) / Config.H2_gamma
        work_per_kg = self.cp * Config.T_ambient * (pow(p_ratio, exponent) - 1)
        # mass_flow is kg/h, convert to kg/s for Watt calculation, then to kW
        power_kw = (work_per_kg * mass_flow / 3600.0) / 1000.0 / self.eta

        t_out = Config.T_ambient * (1 + (pow(p_ratio, exponent) - 1) / self.eta)
        heat_load_j_kg = Config.gas_heat_capacity * 1000 * (t_out - Config.target_temp)
        total_heat_load_kw = (heat_load_j_kg * mass_flow / 3600.0) / 1000.0
        return power_kw, max(0, total_heat_load_kw)


class CascadeStorage:
    def __init__(self):
        self.capacity = Config.storage_capacity_kg
        self.level = Config.storage_initial * self.capacity
        self.min_lvl = Config.storage_min_level * self.capacity
        self.max_lvl = Config.storage_max_level * self.capacity

    def step(self, inflow, outflow):
        # inflow/outflow: kg/h -> delta: kg
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
        # h2_input_kg is flow rate in kg/h
        # Efficiency is kWh/kg. Power = Flow(kg/h) * Eff(kWh/kg) = kW
        power_generated = h2_input_kg * self.efficiency
        actual_power = min(power_generated, self.max_power)

        # Recalculate consumed H2 based on actual power limited
        consumed_h2 = actual_power / self.efficiency
        return actual_power, consumed_h2


class Chiller:
    def __init__(self):
        self.cop = Config.chiller_cop

    def compute_power(self, heat_load_kw):
        return heat_load_kw / self.cop