import numpy as np
from config import Config


class Electrolyzer:
    """
    线性电解槽: 输入功率 -> 产氢量
    优先使用可再生能源，剩余部分使用电网
    """

    def __init__(self):
        self.max_power = Config.ele_max_power
        self.efficiency = Config.ele_efficiency

    def compute(self, power_input, re_available=0.0):
        """
        计算产氢量
        
        参数:
        - power_input: 目标输入功率 (kW)
        - re_available: 可用可再生能源功率 (kW)
        
        返回:
        - h2_flow_kg: 产氢量 (kg/h)
        - actual_power: 实际功率 (kW)
        - green_h2_ratio: 绿氢占比 (0-1)
        - power_from_re: 来自可再生能源的功率 (kW)
        - power_from_grid: 来自电网的功率 (kW)
        """
        power = np.clip(power_input, 0, self.max_power)
        h2_flow_kg = power / self.efficiency
        
        # 优先使用可再生能源
        power_from_re = min(power, re_available)
        power_from_grid = power - power_from_re
        green_h2_ratio = power_from_re / power if power > 0 else 0.0
        
        return h2_flow_kg, power, green_h2_ratio, power_from_re, power_from_grid
    
    def reset(self):
        """重置状态"""
        pass


class Compressor:
    """
    通用压缩机类 - 单级或两级压缩
    基于等熵压缩公式实现
    
    v3.1 新增功能:
    1. 变速驱动 (VSD): 效率随负载率变化
    2. 动态级间冷却: 根据电价调整冷却温度
    """
    def __init__(self, p_in, p_out, max_flow, efficiency=None, name="Compressor"):
        self.name = name
        self.p_in = p_in  # bar
        self.p_out = p_out  # bar
        self.max_flow = max_flow  # kg/h
        self.nominal_eta = efficiency or Config.comp_efficiency  # 额定效率
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
        
        # v3.1: 统计数据
        self.total_energy_consumption = 0.0  # kWh
        self.operating_hours = 0.0
        self.energy_saved_by_vsd = 0.0  # 变速驱动节能量
        self.energy_saved_by_cooling = 0.0  # 动态冷却节能量
    
    def _compute_vsd_efficiency(self, flow_ratio):
        """
        v3.1: 计算变速驱动下的实际效率
        使用线性插值从效率曲线获取
        """
        if not Config.enable_vsd:
            return self.nominal_eta
        
        # 从配置的效率曲线插值
        curve = Config.vsd_efficiency_curve
        load_points = sorted(curve.keys())
        
        if flow_ratio <= load_points[0]:
            return curve[load_points[0]]
        elif flow_ratio >= load_points[-1]:
            return curve[load_points[-1]]
        else:
            # 线性插值
            for i in range(len(load_points) - 1):
                if load_points[i] <= flow_ratio <= load_points[i+1]:
                    x0, x1 = load_points[i], load_points[i+1]
                    y0, y1 = curve[x0], curve[x1]
                    eta = y0 + (y1 - y0) * (flow_ratio - x0) / (x1 - x0)
                    return eta
        return self.nominal_eta
    
    def _compute_intercool_temp(self, electricity_price, cooling_intensity=None):
        """
        v3.1: 计算动态级间冷却温度
        电价高 -> 轻度冷却（节省冷却器电耗）
        电价低 -> 深度冷却（降低压缩机功耗）
        
        v3.2: 支持 RL Agent 控制 cooling_intensity [0,1]
        - cooling_intensity=0: 轻度冷却 (max_temp, 节省冷却器电耗)
        - cooling_intensity=1: 深度冷却 (min_temp, 降低压缩机功耗)
        - 当 cooling_intensity 为 None 时，使用原电价启发式
        """
        if not Config.enable_dynamic_cooling:
            return Config.T_in  # 默认回到进口温度
        
        if cooling_intensity is not None:
            # RL Agent 直接控制: 0=轻度, 1=深度
            T_cool = Config.max_intercool_temp - (Config.max_intercool_temp - Config.min_intercool_temp) * cooling_intensity
            return float(np.clip(T_cool, Config.min_intercool_temp, Config.max_intercool_temp))
        
        if electricity_price > Config.cooling_price_threshold:
            # 高电价: 轻度冷却
            return Config.max_intercool_temp
        else:
            # 低电价: 深度冷却
            # 线性插值
            price_ratio = electricity_price / Config.cooling_price_threshold
            T_cool = Config.min_intercool_temp + (Config.max_intercool_temp - Config.min_intercool_temp) * price_ratio
            return T_cool

    def compute_power(self, mass_flow_kg_h, electricity_price=0.08, cooling_intensity=None):
        """
        计算压缩功耗和产生的热量
        
        v3.1: 新增电价参数用于动态冷却优化
        v3.2: 新增 cooling_intensity 参数供 RL Agent 控制级间冷却强度
        """
        if mass_flow_kg_h <= 0:
            return 0.0, 0.0
        
        # 限制最大流量
        actual_flow = min(mass_flow_kg_h, self.max_flow)
        m_dot = actual_flow / 3600.0  # kg/s
        
        # v3.1: 计算负载率和实际效率
        flow_ratio = actual_flow / self.max_flow
        actual_eta = self._compute_vsd_efficiency(flow_ratio)
        
        # 统计VSD节能
        baseline_power_factor = self.nominal_eta / actual_eta if actual_eta > 0 else 1.0
        
        if self.use_two_stage:
            # 两级压缩
            # v3.1/v3.2: 动态级间冷却温度 (支持 Agent 控制的 cooling_intensity)
            T_intercool = self._compute_intercool_temp(electricity_price, cooling_intensity)
            
            # 第一级: p_in -> p_mid
            term1 = (self.p_mid / self.p_in) ** self.exponent - 1
            work_stage1 = self.cp * self.T_in * term1 / actual_eta  # 使用实际效率
            t_out_1 = self.T_in * (1 + term1)
            
            # 第二级: p_mid -> p_out (冷却到T_intercool)
            term2 = (self.p_out / self.p_mid) ** self.exponent - 1
            work_stage2 = self.cp * T_intercool * term2 / actual_eta  # 使用动态温度
            t_out_2 = T_intercool * (1 + term2)
            
            total_work_j_kg = work_stage1 + work_stage2
            
            # 热负荷 (两级的热量累加)
            heat_1 = m_dot * self.cp * (t_out_1 - T_intercool)  # 级间冷却
            heat_2 = m_dot * self.cp * (t_out_2 - Config.target_temp)  # 出口冷却
            total_heat_kw = (heat_1 + heat_2) / 1000.0
            
            # v3.1: 统计动态冷却节能
            # 冷却到更低温度 -> 第二级功耗降低
            baseline_work2 = self.cp * Config.T_in * term2 / self.nominal_eta
            actual_work2 = self.cp * T_intercool * term2 / actual_eta
            cooling_saving = (baseline_work2 - actual_work2) * m_dot / 1000.0  # kW
            self.energy_saved_by_cooling += max(0, cooling_saving) * Config.dt
            
        else:
            # 单级压缩
            term = (self.p_out / self.p_in) ** self.exponent - 1
            total_work_j_kg = self.cp * self.T_in * term / actual_eta
            t_out = self.T_in * (1 + term)
            
            heat = m_dot * self.cp * (t_out - Config.target_temp)
            total_heat_kw = heat / 1000.0
        
        power_kw = m_dot * total_work_j_kg / 1000.0
        
        # v3.1: 统计VSD节能
        baseline_power = power_kw * baseline_power_factor
        vsd_saving = baseline_power - power_kw
        self.energy_saved_by_vsd += max(0, vsd_saving) * Config.dt
        
        # 累积统计
        self.total_energy_consumption += power_kw * Config.dt
        self.operating_hours += Config.dt
        
        return power_kw, max(0, total_heat_kw)
    
    def get_statistics(self):
        """获取压缩机统计信息"""
        return {
            'name': self.name,
            'total_energy_kwh': self.total_energy_consumption,
            'operating_hours': self.operating_hours,
            'vsd_savings_kwh': self.energy_saved_by_vsd,
            'cooling_savings_kwh': self.energy_saved_by_cooling,
            'total_savings_kwh': self.energy_saved_by_vsd + self.energy_saved_by_cooling
        }
    
    def reset(self):
        """重置统计数据"""
        self.total_energy_consumption = 0.0
        self.operating_hours = 0.0
        self.energy_saved_by_vsd = 0.0
        self.energy_saved_by_cooling = 0.0


class MultiStageCompressorSystem:
    """
    多级级联压缩机系统 (根据HRS架构图)
    C1: 2 bar -> 35 bar
    C2: 35 bar -> 500 bar (级联充装)
    C3: 500 bar -> 700 bar (LDFV快充)
    
    v3.1 新增功能:
    1. 智能旁路控制: 储罐压力充足时跳过压缩
    2. 自适应压力控制: 根据FCEV SOG动态调整目标压力
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
        
        # v3.1: 旁路统计
        self.bypass_activations = {'c1': 0, 'c2': 0, 'c3': 0}
        self.energy_saved_by_bypass = 0.0  # kWh
    
    def _check_bypass(self, tank_pressure, target_pressure, demand, bypass_bias=None):
        """
        v3.1: 判断是否可以启用旁路
        v3.6: bypass_bias [0,1] 0=保守(阈值0.9) 1=积极(阈值0.7)
        """
        if not Config.enable_bypass:
            return False
        
        if bypass_bias is not None:
            # bypass_bias=0 保守(高阈值0.9), bypass_bias=1 积极(低阈值0.7)
            eff_threshold = 0.9 - 0.2 * np.clip(bypass_bias, 0, 1)
        else:
            eff_threshold = Config.bypass_pressure_threshold
        pressure_ok = tank_pressure >= target_pressure * eff_threshold
        demand_ok = demand <= Config.bypass_demand_threshold
        return pressure_ok and demand_ok
    
    def _compute_adaptive_target_pressure(self, avg_fcev_sog, c3_pressure_bias=None):
        """
        v3.1: 根据FCEV平均SOG计算自适应目标压力
        v3.6: c3_pressure_bias [0,1] 0.5=默认, 0=降压省功耗, 1=升压快充
        """
        if not Config.enable_adaptive_pressure:
            base = Config.c3_output_pressure
        else:
            map_points = sorted(Config.adaptive_pressure_map.keys())
            for i in range(len(map_points)):
                if avg_fcev_sog <= map_points[i]:
                    base = Config.adaptive_pressure_map[map_points[i]]
                    break
            else:
                base = Config.adaptive_pressure_map[map_points[-1]]
        
        if c3_pressure_bias is not None:
            # 0.5→1.0倍, 0→0.6倍, 1→1.4倍
            scale = 0.6 + 0.8 * np.clip(c3_pressure_bias, 0, 1)
            return min(base * scale, Config.c3_output_pressure)
        return base
    
    def compute_c1(self, mass_flow, tank_pressure=0, electricity_price=0.08, cooling_intensity=None, bypass_bias=None):
        """
        C1压缩: T1(2bar) -> T2(35bar)
        v3.6: 支持 bypass_bias
        """
        if Config.enable_bypass and self._check_bypass(tank_pressure, Config.c1_output_pressure, mass_flow, bypass_bias):
            # 旁路激活: T2压力足够，直接供气
            self.bypass_activations['c1'] += 1
            
            # 估算节省的能量（正常压缩功耗）
            normal_power, _ = self.c1.compute_power(mass_flow, electricity_price, cooling_intensity)
            self.energy_saved_by_bypass += normal_power * Config.dt
            
            return 0.0, 0.0  # 旁路：功耗和热量均为0
        
        return self.c1.compute_power(mass_flow, electricity_price, cooling_intensity)
    
    def compute_c2(self, mass_flow, tank_pressure=0, electricity_price=0.08, cooling_intensity=None, bypass_bias=None):
        """
        C2压缩: T2(35bar) -> T3系统(500bar)
        v3.6: 支持 bypass_bias
        """
        if Config.enable_bypass and self._check_bypass(tank_pressure, Config.c2_output_pressure, mass_flow, bypass_bias):
            # 旁路激活
            self.bypass_activations['c2'] += 1
            
            normal_power, _ = self.c2.compute_power(mass_flow, electricity_price, cooling_intensity)
            self.energy_saved_by_bypass += normal_power * Config.dt
            
            return 0.0, 0.0
        
        return self.c2.compute_power(mass_flow, electricity_price, cooling_intensity)
    
    def compute_c3(self, mass_flow, avg_fcev_sog=0.5, tank_pressure=0, electricity_price=0.08,
                   bypass_bias=None, c3_pressure_bias=None):
        """
        C3压缩: T3/T4 -> D2 LDFV(700bar)
        v3.6: 支持 bypass_bias, c3_pressure_bias
        """
        target_pressure = self._compute_adaptive_target_pressure(avg_fcev_sog, c3_pressure_bias)
        if Config.enable_bypass and self._check_bypass(tank_pressure, target_pressure, mass_flow, bypass_bias):
            self.bypass_activations['c3'] += 1
            
            # 使用原有目标压力估算节能（保守估计）
            self.c3.p_out = Config.c3_output_pressure
            normal_power, _ = self.c3.compute_power(mass_flow, electricity_price)
            self.energy_saved_by_bypass += normal_power * Config.dt
            
            return 0.0, 0.0
        
        # 动态调整C3目标压力
        original_p_out = self.c3.p_out
        self.c3.p_out = target_pressure
        self.c3.pressure_ratio = target_pressure / self.c3.p_in
        
        # 计算功耗和热量
        power, heat = self.c3.compute_power(mass_flow, electricity_price)
        
        # 恢复原始目标压力（避免影响其他计算）
        self.c3.p_out = original_p_out
        self.c3.pressure_ratio = original_p_out / self.c3.p_in
        
        return power, heat
    
    def compute_total_power(self, flow_c1, flow_c2, flow_c3, 
                           tank_pressures=None, avg_fcev_sog=0.5, electricity_price=0.08):
        """
        计算所有压缩机的总功耗和热负荷
        
        v3.1: 支持旁路和自适应压力
        """
        if tank_pressures is None:
            tank_pressures = {'t2': 0, 't3': 0, 't4': 0}
        
        p1, h1 = self.compute_c1(flow_c1, tank_pressures.get('t2', 0), electricity_price)
        p2, h2 = self.compute_c2(flow_c2, tank_pressures.get('t3', 0), electricity_price)
        p3, h3 = self.compute_c3(flow_c3, avg_fcev_sog, tank_pressures.get('t4', 0), electricity_price)
        
        return (p1 + p2 + p3), (h1 + h2 + h3)
    
    def get_statistics(self):
        """获取压缩机系统统计信息"""
        c1_stats = self.c1.get_statistics()
        c2_stats = self.c2.get_statistics()
        c3_stats = self.c3.get_statistics()
        
        return {
            'c1': c1_stats,
            'c2': c2_stats,
            'c3': c3_stats,
            'bypass_activations': self.bypass_activations.copy(),
            'bypass_savings_kwh': self.energy_saved_by_bypass,
            'total_vsd_savings_kwh': (c1_stats['vsd_savings_kwh'] + 
                                      c2_stats['vsd_savings_kwh'] + 
                                      c3_stats['vsd_savings_kwh']),
            'total_cooling_savings_kwh': (c1_stats['cooling_savings_kwh'] + 
                                          c2_stats['cooling_savings_kwh'] + 
                                          c3_stats['cooling_savings_kwh'])
        }
    
    def reset(self):
        """重置统计数据"""
        self.c1.reset()
        self.c2.reset()
        self.c3.reset()
        self.bypass_activations = {'c1': 0, 'c2': 0, 'c3': 0}
        self.energy_saved_by_bypass = 0.0


class LinearChiller:
    """
    线性冷却机模型
    使用固定COP计算冷却功耗: Power = HeatLoad / COP
    """
    def __init__(self):
        self.rated_capacity = Config.chiller_rated_capacity  # kW
        self.cop = Config.chiller_rated_cop
    
    def compute_power(self, heat_load_kw, chiller_ratio=None):
        """
        计算冷却所需功率 (线性)
        v3.6: chiller_ratio [0,1] 0=最小制冷(30%), 1=全制冷
        """
        if heat_load_kw <= 0:
            return 0.0
        actual_load = min(heat_load_kw, self.rated_capacity)
        power_kw = actual_load / self.cop
        if chiller_ratio is not None:
            power_kw *= (0.3 + 0.7 * np.clip(chiller_ratio, 0, 1))
        return power_kw
    
    def reset(self):
        """重置冷却机状态"""
        pass


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


class BatteryEnergyStorage:
    """
    电池储能系统 (Battery Energy Storage System, BESS)
    
    功能:
    - 电化学储能，快速充放电
    - 电网功率平滑和峰值管理
    - 与氢储能协同工作
    - 考虑充放电效率、SOC约束、功率限制
    
    优势相比氢储能:
    - 响应速度快 (毫秒级 vs 分钟级)
    - 往返效率高 (90-95% vs 30-40%)
    - 适合短时调峰
    
    协同策略:
    - 电池: 短时调峰、功率平滑 (秒-分钟)
    - 氢气: 长时储能、季节调节 (小时-天)
    """
    
    def __init__(self):
        self.capacity = Config.battery_capacity  # kWh
        self.max_charge_power = Config.battery_max_charge_power  # kW
        self.max_discharge_power = Config.battery_max_discharge_power  # kW
        self.charge_efficiency = Config.battery_charge_efficiency  # 充电效率
        self.discharge_efficiency = Config.battery_discharge_efficiency  # 放电效率
        self.min_soc = Config.battery_min_soc
        self.max_soc = Config.battery_max_soc
        self.initial_soc = Config.battery_initial_soc
        
        # 当前状态
        self.current_energy = self.initial_soc * self.capacity  # kWh
        self.total_charge_energy = 0.0  # 累计充电量 (kWh)
        self.total_discharge_energy = 0.0  # 累计放电量 (kWh)
        self.charge_cycles = 0.0  # 充放电循环次数
        
    def get_soc(self):
        """获取当前SOC"""
        return self.current_energy / self.capacity
    
    def get_available_charge_power(self):
        """获取可用充电功率 (kW)"""
        soc = self.get_soc()
        if soc >= self.max_soc:
            return 0.0
        # 可充入的能量 (kWh)
        available_energy = (self.max_soc - soc) * self.capacity
        # 考虑充电效率，可接受的输入功率
        available_power = available_energy / Config.dt / self.charge_efficiency
        return min(available_power, self.max_charge_power)
    
    def get_available_discharge_power(self):
        """获取可用放电功率 (kW)"""
        soc = self.get_soc()
        if soc <= self.min_soc:
            return 0.0
        # 可放出的能量 (kWh)
        available_energy = (soc - self.min_soc) * self.capacity
        # 考虑放电效率，可输出的功率
        available_power = available_energy / Config.dt * self.discharge_efficiency
        return min(available_power, self.max_discharge_power)
    
    def charge(self, power_input, dt=None):
        """
        充电
        
        参数:
            power_input: 输入功率 (kW)
            dt: 时间步长 (h)
        
        返回:
            actual_power: 实际充电功率 (kW)
            energy_stored: 实际存储的能量 (kWh)
        """
        if dt is None:
            dt = Config.dt
        
        # 限制充电功率
        available_power = self.get_available_charge_power()
        actual_power = min(power_input, available_power)
        
        if actual_power <= 0:
            return 0.0, 0.0
        
        # 考虑充电效率
        energy_input = actual_power * dt
        energy_stored = energy_input * self.charge_efficiency
        
        # 更新SOC
        self.current_energy = min(
            self.current_energy + energy_stored,
            self.max_soc * self.capacity
        )
        
        # 统计
        self.total_charge_energy += energy_input
        
        return actual_power, energy_stored
    
    def discharge(self, power_demand, dt=None):
        """
        放电
        
        参数:
            power_demand: 需求功率 (kW)
            dt: 时间步长 (h)
        
        返回:
            actual_power: 实际放电功率 (kW)
            energy_consumed: 实际消耗的储存能量 (kWh)
        """
        if dt is None:
            dt = Config.dt
        
        # 限制放电功率
        available_power = self.get_available_discharge_power()
        actual_power = min(power_demand, available_power)
        
        if actual_power <= 0:
            return 0.0, 0.0
        
        # 考虑放电效率
        energy_output = actual_power * dt
        energy_consumed = energy_output / self.discharge_efficiency
        
        # 更新SOC
        self.current_energy = max(
            self.current_energy - energy_consumed,
            self.min_soc * self.capacity
        )
        
        # 统计
        self.total_discharge_energy += energy_output
        
        # 更新循环次数 (简化：每放电一次电池容量算0.5个循环)
        self.charge_cycles += energy_consumed / self.capacity / 2
        
        return actual_power, energy_consumed
    
    def get_statistics(self):
        """获取电池统计信息"""
        total_throughput = self.total_charge_energy + self.total_discharge_energy
        roundtrip_efficiency = (self.total_discharge_energy / self.total_charge_energy * 100) if self.total_charge_energy > 0 else 0.0
        
        return {
            'current_soc': self.get_soc(),
            'current_energy_kwh': self.current_energy,
            'total_charge_kwh': self.total_charge_energy,
            'total_discharge_kwh': self.total_discharge_energy,
            'total_throughput_kwh': total_throughput,
            'roundtrip_efficiency_pct': roundtrip_efficiency,
            'charge_cycles': self.charge_cycles,
            'degradation_pct': min(self.charge_cycles / Config.battery_lifetime_cycles * 100, 100)
        }
    
    def reset(self):
        """重置电池状态"""
        self.current_energy = self.initial_soc * self.capacity
        self.total_charge_energy = 0.0
        self.total_discharge_energy = 0.0
        self.charge_cycles = 0.0


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
    基于真实用户行为模式的高级需求模拟
    """
    def __init__(self):
        self.vehicle_id_counter = 0
        self.ev_fcev_ratio = Config.ev_fcev_ratio
        self.current_day = 0  # 追踪天数用于周模式
        
    def _get_time_multiplier(self, hour_of_day, day_of_week=None):
        """
        根据时段和星期获取到达率倍数
        考虑真实用户行为模式
        """
        # 基础时段系数
        if hour_of_day in Config.peak_morning_hours:
            # 早高峰 (7-9AM): 通勤充电
            base_multiplier = Config.peak_arrival_multiplier * 1.2
        elif hour_of_day in Config.peak_evening_hours:
            # 晚高峰 (5-7PM): 下班充电
            base_multiplier = Config.peak_arrival_multiplier * 1.0
        elif hour_of_day in Config.midday_hours:
            # 午间 (12-2PM): 中等需求
            base_multiplier = Config.midday_multiplier
        elif hour_of_day in Config.offpeak_hours:
            # 深夜/凌晨 (11PM-6AM): 低需求
            base_multiplier = Config.offpeak_multiplier
        else:
            # 其他时段
            base_multiplier = 1.0
        
        # 星期模式修正
        if day_of_week is not None:
            if day_of_week >= 5:  # 周末 (5=周六, 6=周日)
                # 周末需求分布更平均，峰值降低
                if hour_of_day in Config.peak_morning_hours:
                    base_multiplier *= 0.6  # 周末早高峰需求降低
                elif 10 <= hour_of_day <= 20:  # 周末白天需求分散
                    base_multiplier *= 1.3
            else:  # 工作日 (0-4)
                if hour_of_day in Config.peak_morning_hours or hour_of_day in Config.peak_evening_hours:
                    base_multiplier *= 1.1  # 工作日峰值更明显
        
        # 添加小幅随机波动 (±15%)
        random_factor = np.random.uniform(0.85, 1.15)
        
        return base_multiplier * random_factor
    
    def generate_vehicles(self, current_step):
        """
        生成当前时段到达的车辆
        基于真实用户行为模式，考虑时段、星期、价格等因素
        
        返回:
        - ev_list: EV列表
        - fcev_list: FCEV列表
        """
        # 计算当前时刻和星期
        hour_of_day = int((current_step * Config.dt) % 24)
        day_of_year = int(current_step / 96)  # 第几天
        day_of_week = day_of_year % 7  # 星期几 (0=周一, 6=周日)
        
        # 调整到达率
        time_multiplier = self._get_time_multiplier(hour_of_day, day_of_week)
        adjusted_arrival_rate = Config.base_vehicle_arrival_rate * time_multiplier
        
        # 泊松分布生成到达车辆数
        # 添加最小和最大限制，避免极端情况
        mean_arrivals = adjusted_arrival_rate * Config.dt
        num_arrivals = np.random.poisson(mean_arrivals)
        num_arrivals = min(num_arrivals, 15)  # 单个时段最多15辆车
        
        ev_list = []
        fcev_list = []
        
        for _ in range(num_arrivals):
            self.vehicle_id_counter += 1
            
            # 决定车辆类型
            # EV/FCEV比例随时段微调
            ev_ratio = self.ev_fcev_ratio
            if hour_of_day in Config.peak_morning_hours:
                # 早高峰EV占比稍高 (通勤)
                ev_ratio = min(0.80, self.ev_fcev_ratio + 0.10)
            elif hour_of_day in Config.offpeak_hours:
                # 深夜FCEV占比稍高 (商用车)
                ev_ratio = max(0.60, self.ev_fcev_ratio - 0.10)
            
            if np.random.random() < ev_ratio:
                # 生成EV
                ev = self._generate_ev(self.vehicle_id_counter, current_step, hour_of_day)
                ev_list.append(ev)
            else:
                # 生成FCEV
                fcev = self._generate_fcev(self.vehicle_id_counter, current_step, hour_of_day)
                fcev_list.append(fcev)
        
        return ev_list, fcev_list
    
    def _generate_ev(self, vehicle_id, arrival_time, hour_of_day):
        """
        生成单个EV
        基于真实用户行为模式，考虑时段对SOC和充电模式的影响
        """
        # 电池容量 (考虑车型分布)
        # 小型车(40-50kWh): 30%, 中型车(60-70kWh): 50%, 大型车(80-100kWh): 20%
        car_type_rand = np.random.random()
        if car_type_rand < 0.30:
            # 小型车
            battery_capacity = np.random.uniform(40.0, 50.0)
        elif car_type_rand < 0.80:
            # 中型车
            battery_capacity = np.random.normal(Config.ev_battery_capacity_mean, 5.0)
        else:
            # 大型车/SUV
            battery_capacity = np.random.uniform(80.0, 100.0)
        
        battery_capacity = np.clip(battery_capacity, 40.0, 100.0)
        
        # 到站SOC (根据时段调整分布)
        if hour_of_day in Config.peak_morning_hours:
            # 早高峰: 夜间充电后，SOC较高
            soc_mean = 0.45
            soc_std = 0.15
        elif hour_of_day in Config.peak_evening_hours:
            # 晚高峰: 一天使用后，SOC较低
            soc_mean = 0.20
            soc_std = 0.10
        elif hour_of_day in Config.offpeak_hours:
            # 深夜: 紧急充电，SOC很低
            soc_mean = 0.15
            soc_std = 0.08
        else:
            # 其他时段
            soc_mean = Config.ev_soc_arrival_mean
            soc_std = Config.ev_soc_arrival_std
        
        soc_initial = np.clip(
            np.random.normal(soc_mean, soc_std),
            0.05, 0.80
        )
        
        # 充电模式选择 (根据时段和SOC调整)
        mode_rand = np.random.random()
        
        # 时段影响充电模式选择
        fast_ratio = Config.ev_fast_charge_ratio
        ultra_ratio = Config.ev_ultra_fast_ratio
        
        if hour_of_day in Config.peak_morning_hours or hour_of_day in Config.peak_evening_hours:
            # 高峰期更多人选择快充
            fast_ratio += 0.10
            ultra_ratio += 0.03
        elif hour_of_day in Config.offpeak_hours:
            # 深夜更多人选择慢充
            fast_ratio -= 0.15
        
        # SOC很低时倾向快充
        if soc_initial < 0.15:
            fast_ratio += 0.15
            ultra_ratio += 0.05
        
        # 归一化概率
        total = fast_ratio + ultra_ratio + Config.ev_slow_charge_ratio
        fast_ratio /= total
        ultra_ratio /= total
        
        if mode_rand < fast_ratio:
            charge_mode = 'fast'
            soc_target = Config.ev_soc_target_fast
            dr_flex = Config.ev_dr_flexibility_fast
        elif mode_rand < fast_ratio + ultra_ratio:
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
    
    def _generate_fcev(self, vehicle_id, arrival_time, hour_of_day):
        """
        生成单个FCEV
        基于真实FCEV使用场景 (商用车为主)
        """
        # 储氢罐容量 (考虑车型分布)
        # Hyundai Nexo: 6.3kg (70%)
        # Toyota Mirai: 5.6kg (20%)
        # 商用大车: 8-10kg (10%)
        car_type_rand = np.random.random()
        if car_type_rand < 0.70:
            tank_capacity = Config.fcev_tank_capacity  # 6.3kg (Nexo)
        elif car_type_rand < 0.90:
            tank_capacity = 5.6  # Toyota Mirai
        else:
            tank_capacity = np.random.uniform(8.0, 10.0)  # 商用车
        
        # 到站SOG (State of Gas) - 根据时段和车型调整
        if hour_of_day in Config.peak_morning_hours:
            # 早高峰: 商用车开始一天工作，SOG中等
            sog_mean = 0.35
            sog_std = 0.12
        elif hour_of_day in Config.peak_evening_hours:
            # 晚高峰: 商用车结束工作，SOG较低
            sog_mean = 0.18
            sog_std = 0.08
        elif 10 <= hour_of_day <= 16:
            # 白天: 中途补充，SOG中等偏低
            sog_mean = 0.25
            sog_std = 0.10
        else:
            # 其他时段
            sog_mean = Config.fcev_sog_arrival_mean
            sog_std = Config.fcev_sog_arrival_std
        
        # 商用车SOG普遍更低 (使用强度大)
        if tank_capacity > 7.0:  # 商用车
            sog_mean *= 0.8
        
        sog_initial = np.clip(
            np.random.normal(sog_mean, sog_std),
            0.05, 0.50
        )
        
        # 目标SOG (商用车倾向充更满)
        if tank_capacity > 7.0:
            sog_target = 0.98  # 商用车充到98%
        else:
            sog_target = Config.fcev_sog_target  # 乘用车95%
        
        return FCEVehicle(
            vehicle_id, arrival_time, tank_capacity,
            sog_initial, sog_target
        )


class IntegratedServiceStation:
    """
    集成服务站 (EV充电 + FCEV加氢)
    管理车辆队列、服务调度、需求响应
    
    v3.1: 新增FCEV平均SOG追踪，用于自适应压力控制
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
        
        # v3.1: FCEV SOG追踪
        self.current_fcev_avg_sog = 0.5  # 当前服务的FCEV平均SOG
    
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
        
        # v3.1: 更新当前服务FCEV的平均SOG
        if len(self.fcev_being_served) > 0:
            total_sog = sum(fcev.sog_initial for fcev in self.fcev_being_served)
            self.current_fcev_avg_sog = total_sog / len(self.fcev_being_served)
        else:
            # 无FCEV在服务，使用队列平均值或默认值
            if len(self.fcev_queue) > 0:
                total_sog = sum(fcev.sog_initial for fcev in self.fcev_queue[:3])  # 前3辆
                self.current_fcev_avg_sog = total_sog / min(3, len(self.fcev_queue))
            else:
                self.current_fcev_avg_sog = 0.5  # 默认50%
        
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
