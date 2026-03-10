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
            return Config.no_cooling_intercool_temp  # 无动态冷却 → 最高温度(最差工况)
        
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
    差压级联压缩机系统 (本文创新架构)
    C1: 2 bar  → 35 bar  (T1→T2)
    C2: 35 bar → 500 bar (T2→T3₁/T3₂/T3₃ 差压充装)
    C3: ≤500 bar → 700 bar (T3 在线取气直充 FCEV, 无 T4 缓冲)
    
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
            # t3_3: T3₃(500 bar) — C3 在线取气入口（差压级联，无 T4）
            tank_pressures = {'t2': 0, 't3': 0, 't3_3': 0}

        p1, h1 = self.compute_c1(flow_c1, tank_pressures.get('t2', 0), electricity_price)
        p2, h2 = self.compute_c2(flow_c2, tank_pressures.get('t3', 0), electricity_price)
        p3, h3 = self.compute_c3(flow_c3, avg_fcev_sog, tank_pressures.get('t3_3', 0), electricity_price)
        
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
    差压级联储罐系统 (Differential-Pressure Cascade, 本文创新架构)

    拓扑:
      T1 (2 bar)  → C1 → T2 (35 bar) → C2 → T3₁/T3₂/T3₃ (200/350/500 bar)
                                              ↓ Low→High 差压级联直充
                                         HDV: T3₁→T3₂ 直充 (350 bar)
                                         LDV: T3₁→T3₂→T3₃ 直充 + T3₃→C3→700 bar (增压末段)

    核心创新: T3 三罐采用差压分级 (200 / 350 / 500 bar), 统一 Low→High 级联协议
    - C2 充装时优先填充压力最低的储罐 (最大 deficit 权重)
    - 两种车型均遵循 Low→High 级联: 从最低压储罐开始
    - **动态压力门控**: SOG 边界 = (max_p × SOC - min_dp) / target_p, 随储罐消耗实时变化
      储罐实际压强不足时自动跳过, 无需人工设定固定切换点
    - 350-bar HDV: T3₁→T3₂→T3₃ 级联直充 (T3₃ 在 T3₂ 压力不足时补充顶充)
    - 700-bar LDV: T3₁→T3₂→T3₃ 级联直充 + T3₃→C3 增压末段
    - C3 角色: 末段增压泵 (仅处理级联直充覆盖不到的部分), 非全量压缩
    - 去掉 T4 超高压缓冲罐, C3 在线压缩直接输出至 FCEV 加氢机
    """
    def __init__(self):
        self.t1 = HydrogenTank(
            Config.t1_capacity_kg, Config.t1_max_pressure,
            Config.t1_initial_soc, name="T1"
        )
        self.t2 = HydrogenTank(
            Config.t2_capacity_kg, Config.t2_max_pressure,
            Config.t2_initial_soc, name="T2"
        )
        # 差压级联储罐组: T3₁ < T3₂ < T3₃
        self.t3_1 = HydrogenTank(
            Config.t3_1_capacity_kg, Config.t3_1_max_pressure,
            Config.t3_initial_soc, name="T3_1(200bar)"
        )
        self.t3_2 = HydrogenTank(
            Config.t3_2_capacity_kg, Config.t3_2_max_pressure,
            Config.t3_initial_soc, name="T3_2(350bar)"
        )
        self.t3_3 = HydrogenTank(
            Config.t3_3_capacity_kg, Config.t3_3_max_pressure,
            Config.t3_initial_soc, name="T3_3(500bar)"
        )
        # T4 已移除: C3 在线直充取代超高压缓冲罐

    def _cascade_fill(self, h2_from_c2):
        """
        差压级联充装策略: C2 输出优先填充压力最低的储罐。
        返回分配给各罐的流量 (kg/h)。
        """
        soc1 = self.t3_1.get_soc()
        soc2 = self.t3_2.get_soc()
        soc3 = self.t3_3.get_soc()

        # 容量权重: SOC 越低，权重越大（优先补充压差最大的罐）
        deficit1 = max(0.0, Config.storage_max_level - soc1)
        deficit2 = max(0.0, Config.storage_max_level - soc2)
        deficit3 = max(0.0, Config.storage_max_level - soc3)
        total_deficit = deficit1 + deficit2 + deficit3

        if total_deficit < 1e-6:
            # 三罐均满，均匀分配（excess 会被罐体截断）
            return h2_from_c2 / 3.0, h2_from_c2 / 3.0, h2_from_c2 / 3.0

        f1 = h2_from_c2 * deficit1 / total_deficit
        f2 = h2_from_c2 * deficit2 / total_deficit
        f3 = h2_from_c2 * deficit3 / total_deficit
        return f1, f2, f3

    def _compute_dynamic_sog_boundaries(self, target_pressure):
        """
        根据各储罐当前 **实际压强** 计算 SOG 边界 (Low→High 差压级联).

        实际压强 = max_pressure × SOC，随气体消耗而降低.
        储罐只有在 (实际压强 - 最低压差) > 车辆背压 时才能送气.

        返回 (sog_b1, sog_b2, sog_b3)：各罐能服务到的车辆 SOG 上限.
        """
        min_dp = Config.min_cascade_pressure_diff
        p1_actual = Config.t3_1_max_pressure * self.t3_1.get_soc()
        p2_actual = Config.t3_2_max_pressure * self.t3_2.get_soc()
        p3_actual = Config.t3_3_max_pressure * self.t3_3.get_soc()

        sog_b1 = max(0.0, (p1_actual - min_dp) / target_pressure)
        sog_b2 = max(0.0, (p2_actual - min_dp) / target_pressure)
        sog_b3 = max(0.0, (p3_actual - min_dp) / target_pressure)
        return sog_b1, sog_b2, sog_b3

    def _cascade_discharge_350(self, h2_demand_350, avg_sog_350=0.2):
        """
        350-bar HDV 差压级联充装 (Low→High, SAE J2601):
          Phase 1: T3₁ → 车辆  (实际压强 > 车辆背压 + min_dp)
          Phase 2: T3₂ → 车辆
          Phase 3: T3₃ → 车辆  (T3₂ 压力不足时由 T3₃ 补充顶充)
        SOG 边界为 **动态值**, 随储罐 SOC 变化.
        返回 (take_t3_1, take_t3_2, take_t3_3) kg/h.
        """
        target_p = 350.0
        sog_target = Config.fcev_sog_target
        sog_start = avg_sog_350
        sog_range = sog_target - sog_start
        if sog_range <= 1e-6:
            return 0.0, 0.0, 0.0

        sog_b1, sog_b2, sog_b3 = self._compute_dynamic_sog_boundaries(target_p)

        frac_t31 = max(0.0, min(sog_b1, sog_target) - sog_start) / sog_range
        frac_t32 = max(0.0, min(sog_b2, sog_target) - max(sog_start, sog_b1)) / sog_range
        frac_t33 = max(0.0, min(sog_b3, sog_target) - max(sog_start, sog_b2)) / sog_range

        avail1 = max(0.0, (self.t3_1.level - self.t3_1.min_lvl) / Config.dt)
        avail2 = max(0.0, (self.t3_2.level - self.t3_2.min_lvl) / Config.dt)
        avail3 = max(0.0, (self.t3_3.level - self.t3_3.min_lvl) / Config.dt)

        take1 = min(h2_demand_350 * frac_t31, avail1)
        take2 = min(h2_demand_350 * frac_t32, avail2)
        take3 = min(h2_demand_350 * frac_t33, avail3)

        return take1, take2, take3

    def _cascade_discharge_700(self, h2_demand_700, avg_sog_700=0.2):
        """
        700-bar LDV 差压级联充装 + C3 增压 (Low→High, SAE J2601):
          Phase 1-3: T3₁→T3₂→T3₃ 直充 (动态压力门控)
          Phase 4:   T3₃ → C3 → 700 bar (增压泵, 级联直充覆盖不到的部分)
        SOG 边界为 **动态值**, 随储罐 SOC 变化.
        返回 (take_t3_1, take_t3_2, take_t3_3_direct, take_for_c3) kg/h.
        """
        target_p = 700.0
        sog_target = Config.fcev_sog_target
        sog_start = avg_sog_700
        sog_range = sog_target - sog_start
        if sog_range <= 1e-6:
            return 0.0, 0.0, 0.0, 0.0

        sog_b1, sog_b2, sog_b3 = self._compute_dynamic_sog_boundaries(target_p)

        frac_t31 = max(0.0, min(sog_b1, sog_target) - sog_start) / sog_range
        frac_t32 = max(0.0, min(sog_b2, sog_target) - max(sog_start, sog_b1)) / sog_range
        frac_t33 = max(0.0, min(sog_b3, sog_target) - max(sog_start, sog_b2)) / sog_range
        frac_c3  = max(0.0, sog_target - max(sog_start, sog_b3)) / sog_range

        avail1 = max(0.0, (self.t3_1.level - self.t3_1.min_lvl) / Config.dt)
        avail2 = max(0.0, (self.t3_2.level - self.t3_2.min_lvl) / Config.dt)
        avail3 = max(0.0, (self.t3_3.level - self.t3_3.min_lvl) / Config.dt)

        take1 = min(h2_demand_700 * frac_t31, avail1)
        take2 = min(h2_demand_700 * frac_t32, avail2)
        take3_direct = min(h2_demand_700 * frac_t33, avail3)

        remaining_avail3 = avail3 - take3_direct
        take_c3 = min(h2_demand_700 * frac_c3, remaining_avail3)

        return take1, take2, take3_direct, take_c3

    def step_all(self, h2_from_ele, h2_to_c1, h2_from_c1, h2_to_c2,
                 h2_from_c2, h2_demand_350, h2_for_fc,
                 h2_demand_700, avg_sog_350=0.2, avg_sog_700=0.2):
        """
        更新所有储罐状态 (统一 Low→High 差压级联协议, 动态压力门控).

        H₂ 流动规则 (SOG 边界随储罐 SOC 动态变化):
          T1  ← Electrolyzer;  T1 → C1
          T2  ← C1;            T2 → C2
          T3₁ ← C2(差压充装);  T3₁ → FuelCell + 350-bar(Phase1) + 700-bar(Phase1)
          T3₂ ← C2(差压充装);  T3₂ → 350-bar(Phase2) + 700-bar(Phase2)
          T3₃ ← C2(差压充装);  T3₃ → 350-bar(Phase3 顶充) + 700-bar(Phase3) + C3(Phase4)
          C3  : T3₃ → 700 bar → LDV (仅处理级联直充覆盖不到的增压末段)

        当储罐实际压强 (max_p × SOC) ≤ 车辆背压 + min_dp 时, 该罐不参与充装.

        返回: (overall_soc, total_excess, total_shortage, c3_flow)
          c3_flow: 需要经过 C3 增压的流量 (kg/h), 供 env.py 计算 C3 功耗
        """
        # T1 / T2
        soc_t1, excess_t1, short_t1 = self.t1.step(h2_from_ele, h2_to_c1)
        soc_t2, excess_t2, short_t2 = self.t2.step(h2_from_c1, h2_to_c2)

        # C2 差压充装分配 (按 deficit 权重)
        fill1, fill2, fill3 = self._cascade_fill(h2_from_c2)

        # 350-bar HDV: T3₁→T3₂→T3₃ (动态压力门控, T3₃ 在 T3₂ 压力不足时顶充)
        take_350_t1, take_350_t2, take_350_t3 = \
            self._cascade_discharge_350(h2_demand_350, avg_sog_350)

        # 700-bar LDV: T3₁→T3₂→T3₃ 直充 + T3₃→C3 增压
        take_700_t1, take_700_t2, take_700_t3_direct, take_700_c3 = \
            self._cascade_discharge_700(h2_demand_700, avg_sog_700)

        # T3₁: FC + 350-bar(Phase1) + 700-bar(Phase1 直充)
        soc_t3_1, ex1, sh1 = self.t3_1.step(fill1, h2_for_fc + take_350_t1 + take_700_t1)
        # T3₂: 350-bar(Phase2) + 700-bar(Phase2 直充)
        soc_t3_2, ex2, sh2 = self.t3_2.step(fill2, take_350_t2 + take_700_t2)
        # T3₃: 350-bar(Phase3 顶充) + 700-bar(Phase3 直充) + C3(Phase4 增压)
        soc_t3_3, ex3, sh3 = self.t3_3.step(fill3, take_350_t3 + take_700_t3_direct + take_700_c3)

        total_excess = excess_t1 + excess_t2 + ex1 + ex2 + ex3
        total_shortage = short_t1 + short_t2 + sh1 + sh2 + sh3

        total_capacity = (self.t1.capacity + self.t2.capacity +
                          self.t3_1.capacity + self.t3_2.capacity + self.t3_3.capacity)
        total_mass = (self.t1.level + self.t2.level +
                      self.t3_1.level + self.t3_2.level + self.t3_3.level)
        overall_soc = total_mass / total_capacity

        return overall_soc, total_excess, total_shortage, take_700_c3

    def get_total_soc(self):
        total_capacity = (self.t1.capacity + self.t2.capacity +
                          self.t3_1.capacity + self.t3_2.capacity + self.t3_3.capacity)
        total_mass = (self.t1.level + self.t2.level +
                      self.t3_1.level + self.t3_2.level + self.t3_3.level)
        return total_mass / total_capacity

    def get_tank_socs(self):
        return {
            't1':   self.t1.get_soc(),
            't2':   self.t2.get_soc(),
            't3_1': self.t3_1.get_soc(),  # 200 bar
            't3_2': self.t3_2.get_soc(),  # 350 bar
            't3_3': self.t3_3.get_soc(),  # 500 bar
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


class FCEVehicle(Vehicle):
    """
    燃料电池车 (Fuel Cell Electric Vehicle)
    基于SAE J2601协议，快速加氢（3-5分钟）
    target_pressure: 350 bar (公交/卡车) 或 700 bar (乘用车)
    """
    def __init__(self, vehicle_id, arrival_time, tank_capacity, sog_initial, sog_target,
                 target_pressure=700):
        super().__init__(vehicle_id, arrival_time, 'FCEV')
        self.tank_capacity = tank_capacity  # kg H2
        self.sog_initial = sog_initial  # State of Gas (0-1)
        self.sog_target = sog_target
        self.target_pressure = target_pressure  # 350 or 700 bar
        
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
        """计算加氢收入 (按目标压力区分单价)"""
        price = (Config.fcev_350bar_service_price if self.target_pressure == 350
                 else Config.fcev_service_price)
        return self.h2_needed * price


class FCEVDemandGenerator:
    """
    FCEV需求生成器 (仅燃料电池车加氢，已移除EV)
    基于真实用户行为模式的需求模拟
    """
    def __init__(self):
        self.vehicle_id_counter = 0
        self.current_day = 0  # 追踪天数用于周模式
        
    def _get_time_multiplier(self, hour_of_day, day_of_week=None):
        """根据时段和星期获取到达率倍数"""
        if hour_of_day in Config.peak_morning_hours:
            base_multiplier = Config.peak_arrival_multiplier * 1.2
        elif hour_of_day in Config.peak_evening_hours:
            base_multiplier = Config.peak_arrival_multiplier * 1.0
        elif hour_of_day in Config.midday_hours:
            base_multiplier = Config.midday_multiplier
        elif hour_of_day in Config.offpeak_hours:
            base_multiplier = Config.offpeak_multiplier
        else:
            base_multiplier = 1.0
        
        if day_of_week is not None:
            if day_of_week >= 5:
                if hour_of_day in Config.peak_morning_hours:
                    base_multiplier *= 0.6
                elif 10 <= hour_of_day <= 20:
                    base_multiplier *= 1.3
            else:
                if hour_of_day in Config.peak_morning_hours or hour_of_day in Config.peak_evening_hours:
                    base_multiplier *= 1.1
        
        return base_multiplier * np.random.uniform(0.6, 1.4)
    
    def generate_vehicles(self, current_step):
        """
        生成当前时段到达的FCEV车辆
        
        返回:
        - fcev_list: FCEV列表
        """
        hour_of_day = int((current_step * Config.dt) % 24)
        day_of_year = int(current_step / 96)
        day_of_week = day_of_year % 7
        
        time_multiplier = self._get_time_multiplier(hour_of_day, day_of_week)
        adjusted_arrival_rate = Config.base_vehicle_arrival_rate * time_multiplier
        
        mean_arrivals = adjusted_arrival_rate * Config.dt
        num_arrivals = min(np.random.poisson(mean_arrivals), 15)
        
        fcev_list = []
        for _ in range(num_arrivals):
            self.vehicle_id_counter += 1
            fcev = self._generate_fcev(self.vehicle_id_counter, current_step, hour_of_day)
            fcev_list.append(fcev)
        
        return fcev_list
    
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
        
        # 350-bar 车型比例 (公交/卡车)
        target_pressure = 350 if np.random.random() < Config.fcev_350bar_ratio else 700
        
        return FCEVehicle(
            vehicle_id, arrival_time, tank_capacity,
            sog_initial, sog_target, target_pressure
        )


class FCEVServiceStation:
    """
    FCEV加氢服务站 (统一 Low→High 差压级联协议)
    - 350-bar HDV: T3₁→T3₂ 级联直充 (低压优先, SOG 门控)
    - 700-bar LDV: T3₁→T3₂→T3₃ 级联直充 + T3₃→C3→700 bar 增压末段
    - 分别追踪两类车型的平均 SOG, 用于差压级联和 APC 计算
    """
    def __init__(self):
        self.fcev_dispensers = Config.max_concurrent_fcev
        
        self.fcev_queue = []
        self.fcev_being_served = []
        
        self.total_fcev_served = 0
        self.total_fcev_revenue = 0.0
        self.total_vehicles_delayed = 0
        
        self.current_fcev_350_avg_sog = 0.2
        self.current_fcev_700_avg_sog = 0.2
    
    def add_vehicles(self, fcev_list):
        self.fcev_queue.extend(fcev_list)
    
    def step(self, h2_available_350, h2_available_700, dt=None):
        """
        执行一个时间步

        参数:
        - h2_available_350: T3₁+T3₂ 可供 350-bar 车型的氢气量 (kg)
        - h2_available_700: T3₃+T3₂+T3₁ 级联可供 700-bar 车型经 C3 使用的氢气量 (kg)

        返回:
        - h2_demand_350: 350-bar 需求 (kg/h)
        - h2_demand_700: 700-bar 需求 (kg/h，送入 C3)
        - fcev_revenue: 加氢收入
        - unmet_demand_penalty: 未满足需求惩罚
        """
        if dt is None:
            dt = Config.dt
        
        self._update_serving_vehicles(dt)
        
        h2_demand_350, h2_demand_700, revenue, penalty = \
            self._serve_fcev_queue(h2_available_350, h2_available_700, dt)
        
        # 更新两类车型的平均 SOG (用于差压级联门控和 C3 APC)
        for tp, attr in [(350, 'current_fcev_350_avg_sog'), (700, 'current_fcev_700_avg_sog')]:
            serving = [v for v in self.fcev_being_served if v.target_pressure == tp]
            queued  = [v for v in self.fcev_queue if v.target_pressure == tp]
            if serving:
                setattr(self, attr, sum(v.sog_initial for v in serving) / len(serving))
            elif queued:
                sample = queued[:3]
                setattr(self, attr, sum(v.sog_initial for v in sample) / len(sample))
            else:
                setattr(self, attr, 0.2)
        
        return h2_demand_350, h2_demand_700, revenue, penalty
    
    def _update_serving_vehicles(self, dt):
        for fcev in self.fcev_being_served[:]:
            if fcev.service_start_time is not None and dt >= fcev.fill_time_hours:
                fcev.is_completed = True
                self.fcev_being_served.remove(fcev)
                self.total_fcev_served += 1
                self.total_fcev_revenue += fcev.get_refueling_revenue()
    
    def _serve_fcev_queue(self, h2_available_350, h2_available_700, dt):
        """按目标压力分别分配氢气资源"""
        h2_demand_350 = 0.0
        h2_demand_700 = 0.0
        revenue = 0.0
        penalty = 0.0
        
        available_dispensers = self.fcev_dispensers - len(self.fcev_being_served)
        avail_350 = h2_available_350 * dt  # 转换为本时段可用量 (kg)
        avail_700 = h2_available_700 * dt
        
        for fcev in self.fcev_queue[:]:
            if available_dispensers <= 0:
                break
            
            if fcev.target_pressure == 350:
                h2_avail = avail_350
            else:
                h2_avail = avail_700
            
            if fcev.check_delay_eligibility(h2_avail / dt if dt > 0 else 0):
                self.total_vehicles_delayed += 1
                penalty += Config.penalty_vehicle_waiting * dt * 2
                continue
            
            if fcev.h2_needed <= h2_avail:
                self.fcev_being_served.append(fcev)
                self.fcev_queue.remove(fcev)
                fcev.service_start_time = 0
                fcev.is_being_served = True
                revenue += fcev.get_refueling_revenue()
                available_dispensers -= 1
                if fcev.target_pressure == 350:
                    h2_demand_350 += fcev.h2_flow_rate
                    avail_350 -= fcev.h2_needed
                else:
                    h2_demand_700 += fcev.h2_flow_rate
                    avail_700 -= fcev.h2_needed
            else:
                penalty += Config.penalty_unmet_h2_demand * fcev.h2_needed * 0.1
        
        return h2_demand_350, h2_demand_700, revenue, penalty
    
    def get_statistics(self):
        queue_350 = sum(1 for v in self.fcev_queue if v.target_pressure == 350)
        queue_700 = sum(1 for v in self.fcev_queue if v.target_pressure == 700)
        return {
            'fcev_served': self.total_fcev_served,
            'fcev_revenue': self.total_fcev_revenue,
            'vehicles_delayed': self.total_vehicles_delayed,
            'fcev_queue_length': len(self.fcev_queue),
            'fcev_queue_350': queue_350,
            'fcev_queue_700': queue_700,
            'fcev_being_served': len(self.fcev_being_served)
        }
