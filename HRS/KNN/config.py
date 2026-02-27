import numpy as np
import os


class Config:
    # --- 路径配置 (使用 r 前缀处理 Windows 路径) ---
    # 请确保这些路径指向你的实际文件位置
    path_price = r"E:\桌面图标\Graduation Thesis\Hydrogen\HydrogenChargingHub\price_after_MAD_96.pkl"
    path_pv = r"E:\桌面图标\Graduation Thesis\Hydrogen\HydrogenChargingHub\pv_power_100.pkl"
    path_wind = r"E:\桌面图标\Graduation Thesis\Hydrogen\HydrogenChargingHub\wd_power_150.pkl"

    # --- 全局模拟参数 ---
    dt = 0.25  # 15分钟一个点 (0.25小时)
    steps_per_day = 96
    episode_length = 96  # 训练一个回合代表一天

    # --- I2S (Identical Initial State) 约束参数 ---
    enable_i2s_constraint = True

    # [修改] 降低I2S惩罚权重，避免过度保守
    # 原值10000导致Agent不敢制氢（害怕SOC偏离）
    # v3.5: 2000→1500 进一步放宽，改善Profit曲线向下趋势
    i2s_penalty_weight = 1500.0

    # --- 物理组件参数 ---
    # 1. 电解槽
    ele_max_power = 1000.0  # kW
    ele_efficiency = 50.0  # kWh/kg
    
    # 2. 多级级联压缩机系统 (Multi-Stage Cascade Compressor System)
    # 公共压缩机参数
    comp_efficiency = 0.75  # 额定效率（50%负载）
    H2_gamma = 1.41
    H2_R = 4124.0  # J/(kg·K)
    H2_molar_mass = 2.016
    T_in = 298.15  # K
    
    # v3.1: 变速驱动效率曲线 (Variable Speed Drive)
    # 负载率 -> 效率映射
    vsd_efficiency_curve = {
        0.0: 0.45,   # 空载效率低
        0.2: 0.60,   # 20%负载
        0.5: 0.75,   # 50%负载（额定点）
        0.8: 0.78,   # 80%负载（最优点）
        1.0: 0.74    # 满载效率略降
    }
    enable_vsd = True  # 启用变速驱动
    
    # v3.1: 智能旁路控制 (Intelligent Bypass)
    enable_bypass = True  # 启用旁路控制
    bypass_pressure_threshold = 0.8  # 储罐压力>目标压力80%时可旁路
    bypass_demand_threshold = 5.0  # kg/h，需求低于此值时考虑旁路
    
    # v3.1: 动态级间冷却 (Dynamic Intercooling)
    enable_dynamic_cooling = True  # 启用动态冷却
    min_intercool_temp = 300.15  # K (27°C) 深度冷却
    max_intercool_temp = 313.15  # K (40°C) 轻度冷却
    cooling_price_threshold = 0.10  # $/kWh 高于此电价则轻度冷却
    
    # v3.1: 压力自适应控制 (Adaptive Pressure)
    enable_adaptive_pressure = True  # 启用自适应压力
    # 根据FCEV SOG分段充装压力目标
    adaptive_pressure_map = {
        0.0: 700,   # SOG 0-30%: 700bar快速充装
        0.3: 700,
        0.4: 500,   # SOG 30-60%: 500bar中速充装
        0.6: 500,
        0.7: 350,   # SOG 60-80%: 350bar慢速充装
        0.8: 350,
        0.9: 200    # SOG 80%+: 200bar涓流充电
    }
    
    # C1: 第一级压缩机 (Electrolyzer output -> T2)
    c1_input_pressure = 2.0  # bar (T1压力)
    c1_output_pressure = 35.0  # bar (T2压力)
    c1_max_flow = 20.0  # kg/h
    
    # C2: 第二级级联压缩机 (T2 -> T3 cascaded subsystem)
    c2_input_pressure = 35.0  # bar
    c2_output_pressure = 500.0  # bar (填充T3₁, T3₂, T3₃)
    c2_max_flow = 20.0  # kg/h
    
    # C3: 第三级压缩机 (T3/T4 -> D2 for LDFV)
    c3_input_pressure = 500.0  # bar
    c3_output_pressure = 700.0  # bar (LDFV充装压力) - 可动态调整
    c3_max_flow = 15.0  # kg/h

    # 3. 多储罐系统 (Multi-Tank Storage System)
    # T1: 缓冲罐 (Electrolyzer output buffer)
    t1_capacity_kg = 100.0  # kg
    t1_max_pressure = 2.0  # barg
    t1_initial_soc = 0.5
    
    # T2: 中压储罐
    t2_capacity_kg = 200.0  # kg
    t2_max_pressure = 35.0  # barg
    t2_initial_soc = 0.5
    
    # T3: 高压级联储罐组 (Cascaded high-pressure tanks)
    t3_1_capacity_kg = 150.0  # kg
    t3_2_capacity_kg = 150.0  # kg
    t3_3_capacity_kg = 150.0  # kg
    t3_max_pressure = 500.0  # barg
    t3_initial_soc = 0.5
    
    # T4: 超高压缓冲罐 (Buffer for 7kg LDFV fast service)
    t4_capacity_kg = 16.0  # kg (for 7kg LDFV service)
    t4_max_pressure = 900.0  # barg
    t4_initial_soc = 0.5
    
    # 储罐通用参数
    storage_min_level = 0.05
    storage_max_level = 0.95
    storage_initial = 0.5  # I2S 目标 SOC (用于总系统)

    # 4. 燃料电池
    fc_max_power = 500.0
    fc_efficiency = 16.0  # kWh/kg

    # 5. 冷却机 (Chiller - 线性模型)
    chiller_rated_capacity = 500.0  # kW (额定冷却能力)
    chiller_rated_cop = 3.5  # 固定COP
    
    # 热力学参数
    target_temp = 298.15  # K (冷却目标温度)

    # 6. FCEV加氢需求参数 (基于SAE J2601协议)
    # Hyundai Nexo规格 (参考图片)
    fcev_tank_capacity = 6.3  # kg H2
    fcev_tank_volume = 156.0  # L
    fcev_target_pressure = 700.0  # bar
    fcev_min_pressure = 350.0  # bar (最低充装压力)
    
    # 加氢时间和速率 (SAE J2601)
    fcev_target_fill_time = 4.0  # minutes (3-5分钟目标)
    fcev_min_fill_time = 3.0  # minutes
    fcev_max_fill_time = 5.0  # minutes
    
    # 压力爬升率 (APRR - Average Pressure Ramp Rate)
    fcev_aprr_target = 10.0  # bar/min
    fcev_max_pressure_ramp = 15.0  # bar/min (最大爬升率)
    
    # 到站状态分布
    fcev_sog_arrival_mean = 0.20  # State of Gas平均20% (低渗透率，充装需求高)
    fcev_sog_arrival_std = 0.10
    fcev_sog_target = 0.95  # 目标充装到95%
    
    # 加氢服务价格
    fcev_service_price = 12.0  # $/kg (略高于生产成本)
    
    # 7. EV充电需求参数
    # 典型EV规格 (20% SOC初始 -> 80% SOC快充策略)
    ev_battery_capacity_mean = 65.0  # kWh (平均电池容量)
    ev_battery_capacity_std = 15.0  # kWh (标准差，考虑不同车型)
    
    # 充电功率
    ev_fast_charge_power = 150.0  # kW (DC快充)
    ev_slow_charge_power = 11.0  # kW (AC慢充 Level 2)
    ev_ultra_fast_charge_power = 350.0  # kW (超快充，如Tesla V3)
    
    # 充电时间特性
    ev_fast_charge_time = 0.5  # hours (30分钟快充策略: 20%->80%)
    ev_slow_charge_time = 6.0  # hours (慢充完全充满)
    
    # 到站状态分布
    ev_soc_arrival_mean = 0.25  # 平均到站SOC 25%
    ev_soc_arrival_std = 0.12
    ev_soc_target_fast = 0.80  # 快充目标80% (保护电池)
    ev_soc_target_slow = 0.95  # 慢充目标95%
    
    # EV充电服务价格
    ev_service_price = 0.35  # $/kWh (含服务费)
    
    # 8. 混合需求场景参数 (EV + FCEV)
    # 车辆到达率模型
    base_vehicle_arrival_rate = 10.0  # vehicles/hour (基准)
    ev_fcev_ratio = 0.70  # EV占70%, FCEV占30% (当前市场渗透率)
    
    # 充电模式分布 (EV)
    ev_fast_charge_ratio = 0.75  # 75%选择快充
    ev_slow_charge_ratio = 0.20  # 20%慢充
    ev_ultra_fast_ratio = 0.05  # 5%超快充
    
    # 时段系数 (模拟一天的需求曲线)
    # 高峰期: 早7-9AM (通勤), 晚5-7PM (下班)
    # 午间: 12-2PM (中度)
    # 深夜: 11PM-6AM (低谷)
    peak_morning_hours = [7, 8]
    peak_evening_hours = [17, 18]
    midday_hours = [12, 13]
    offpeak_hours = [0, 1, 2, 3, 4, 5, 22, 23]
    
    peak_arrival_multiplier = 2.8  # 高峰期到达率倍数
    midday_multiplier = 1.5
    offpeak_multiplier = 0.3
    
    # 9. 需求响应 (Demand Response) 能力
    enable_demand_response = True
    
    # 电价阈值
    dr_price_threshold_high = 0.12  # $/kWh (高电价)
    dr_price_threshold_low = 0.04  # $/kWh (低电价)
    
    # EV需求响应灵活性 (图片显示EV可参与DR)
    ev_dr_flexibility_fast = 0.30  # 快充30%可延迟
    ev_dr_flexibility_slow = 0.60  # 慢充60%可延迟
    ev_max_delay_hours = 2.0  # 最大延迟2小时
    
    # FCEV需求响应灵活性 (图片显示FCEV难以参与DR，加氢时间短)
    fcev_dr_flexibility = 0.05  # 仅5%可短暂延迟
    fcev_max_delay_minutes = 10.0  # 最大延迟10分钟
    
    # 10. 加氢站容量约束 (基于级联储罐系统)
    # 同时服务能力
    max_concurrent_fcev = 3  # 最多同时加氢3辆FCEV (D1/D2/D3)
    max_concurrent_ev_fast = 4  # 快充桩数量
    max_concurrent_ev_slow = 8  # 慢充桩数量
    
    # 11. 经济参数
    hydrogen_price = 10.0  # $/kg (生产成本)
    electricity_price_sell_coef = 1.0
    
    # 储能套利激励参数 (新增)
    enable_arbitrage_bonus = True  # 启用储能套利奖励
    # v3.5: 100→50 降低人工奖励比重，让Profit信号主导，改善训练曲线
    arbitrage_bonus_coef = 50.0  # 套利奖励系数
    soc_health_bonus = 20.0  # SOC在[0.4,0.6]时的健康奖励 (原硬编码50)
    price_threshold_low = 0.06  # $/kWh (低电价阈值，低于此值鼓励制氢)
    price_threshold_high = 0.10  # $/kWh (高电价阈值，高于此值鼓励放电)
    
    # 惩罚参数
    penalty_unmet_h2_demand = 500.0  # $/kg 缺氢惩罚
    penalty_unmet_ev_demand = 150.0  # $/vehicle 无法服务EV惩罚
    penalty_vehicle_waiting = 30.0  # $/vehicle/hour 等待时间惩罚
    
    # 12. 电池储能系统 (Battery Energy Storage System, BESS) - v2.6新增
    # 核心参数
    battery_capacity = 500.0  # kWh (储能容量，约相当于8辆EV电池)
    battery_max_charge_power = 250.0  # kW (最大充电功率，C-rate=0.5)
    battery_max_discharge_power = 250.0  # kW (最大放电功率，C-rate=0.5)
    
    # 效率参数
    battery_charge_efficiency = 0.95  # 充电效率 (典型锂电池)
    battery_discharge_efficiency = 0.95  # 放电效率
    # 往返效率 = 0.95 × 0.95 = 90.25% (远高于氢储能的30-40%)
    
    # SOC约束
    battery_min_soc = 0.10  # 最小SOC (保护电池寿命)
    battery_max_soc = 0.90  # 最大SOC (保护电池寿命)
    battery_initial_soc = 0.50  # 初始SOC
    
    # 寿命参数
    battery_lifetime_cycles = 5000  # 循环寿命 (80% DOD)
    battery_degradation_cost = 0.05  # $/kWh (折旧成本)
    
    # 电池储能策略
    enable_battery_storage = True  # 启用电池储能
    battery_priority_threshold = 100.0  # kW (功率阈值，小于此值优先用电池)
    
    # 协同策略 (电池 vs 氢储能)
    # - 电池: 短时调峰 (秒-分钟)，高效率，快响应
    # - 氢气: 长时储能 (小时-天)，大容量，低效率
    # - 协同: 电池削峰填谷，氢气季节调节
