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

    # [关键修改] 大幅提高惩罚权重
    # 逻辑: 满罐氢气价值约 $5000 (500kg * $10).
    # 如果 Agent 偷光氢气 (偏差0.5), 惩罚必须远大于 $5000.
    # 设为 10000.0 可以确保 Agent 即使在探索初期也不敢随意违背 I2S 约束.
    i2s_penalty_weight = 10000.0

    # --- 物理组件参数 ---
    # 1. 电解槽
    ele_max_power = 1000.0  # kW
    ele_efficiency = 50.0  # kWh/kg

    # 2. 多级级联压缩机系统 (Multi-Stage Cascade Compressor System)
    # 公共压缩机参数
    comp_efficiency = 0.75
    H2_gamma = 1.41
    H2_R = 4124.0  # J/(kg·K)
    H2_molar_mass = 2.016
    T_in = 298.15  # K
    
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
    c3_output_pressure = 700.0  # bar (LDFV充装压力)
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

    # 5. 冷却机 (Chiller - 复杂非线性模型)
    # 基础参数
    chiller_rated_capacity = 500.0  # kW (额定冷却能力)
    chiller_rated_cop = 3.5  # 额定工况COP
    chiller_min_plr = 0.1  # 最小部分负荷率 (Part Load Ratio)
    
    # 非线性性能曲线参数 (基于实际工业冷水机组性能)
    # COP = f(PLR, T_ambient) 的多项式系数
    # COP(PLR) = a0 + a1*PLR + a2*PLR^2 + a3*PLR^3
    chiller_cop_plr_coef = [0.5, 0.8, 0.7, -1.0]  # 部分负荷性能曲线
    
    # 温度修正系数 (环境温度影响)
    chiller_temp_coef = [1.2, -0.008, 0.00002]  # COP_temp = b0 + b1*T + b2*T^2
    
    # 启停能耗惩罚
    chiller_startup_energy = 10.0  # kWh (每次启动能耗)
    chiller_min_runtime = 0.5  # hours (最小运行时间)
    
    # 热力学参数
    gas_heat_capacity = 14.3  # kJ/(kg*K)
    target_temp = 298.15  # K (冷却目标温度)
    ambient_temp_nominal = 298.15  # K (名义环境温度)

    # 6. 经济参数
    hydrogen_price = 10.0  # $/kg
    electricity_price_sell_coef = 1.0
    penalty_unmet_demand = 500.0  # 缺氢惩罚