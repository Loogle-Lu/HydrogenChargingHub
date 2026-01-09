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

    # 2. 级联压缩机 (Cascade Compressor - 论文 Eq.10)
    comp_efficiency = 0.75
    comp_input_pressure = 1.0  # bar
    comp_output_pressure = 350.0  # bar
    H2_gamma = 1.41
    H2_R = 4124.0
    H2_molar_mass = 2.016
    T_in = 298.15

    # 3. 储氢罐
    storage_capacity_kg = 500.0
    storage_min_level = 0.05
    storage_max_level = 0.95
    storage_initial = 0.5  # I2S 目标 SOC

    # 4. 燃料电池
    fc_max_power = 500.0
    fc_efficiency = 16.0  # kWh/kg

    # 5. 冷却机 (Chiller - 新增)
    chiller_cop = 3.0
    gas_heat_capacity = 14.3  # kJ/(kg*K)
    target_temp = 298.15  # K (冷却回常温)

    # 6. 经济参数
    hydrogen_price = 10.0  # $/kg
    electricity_price_sell_coef = 1.0
    penalty_unmet_demand = 500.0  # 缺氢惩罚