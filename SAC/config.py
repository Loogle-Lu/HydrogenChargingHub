import numpy as np
import os


class Config:
    # --- 路径配置 ---
    # 使用原始字符串 r"..." 避免 Windows 路径报错
    path_price = r"E:\桌面图标\Graduation Thesis\Hydrogen\HydrogenChargingHub\price_after_MAD_96.pkl"
    path_pv = r"E:\桌面图标\Graduation Thesis\Hydrogen\HydrogenChargingHub\pv_power_100.pkl"
    path_wind = r"E:\桌面图标\Graduation Thesis\Hydrogen\HydrogenChargingHub\wd_power_150.pkl"

    # --- 全局模拟参数 ---
    dt = 0.25  # 15分钟一个点
    steps_per_day = 96
    episode_length = 96  # 训练一个回合代表一天

    # --- I2S (Identical Initial State) 约束参数 ---
    enable_i2s_constraint = True
    # [修改] 大幅提高惩罚权重，强迫 SOC 回归，解决 SOC 持续下降问题
    i2s_penalty_weight = 200.0

    # --- 物理组件参数 ---
    # 电解槽
    ele_max_power = 1000.0  # kW
    ele_efficiency = 50.0  # kWh/kg

    # 压缩机
    comp_efficiency = 0.75
    comp_input_pressure = 30.0  # bar
    comp_output_pressure = 350.0  # bar
    H2_gamma = 1.41
    H2_R = 4124.0
    T_ambient = 298.15

    # 储氢罐
    storage_capacity_kg = 500.0
    storage_min_level = 0.05
    storage_max_level = 0.95
    storage_initial = 0.5  # I2S 目标值

    # 燃料电池
    fc_max_power = 500.0
    fc_efficiency = 16.0  # kWh/kg

    # 冷却机
    chiller_cop = 3.0
    gas_heat_capacity = 14.3
    target_temp = 253.15

    # 经济参数
    # [修改] 降低氢气价格 (60 -> 10)，使 FC 发电在电价高峰期具有竞争力，否则 Agent 只会卖氢气不发电
    hydrogen_price = 10.0  # $/kg

    # [修改] 提高卖电系数，鼓励向电网反送电
    electricity_price_sell_coef = 1.0

    # [修改] 提高缺氢惩罚，防止 SOC 耗尽
    penalty_unmet_demand = 500.0