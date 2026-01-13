# EV/FCEV混合需求建模完整总结

## 更新日期
2026-01-13  15:30

## 概述

基于您提供的三张图片，实现了完整的EV充电和FCEV加氢混合需求建模。系统现在能够同时服务电动车和燃料电池车，并考虑两种车辆的不同特性和需求响应能力。

---

## 1. 核心设计理念

### 1.1 EV vs FCEV 差异对比

| 特性 | EV (电动车) | FCEV (燃料电池车) |
|------|------------|------------------|
| **充能时间** | 30分钟-6小时 | 3-5分钟 |
| **能源形式** | 电力 (kWh) | 氢气 (kg) |
| **充能方式** | 快充/慢充/超快充 | SAE J2601协议加氢 |
| **需求响应能力** | 高 (30-60%可延迟) | 极低 (仅5%可延迟) |
| **服务价格** | $0.35/kWh | $12/kg |
| **典型容量** | 65 kWh | 6.3 kg H2 |
| **到站SOC/SOG** | 25% ± 12% | 20% ± 10% |
| **目标SOC/SOG** | 80% (快充), 95% (慢充) | 95% |

### 1.2 系统架构

```
┌─────────────────────────────────────────────────────────┐
│         集成充电加氢站 (Integrated Station)              │
└─────────────────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
  ┌─────▼──────┐         ┌─────▼──────┐
  │  EV充电站   │         │  FCEV加氢站 │
  │  (多桩)     │         │  (SAE J2601)│
  └─────┬──────┘         └─────┬──────┘
        │                       │
  ┌─────▼──────┐         ┌─────▼──────┐
  │ 电网供电    │         │ 储氢罐供氢  │
  │ + 可再生能源│         │ (级联系统)  │
  └────────────┘         └─────────────┘
```

---

## 2. 详细技术实现

### 2.1 FCEV加氢站模型 (SAE J2601协议)

#### 2.1.1 车辆规格 (基于Hyundai Nexo)
```python
class FCEVehicle:
    tank_capacity = 6.3 kg H2
    tank_volume = 156 L
    target_pressure = 700 bar
    fill_time = 3-5 minutes (目标4分钟)
```

#### 2.1.2 加氢协议关键参数

**压力爬升率控制**：
- **APRR** (Average Pressure Ramp Rate): 10 bar/min
- **最大爬升率**: 15 bar/min
- **目的**: 避免过热，确保安全

**加氢时间计算**：
```
fill_time = h2_needed / h2_flow_rate
h2_flow_rate = (SOG_target - SOG_initial) × tank_capacity / fill_time_hours
```

**示例**：
- 到站SOG: 20%
- 目标SOG: 95%
- 需求H2: (0.95 - 0.20) × 6.3 = 4.725 kg
- 加氢时间: 4分钟 = 0.067小时
- 流量: 4.725 / 0.067 = 70.5 kg/h

#### 2.1.3 供氢来源

FCEV从多级储罐系统获取氢气：

1. **D1分配器** (350 bar HDFV):
   - 从T3₁直接供应
   - 服务重型燃料电池车

2. **D2分配器** (700 bar LDFV):
   - 从T4超高压储罐供应
   - T4由C3压缩机充装 (500→700 bar)
   - 7kg缓冲容量，快速罐对罐充装

3. **D3分配器** (300 bar MEGC):
   - 从T3₂/T3₃供应
   - 服务氢气容器充装

#### 2.1.4 需求响应特性

**灵活性**: 极低 (5%)
- 加氢时间短，客户难以接受延迟
- 仅在氢气严重短缺时可短暂等待 (<10分钟)
- 高电价对FCEV加氢影响小

### 2.2 EV充电站模型

#### 2.2.1 充电模式

| 模式 | 功率 | 充电时间 | 占比 | 应用场景 |
|------|------|---------|------|---------|
| **超快充** | 350 kW | 15-20分钟 | 5% | 紧急充电 |
| **快充** | 150 kW | 30分钟 | 75% | 主流选择 |
| **慢充** | 11 kW | 6小时 | 20% | 过夜/长期停车 |

#### 2.2.2 充电需求计算

```python
class EVehicle:
    energy_needed = (SOC_target - SOC_initial) × battery_capacity
    charge_time = energy_needed / charge_power
```

**示例**：
- 电池容量: 65 kWh
- 到站SOC: 25%
- 目标SOC: 80% (快充策略)
- 需求电量: (0.80 - 0.25) × 65 = 35.75 kWh
- 快充时间: 35.75 / 150 = 0.238小时 ≈ 14分钟

#### 2.2.3 需求响应能力

**灵活性**: 高

| 充电模式 | DR灵活性 | 最大延迟 | 适用场景 |
|---------|---------|---------|---------|
| 快充 | 30% | 2小时 | 高电价时段 |
| 慢充 | 60% | 2小时 | 低SOC + 长停留时间 |
| 超快充 | 15% | 30分钟 | 紧急需求，难延迟 |

**DR触发条件**：
```python
if price > 0.12:  # 高电价阈值 ($/kWh)
    if random() < ev.dr_flexibility:
        delay_charging()
```

**收益**：
- 降低高峰时段电网负荷
- 利用低电价时段充电，降低成本
- 提高可再生能源利用率

### 2.3 混合需求生成器

#### 2.3.1 车辆到达模型

**基准到达率**: 10 vehicles/hour

**时段系数**：

| 时段 | 时间 | 倍数 | 到达率 (v/h) |
|------|------|------|------------|
| **早高峰** | 7-9 AM | 2.8× | 28 |
| **午间** | 12-2 PM | 1.5× | 15 |
| **晚高峰** | 5-7 PM | 2.8× | 28 |
| **低谷** | 11PM-6AM | 0.3× | 3 |
| **其他** | - | 1.0× | 10 |

**到达过程**: 泊松分布

```python
num_arrivals = np.random.poisson(arrival_rate × dt)
```

#### 2.3.2 车辆类型分布

- **EV占比**: 70% (反映当前市场渗透率)
- **FCEV占比**: 30%

**随机生成逻辑**：
```python
for each arrival:
    if random() < 0.70:
        generate EV
    else:
        generate FCEV
```

### 2.4 集成服务站运营

#### 2.4.1 设施容量

```python
class IntegratedServiceStation:
    # EV充电桩
    max_concurrent_ev_fast = 4  # 快充桩
    max_concurrent_ev_slow = 8  # 慢充桩
    
    # FCEV加氢枪
    max_concurrent_fcev = 3  # D1/D2/D3
```

#### 2.4.2 服务调度逻辑

**EV充电调度**：
```
1. 检查需求响应条件 (电价)
2. 判断是否可延迟
3. 分配可用充电桩
4. 计算充电功率和收入
5. 如无法服务，加入等待队列
```

**FCEV加氢调度**：
```
1. 检查氢气可用性
2. 判断是否可短暂延迟
3. 分配可用加氢枪
4. 计算氢气流量和收入
5. 如氢气不足，产生惩罚
```

#### 2.4.3 队列管理

- **FIFO原则**: 先到先服务
- **动态优先级**: FCEV优先级略高 (加氢时间短)
- **等待惩罚**: 
  - EV: $30/vehicle/hour
  - FCEV: $60/vehicle/hour (更高，因为客户预期快速服务)

---

## 3. 系统能量流与经济模型

### 3.1 完整能量流

```
可再生能源 (PV + Wind)
      ↓
┌─────────────┐
│ 电网互动     │ ←→ 外部电网
└─────────────┘
      ↓
┌─────────────────────────────────────┐
│          电力分配                    │
├─────────────────────────────────────┤
│  1. EV充电负荷 (直接)                │
│  2. 电解槽 (制氢)                    │
│  3. 压缩机 (C1/C2/C3)               │
│  4. Chiller (冷却)                  │
│  5. 燃料电池 (应急发电)              │
└─────────────────────────────────────┘
      ↓
氢气生产 → 储罐系统 → FCEV加氢
```

### 3.2 收入来源

| 收入类型 | 单价 | 计算方式 | 占比预估 |
|---------|------|---------|---------|
| **EV充电** | $0.35/kWh | energy × price | 40-50% |
| **FCEV加氢** | $12/kg | h2_mass × price | 30-40% |
| **电网售电** | 电价×系数 | excess_power × price | 10-20% |

**示例日收入估算**：
- EV服务: 50辆 × 35kWh × $0.35 = $612.5
- FCEV服务: 20辆 × 4.5kg × $12 = $1,080
- 电网售电: 200kWh × $0.08 = $16
- **总计**: ~$1,708/天

### 3.3 成本构成

| 成本类型 | 计算方式 | 占比预估 |
|---------|---------|---------|
| **购电成本** | net_power (if <0) × price | 40-50% |
| **制氢成本** | 已含在购电中 | - |
| **设备折旧** | 固定 (未建模) | - |
| **运维成本** | 固定 (未建模) | - |
| **缺氢惩罚** | shortage × $500/kg | <5% |
| **等待惩罚** | queue × $30-60/v/h | <5% |

---

## 4. 强化学习环境更新

### 4.1 观察空间 (8维)

```python
observation = [
    storage_total_soc,     # 0-1, 总氢气储量
    storage_t1_soc,        # 0-1, T1低压罐
    storage_t2_soc,        # 0-1, T2中压罐
    electricity_price,     # $/kWh
    renewable_power,       # kW
    ev_queue_norm,         # 0-1, 归一化EV队列长度
    fcev_queue_norm,       # 0-1, 归一化FCEV队列长度
    hour_of_day_norm       # 0-1, 时段 (0=午夜, 0.5=正午)
]
```

**关键改进**：
- 添加了EV/FCEV队列信息，Agent可感知需求压力
- 添加时段信息，Agent可学习时变需求模式

### 4.2 动作空间 (2维)

```python
action = [
    electrolyzer_ratio,    # 0-1, 电解槽功率比例
    fuel_cell_ratio        # 0-1, 燃料电池功率比例
]
```

**自动化组件**：
- 压缩机C1/C2/C3: 基于储罐SOC和需求自动调节
- EV/FCEV服务站: 自动调度和服务

### 4.3 奖励函数

```python
reward = (ev_revenue + fcev_revenue + grid_revenue) 
         - (grid_cost) 
         - (shortage_penalty + waiting_penalty)
         - (I2S_penalty if done)
```

**奖励设计原则**：
1. **正向激励**: 服务更多车辆，增加收入
2. **成本优化**: 低电价时制氢，高电价时用燃料电池
3. **惩罚机制**: 
   - 缺氢导致无法服务 FCEV → 重罚
   - 长队列等待 → 持续惩罚
   - 终端SOC偏离初始值 → I2S惩罚

---

## 5. 关键特性与创新点

### 5.1 真实性

✅ **FCEV加氢**:
- 基于SAE J2601实际协议
- 3-5分钟快速充装
- 700bar高压氢气供应
- Hyundai Nexo真实规格

✅ **EV充电**:
- 多种充电模式 (快/慢/超快)
- 真实充电时间
- 基于实际电池容量分布

✅ **时变需求**:
- 早晚高峰模式
- 泊松到达过程
- 随机SOC/SOG分布

### 5.2 复杂性

✅ **多能源耦合**:
- 电力 (EV充电)
- 氢气 (FCEV加氢)
- 电-氢转换 (电解槽)
- 氢-电转换 (燃料电池)

✅ **多时间尺度**:
- 短时: FCEV加氢 (3-5分钟)
- 中时: EV快充 (30分钟)
- 长时: EV慢充 (6小时)
- 决策: 15分钟 (0.25小时)

✅ **多目标优化**:
- 服务质量 (减少等待)
- 经济效益 (最大化利润)
- 能源效率 (利用可再生能源)
- 储能管理 (I2S约束)

### 5.3 需求响应能力差异

**EV (高DR能力)**:
- 充电可延迟 2小时
- 利用价格信号调节
- 有助于电网削峰填谷

**FCEV (低DR能力)**:
- 几乎无法延迟
- 即时服务需求
- 对氢气供应稳定性要求高

**系统设计启示**:
- 氢气储备必须充足 (FCEV刚性需求)
- EV充电可作为灵活负荷
- 综合考虑两种需求特性优化调度

---

## 6. 使用示例

### 6.1 训练代码

```python
from env import HydrogenEnv
from sac import SAC

env = HydrogenEnv()
agent = SAC(state_dim=8, action_dim=2)

for episode in range(200):
    state = env.reset()
    episode_reward = 0
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        # 观察EV/FCEV统计
        print(f"EV served: {info['ev_served']}, "
              f"FCEV served: {info['fcev_served']}, "
              f"EV queue: {info['ev_queue']}, "
              f"FCEV queue: {info['fcev_queue']}")
        
        agent.update(replay_buffer, batch_size=64)
        state = next_state
        episode_reward += reward
```

### 6.2 关键Info输出

```python
info = {
    # 氢气系统
    'soc': 0.52,
    'tank_socs': {'t1': 0.48, 't2': 0.55, ...},
    
    # 能量平衡
    'h2_production_load': 850.2,  # kW
    'ev_charging_load': 450.0,    # kW
    'total_load': 1300.2,         # kW
    'net_power': -120.5,          # kW (从电网购电)
    
    # EV/FCEV服务
    'ev_served': 35,
    'fcev_served': 12,
    'ev_revenue': 428.75,         # $
    'fcev_revenue': 648.00,       # $
    'ev_queue': 2,
    'fcev_queue': 1,
    'vehicles_delayed': 5,
    
    # 经济
    'profit': 956.20              # $
}
```

---

## 7. 参数配置速查

### 7.1 FCEV参数 (config.py)

```python
# Hyundai Nexo规格
fcev_tank_capacity = 6.3  # kg
fcev_target_pressure = 700.0  # bar
fcev_target_fill_time = 4.0  # minutes

# 到站分布
fcev_sog_arrival_mean = 0.20  # 20%
fcev_sog_arrival_std = 0.10
fcev_sog_target = 0.95  # 95%

# 价格和DR
fcev_service_price = 12.0  # $/kg
fcev_dr_flexibility = 0.05  # 5%
```

### 7.2 EV参数 (config.py)

```python
# 电池和充电
ev_battery_capacity_mean = 65.0  # kWh
ev_fast_charge_power = 150.0  # kW
ev_slow_charge_power = 11.0  # kW

# 到站分布
ev_soc_arrival_mean = 0.25  # 25%
ev_soc_target_fast = 0.80  # 80%

# 价格和DR
ev_service_price = 0.35  # $/kWh
ev_dr_flexibility_fast = 0.30  # 30%
ev_dr_flexibility_slow = 0.60  # 60%
```

### 7.3 需求生成参数

```python
# 到达率
base_vehicle_arrival_rate = 10.0  # v/h
ev_fcev_ratio = 0.70  # 70% EV, 30% FCEV

# 时段系数
peak_arrival_multiplier = 2.8
offpeak_multiplier = 0.3

# 设施容量
max_concurrent_ev_fast = 4
max_concurrent_ev_slow = 8
max_concurrent_fcev = 3
```

---

## 8. 验证与测试

### 8.1 单元测试

测试各组件独立功能：

```bash
python test_ev_fcev_components.py
```

预期输出：
- ✓ FCEV加氢时间: 3-5分钟
- ✓ EV充电功率: 正确计算
- ✓ 车辆到达: 泊松分布
- ✓ 队列管理: FIFO正常

### 8.2 集成测试

测试完整环境：

```bash
python test_integrated_env.py
```

观察指标：
- EV/FCEV服务数量
- 队列长度变化
- 收入统计
- 氢气消耗速率

### 8.3 预期结果

**一天运行 (96步)**:
- EV服务: 30-50辆
- FCEV服务: 15-25辆
- 总收入: $1,200-1,800
- 平均队列: <3辆
- 延迟车辆: <10%

---

## 9. 未来扩展方向

### 9.1 V2G (Vehicle-to-Grid)

允许EV向电网反向供电：
- 高电价时EV放电
- 需考虑电池损耗
- 额外收入来源

### 9.2 预测性调度

基于机器学习预测：
- 未来需求量
- 电价趋势
- 最优制氢时间

### 9.3 多站点协同

考虑多个充电加氢站网络：
- 车辆引导到负载低的站点
- 站点间氢气调配
- 区域优化

### 9.4 电池/氢气储能套利

利用价格差异：
- 低电价时充电/制氢
- 高电价时放电/用氢发电
- 日前/实时市场参与

---

## 10. 参考资料

### 10.1 技术标准

- **SAE J2601**: 氢燃料加注通信协议
- **IEC 61851**: EV充电系统标准
- **ISO 17268**: 氢气加注连接装置

### 10.2 车辆规格

- **Hyundai Nexo**: 6.3kg, 156L, 700bar
- **Toyota Mirai**: 5.6kg, 142L, 700bar
- **Tesla Model 3**: 75kWh, 250kW充电

### 10.3 研究论文

[60] SAE J2601标准详解
[61] EV充电需求响应方法
[62] 氢气加注站运营优化

---

## 总结

本次更新实现了完整的EV/FCEV混合需求建模，系统现在能够：

✅ **同时服务两种车辆类型**
✅ **考虑真实充能协议** (SAE J2601、DC快充)
✅ **模拟时变需求模式** (早晚高峰)
✅ **实现需求响应** (EV高DR, FCEV低DR)
✅ **多能源耦合** (电力+氢气)
✅ **经济优化** (多收入源+成本控制)

这使得项目更加贴近实际运营场景，为氢能充装站的智能调度和经济优化提供了完整的仿真平台。

---

**作者**: AI Assistant (Claude Sonnet 4.5)  
**版本**: v2.0 - EV/FCEV混合需求建模  
**最后更新**: 2026-01-13
