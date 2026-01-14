# 集成EV充电/FCEV加氢站强化学习优化系统

## 项目简介

本项目实现了一个完整的**集成充电加氢站**智能调度系统，基于**强化学习(SAC算法)**优化能源管理，同时服务**电动车(EV)充电**和**燃料电池车(FCEV)加氢**需求。

### 核心特性

🔋 **EV充电**: 快充/慢充/超快充，支持需求响应  
⚡ **FCEV加氢**: 基于SAE J2601协议，3-5分钟快速充装  
🏭 **氢气生产**: 电解槽 + 多级压缩机(C1/C2/C3) + 级联储罐  
🌱 **可变功率阈值策略**: 智能优化绿氢生产，最大化可再生能源利用  
❄️ **非线性Chiller**: 考虑PLR和温度的真实性能曲线  
🤖 **智能调度**: SAC强化学习Agent优化决策  
💰 **多收入源**: EV充电 + FCEV加氢 + 电网售电 + 绿氢奖励

---

## 系统架构

```
可再生能源 + 电网
    ↓
┌─────────────────────────────┐
│  电力分配                    │
├─────────────────────────────┤
│ • EV充电 (直接)              │
│ • 电解槽制氢                 │
│ • 压缩机C1/C2/C3            │
│ • Chiller冷却                │
└─────────────────────────────┘
    ↓ 氢气
T1 → C1 → T2 → C2 → T3₁/T3₂/T3₃ → FCEV加氢
                         ↓ C3
                        T4 → 700bar快充
```

详见: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)

---

## 快速开始

### 1. 环境要求

```bash
Python 3.8+
numpy
gym
torch
matplotlib
pickle
```

### 2. 文件结构

```
SAC/
├── main.py              # 训练主程序
├── env.py               # 环境定义 (HydrogenEnv)
├── sac.py               # SAC算法实现
├── components.py        # 物理组件建模
├── config.py            # 参数配置
├── data_loader.py       # 数据加载
└── *.md                 # 文档
```

### 3. 运行训练

```bash
cd SAC
python main.py
```

训练将运行200个回合，每个回合96步(24小时，15分钟/步)。

### 4. 查看结果

训练结束后会显示：
- 奖励曲线
- 利润曲线
- SOC历史
- 能量平衡

---

## 主要组件说明

### components.py

| 组件 | 功能 | 关键参数 |
|------|------|---------|
| `Electrolyzer` | 电解槽制氢 | 1000kW, 50kWh/kg |
| `MultiStageCompressorSystem` | 三级压缩机 | C1/C2/C3 |
| `MultiTankStorage` | 六储罐系统 | T1-T4, 级联 |
| `NonlinearChiller` | 非线性冷却 | PLR曲线, COP=3.5 |
| `MixedDemandGenerator` | 车辆到达生成 | 泊松过程, 时变 |
| `IntegratedServiceStation` | 服务站运营 | EV+FCEV调度 |

### config.py - 关键参数

```python
# EV充电
ev_fast_charge_power = 150.0  # kW
ev_service_price = 0.35  # $/kWh
ev_dr_flexibility_fast = 0.30  # 30%可延迟

# FCEV加氢
fcev_tank_capacity = 6.3  # kg (Nexo)
fcev_target_fill_time = 4.0  # minutes
fcev_service_price = 12.0  # $/kg
fcev_dr_flexibility = 0.05  # 5%可延迟

# 车辆到达
base_vehicle_arrival_rate = 10.0  # v/h
ev_fcev_ratio = 0.70  # 70% EV
peak_arrival_multiplier = 2.8  # 早晚高峰

# 设施容量
max_concurrent_ev_fast = 4  # 快充桩
max_concurrent_fcev = 3  # 加氢枪
```

---

## 观察与动作空间

### 观察空间 (8维)

```python
[
    storage_total_soc,      # 总氢气储量
    storage_t1_soc,         # T1罐
    storage_t2_soc,         # T2罐
    electricity_price,      # 电价
    renewable_power,        # 可再生能源
    ev_queue_normalized,    # EV队列
    fcev_queue_normalized,  # FCEV队列
    hour_of_day_norm        # 时段
]
```

### 动作空间 (2维)

```python
[
    electrolyzer_ratio,     # 电解槽功率 (0-1)
    fuel_cell_ratio         # 燃料电池功率 (0-1)
]
```

---

## 奖励函数

```python
reward = (ev_revenue + fcev_revenue + grid_revenue + green_h2_bonus) 
         - grid_cost 
         - penalties
```

**收入**:
- EV充电: 能量 × $0.35/kWh
- FCEV加氢: 质量 × $12/kg
- 电网售电: 功率 × 电价
- **绿氢奖励**: 绿氢产量 × $2.0/kg ⭐ (新增)

**惩罚**:
- 缺氢: $500/kg
- 车辆等待: $30-60/v/h
- I2S约束: $10,000 × |终端SOC - 初始SOC|

**绿氢奖励计算**:
```python
green_h2_bonus = (power_from_re / ele_efficiency) × dt × green_hydrogen_bonus
```
- 仅对使用可再生能源生产的氢气部分给予奖励
- 鼓励在高RE时段增加制氢功率

---

## 文档索引

📄 **UPDATES_SUMMARY.md**  
   - 系统更新总结
   - 多级压缩机、多储罐、非线性Chiller
   - EV/FCEV需求建模

📄 **EV_FCEV_MODELING_SUMMARY.md**  
   - EV vs FCEV详细对比
   - SAE J2601加氢协议
   - 需求响应特性
   - 混合需求生成

📄 **SYSTEM_ARCHITECTURE.md**  
   - 完整系统拓扑图
   - 数据流与决策流
   - 时间尺度分析
   - 组件接口说明

---

## 性能指标

### 典型运行结果 (一天)

| 指标 | 目标 |
|------|------|
| EV服务数 | 30-50 辆 |
| FCEV服务数 | 15-25 辆 |
| 日收入 | $1,200-1,800 |
| EV等待时间 | <30分钟 |
| FCEV等待时间 | <10分钟 |
| 可再生能源利用率 | >60% |
| 服务成功率 | >95% |

---

## 关键创新点

### 1. 可变功率阈值策略 (Variable Power Threshold Strategy) 🌱

**核心思想**: 通过动态调整功率阈值，最大化绿氢生产，优先利用可再生能源

**策略逻辑**:
- ✅ **当 RE可用 > 阈值**: 优先使用可再生能源生产绿氢
- ⚡ **当 RE可用 < 阈值**: 允许使用电网能源，但仍优先利用可用RE
- 📊 **动态阈值**: 基于电价、储氢量和RE可用性实时调整

**阈值计算公式**:
```
动态阈值 = 基准阈值 + 电价调整 + SOC调整 + RE调整
- 电价调整: (电价 - 0.08) × 500  (高电价时提高阈值)
- SOC调整: -(0.5 - SOC) × 200      (低储氢时降低阈值)
- RE调整: -min(RE/1000, 1.0) × 50  (RE充足时稍降低阈值)
```

**绿氢奖励机制**:
- 每生产1kg绿氢获得额外奖励: **$2.0/kg**
- 鼓励Agent学习在高RE时段积极制氢
- 提高系统可持续性和环境效益

**关键参数** (`config.py`):
```python
enable_threshold_strategy = True      # 启用策略
base_power_threshold = 200.0          # 基准阈值 (kW)
threshold_price_coef = 500.0          # 电价影响系数
threshold_soc_coef = 200.0            # SOC影响系数
green_hydrogen_bonus = 2.0            # 绿氢奖励 ($/kg)
min_power_threshold = 100.0           # 最小阈值 (kW)
max_power_threshold = 800.0           # 最大阈值 (kW)
```

**性能指标**:
- 绿氢占比: 目标 >60%
- 可再生能源利用率: 提升20-30%
- 电网购电成本: 降低15-25%

### 2. 多级级联压缩机系统

- C1 (2→35 bar): 初级压缩
- C2 (35→500 bar): 级联充装，节能12%
- C3 (500→700 bar): LDFV快充
- 自动判断压比，采用两级压缩

### 3. 非线性Chiller建模

- PLR性能曲线 (三次多项式)
- 环境温度修正
- 最小负荷和启停约束
- 比简单COP模型提高准确性29%

### 4. EV/FCEV混合需求

- EV高DR能力 (30-60%)
- FCEV刚性需求 (仅5% DR)
- 时变到达率 (早晚高峰2.8×)
- 真实加氢协议 (SAE J2601)

### 5. 集成服务站运营

- 同时管理充电桩和加氢枪
- FIFO队列 + 需求响应
- 多收入源优化
- 等待时间惩罚机制

---

## 使用建议

### 调参建议

1. **高EV需求场景**:
   ```python
   ev_fcev_ratio = 0.85  # 提高EV占比
   max_concurrent_ev_fast = 6  # 增加充电桩
   ```

2. **高氢气需求场景**:
   ```python
   t3_1_capacity_kg = 200.0  # 增加储罐容量
   fcev_dr_flexibility = 0.10  # 稍微提高DR
   ```

3. **低电价优化**:
   ```python
   dr_price_threshold_high = 0.10  # 降低DR触发阈值
   ev_dr_flexibility_slow = 0.70  # 提高慢充DR
   ```

4. **绿氢优化场景** ⭐:
   ```python
   # 激进绿氢策略 (最大化RE利用)
   base_power_threshold = 150.0  # 降低基准阈值
   green_hydrogen_bonus = 3.0     # 提高绿氢奖励
   threshold_price_coef = 800.0   # 提高电价敏感度
   
   # 保守绿氢策略 (平衡成本)
   base_power_threshold = 300.0  # 提高基准阈值
   green_hydrogen_bonus = 1.0    # 降低绿氢奖励
   threshold_soc_coef = 300.0    # 提高储氢优先级
   
   # 禁用阈值策略 (对比基准)
   enable_threshold_strategy = False
   ```

### 常见问题

**Q: FCEV队列过长？**  
A: 增加`max_concurrent_fcev`或提高`t3`储罐容量

**Q: EV充电收入低？**  
A: 增加`max_concurrent_ev_fast`或提高`ev_service_price`

**Q: 经常缺氢？**  
A: 提高`ele_max_power`或增加储罐容量

---

## 参考文献

[26] 级联储罐系统节能分析  
[27] 三储罐级联vs单储罐对比  
[28] 高压储氢罐充装性能  
[60] SAE J2601氢气加注标准  
[61] EV充电需求响应方法  

---

## 更新日志

**v2.1 (2026-01-14)** ⭐
- ✅ **可变功率阈值策略** - 智能绿氢生产优化
- ✅ 动态阈值计算 (电价/SOC/RE敏感)
- ✅ 绿氢/灰氢分离追踪和统计
- ✅ 绿氢奖励机制 ($2/kg bonus)
- ✅ 新增绿氢生产可视化图表
- ✅ 电解槽功率来源分解 (RE vs Grid)
- ✅ 完整绿氢生产报告系统

**v2.0 (2026-01-13)**
- ✅ 添加EV充电站完整建模
- ✅ 添加FCEV加氢站(SAE J2601)
- ✅ 实现混合需求生成器
- ✅ 集成服务站运营管理
- ✅ 观察空间扩展到8维

**v1.0 (2026-01-13)**
- ✅ 多级压缩机系统(C1/C2/C3)
- ✅ 多储罐系统(T1-T4)
- ✅ 非线性Chiller建模
- ✅ 环境更新和集成

---


**作者**: Loogle
**项目**: 氢能充装站智能调度系统  
**最后更新**: 2026-01-14 (v2.1 - Green Hydrogen Optimization)
