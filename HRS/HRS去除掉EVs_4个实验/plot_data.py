import pickle
import os
import random
import matplotlib.pyplot as plt
import numpy as np

# 路径配置
try:
    from config import Config
    path_price = Config.path_price
    path_pv = Config.path_pv
    path_wind = Config.path_wind
except ImportError:
    # 方案二：若无法导入 config，则根据脚本位置自动推断项目根目录
    # 从 HRS/HRS去除掉EVs/plot_data.py 上溯三级到项目根目录 HydrogenChargingHub
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
    path_price = os.path.join(_project_root, "price_after_MAD_96.pkl")
    path_pv = os.path.join(_project_root, "pv_power_100.pkl")
    path_wind = os.path.join(_project_root, "wd_power_150.pkl")

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

try:
    price_data = load_pickle(path_price)
    pv_data = load_pickle(path_pv)
    wind_data = load_pickle(path_wind)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print(f"  请检查以下路径是否存在：")
    print(f"    price: {path_price}")
    print(f"    pv:    {path_pv}")
    print(f"    wind: {path_wind}")
    print("  若路径不对，可在 config.py 中修改 path_price / path_pv / path_wind。")
    exit()

# ========== 复刻历史版本 (With green power)PPO 3D-C.py 的数据处理逻辑 ==========
# 图四效果来自：多日平铺 + 电价高斯噪声 + 新能源自相关乘性噪声
DAYS = 5
STEPS_TOTAL = 96 * (DAYS + 1)  # 576 步，与历史脚本一致
np.random.seed(42)
random.seed(42)


def correlated_noise(n, rho=0.9):
    """自相关乘性噪声：模拟天气的持续性（云遮住后会持续几小时）"""
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + np.random.normal(0, 0.1)
    return np.exp(x)


# 1. 电价：96 步平铺 + 高斯噪声（历史脚本: full_price + N(0, 0.5), clip 0.1~100）
base_price = np.asarray(price_data).flatten()[:96]
full_price = np.tile(base_price, (DAYS + 1))[:STEPS_TOTAL]
# 噪声尺度按数据量级调整：原始 price 0.04~0.21，用 0.03 产生可见波动
noise_scale = 0.03
price_plot = np.clip(full_price + np.random.normal(0, noise_scale, STEPS_TOTAL), 0.01, 2.0)

# 2. 光伏：随机选天拼接 + 缩放 + 自相关噪声
raw_pv = []
for _ in range(DAYS + 1):
    row = pv_data[random.randint(0, len(pv_data) - 1)] if np.ndim(pv_data) > 1 else pv_data
    raw_pv.extend(np.asarray(row).flatten()[:96])
ac_pv_base = np.array(raw_pv[:STEPS_TOTAL])
pv_plot = ac_pv_base * correlated_noise(STEPS_TOTAL)

# 3. 风电：同上
raw_wd = []
for _ in range(DAYS + 1):
    row = wind_data[random.randint(0, len(wind_data) - 1)] if np.ndim(wind_data) > 1 else wind_data
    raw_wd.extend(np.asarray(row).flatten()[:96])
ac_wd_base = np.array(raw_wd[:STEPS_TOTAL])
wind_plot = ac_wd_base * correlated_noise(STEPS_TOTAL, rho=0.8)

x_axis = np.arange(STEPS_TOTAL)

plt.rcParams.update({'font.size': 12})
figsize = (10, 4)

# 1. 电价（紫色，与图四一致）
plt.figure(figsize=figsize)
plt.plot(x_axis, price_plot, color='purple', linewidth=1.5)
plt.title("Grid Electricity Price (From Data)")
plt.xlabel("Time Step (15 min)")
plt.ylabel("Price (p.u.)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, STEPS_TOTAL - 1)
plt.tight_layout()
plt.savefig("data_price.png", dpi=300)
plt.close()

# 2. 光伏
plt.figure(figsize=figsize)
plt.plot(x_axis, pv_plot, color='#ff7f0e', linewidth=1.5)
plt.title("PV Generation (From Data)")
plt.xlabel("Time Step (15 min)")
plt.ylabel("Power (kW)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, STEPS_TOTAL - 1)
plt.tight_layout()
plt.savefig("data_pv.png", dpi=300)
plt.close()

# 3. 风电
plt.figure(figsize=figsize)
plt.plot(x_axis, wind_plot, color='#2ca02c', linewidth=1.5)
plt.title("Wind Generation (From Data)")
plt.xlabel("Time Step (15 min)")
plt.ylabel("Power (kW)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, STEPS_TOTAL - 1)
plt.tight_layout()
plt.savefig("data_wind.png", dpi=300)
plt.close()

print("三张图片已生成（图四同款处理）：data_price.png, data_pv.png, data_wind.png")