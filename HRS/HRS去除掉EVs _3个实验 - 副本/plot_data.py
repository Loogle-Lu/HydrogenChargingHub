import os
import matplotlib.pyplot as plt
import numpy as np

from data_loader import synthesize_series

# 路径配置
try:
    from config import Config

    path_price = Config.path_price
    path_pv = Config.path_pv
    path_wind = Config.path_wind
except ImportError:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
    path_price = os.path.join(_project_root, "price_after_MAD_96.pkl")
    path_pv = os.path.join(_project_root, "pv_power_100.pkl")
    path_wind = os.path.join(_project_root, "wd_power_150.pkl")


def load_pickle(path):
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


try:
    price_data = load_pickle(path_price)
    pv_data = load_pickle(path_pv)
    wind_data = load_pickle(path_wind)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("  请检查以下路径是否存在：")
    print(f"    price: {path_price}")
    print(f"    pv:    {path_pv}")
    print(f"    wind: {path_wind}")
    print("  若路径不对，可在 config.py 中修改 path_price / path_pv / path_wind。")
    raise SystemExit(1) from e

# 与历史脚本一致：多日长度；合成逻辑与 data_loader / 训练环境统一
DAYS = 5
try:
    STEPS_TOTAL = Config.steps_per_day * (DAYS + 1)
except NameError:
    STEPS_TOTAL = 96 * (DAYS + 1)
rng = np.random.default_rng(42)

price_plot, pv_plot, wind_plot = synthesize_series(
    price_data, pv_data, wind_data, STEPS_TOTAL, rng
)

x_axis = np.arange(STEPS_TOTAL)

plt.rcParams.update({"font.size": 12})
figsize = (10, 4)

plt.figure(figsize=figsize)
plt.plot(x_axis, price_plot, color="purple", linewidth=1.5)
plt.title("Grid Electricity Price (synthetic, same as DataLoader)")
plt.xlabel("Time Step (15 min)")
plt.ylabel("Price (p.u.)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.xlim(0, STEPS_TOTAL - 1)
plt.tight_layout()
plt.savefig("data_price.png", dpi=300)
plt.close()

plt.figure(figsize=figsize)
plt.plot(x_axis, pv_plot, color="#ff7f0e", linewidth=1.5)
plt.title("PV Generation (synthetic, same as DataLoader)")
plt.xlabel("Time Step (15 min)")
plt.ylabel("Power (kW)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.xlim(0, STEPS_TOTAL - 1)
plt.tight_layout()
plt.savefig("data_pv.png", dpi=300)
plt.close()

plt.figure(figsize=figsize)
plt.plot(x_axis, wind_plot, color="#2ca02c", linewidth=1.5)
plt.title("Wind Generation (synthetic, same as DataLoader)")
plt.xlabel("Time Step (15 min)")
plt.ylabel("Power (kW)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.xlim(0, STEPS_TOTAL - 1)
plt.tight_layout()
plt.savefig("data_wind.png", dpi=300)
plt.close()

print("三张图片已生成（与 data_loader 训练数据合成规则一致）：")
print("  data_price.png, data_pv.png, data_wind.png")
