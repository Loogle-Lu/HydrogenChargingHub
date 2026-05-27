import numpy as np
import pickle
import os
import random
from config import Config

# 与 plot_data.py 历史逻辑一致（电价加性高斯；PV/WD 随机日拼接 × 自相关乘性噪声）
PRICE_NOISE_SCALE = 0.03
CORR_NOISE_STD = 0.1
PV_CORR_RHO = 0.9
WIND_CORR_RHO = 0.8


def correlated_noise(n, rho=0.9, rng=None):
    """自相关乘性噪声：x_t = rho*x_{t-1} + N(0,0.1)，因子为 exp(x)。"""
    if rng is None:
        rng = np.random.default_rng()
    x = np.zeros(int(n))
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal(0.0, CORR_NOISE_STD)
    return np.exp(x)


def synthesize_series(price_raw, pv_raw, wind_raw, total_steps, rng):
    """
    按 plot_data 同款规则生成对齐的电价 / PV / WD 序列（长度 total_steps）。
    - 电价：96 点日曲线平铺 + 独立高斯噪声 + clip
    - PV：随机抽日拼接至总长 × correlated_noise(rho=0.9)
    - 风：随机抽日拼接 × correlated_noise(rho=0.8)
    """
    n = int(total_steps)
    if n <= 0:
        return np.array([]), np.array([]), np.array([])

    base_price = np.asarray(price_raw).flatten()[: Config.steps_per_day]
    if base_price.size < Config.steps_per_day:
        base_price = np.pad(
            base_price,
            (0, Config.steps_per_day - base_price.size),
            constant_values=0.1,
        )
    reps = (n + Config.steps_per_day - 1) // Config.steps_per_day
    full_price = np.tile(base_price, reps)[:n]
    price = np.clip(
        full_price + rng.normal(0.0, PRICE_NOISE_SCALE, n),
        0.01,
        2.0,
    )

    def _concat_random_days(arr, n_steps):
        out = []
        while len(out) < n_steps:
            if np.ndim(arr) > 1 and len(arr) > 0:
                row = arr[int(rng.integers(0, len(arr)))]
            else:
                row = arr
            out.extend(np.asarray(row).flatten()[: Config.steps_per_day])
        return np.array(out[:n_steps], dtype=np.float64)

    ac_pv_base = _concat_random_days(pv_raw, n)
    pv = ac_pv_base * correlated_noise(n, PV_CORR_RHO, rng)

    ac_wd_base = _concat_random_days(wind_raw, n)
    wind = ac_wd_base * correlated_noise(n, WIND_CORR_RHO, rng)

    return price, np.maximum(pv, 0.0), np.maximum(wind, 0.0)


class DataLoader:
    def __init__(self):
        files = {
            "price": Config.path_price,
            "pv": Config.path_pv,
            "wind": Config.path_wind,
        }

        self.data = {}
        for key, filename in files.items():
            real_path = self._find_file(filename)
            if real_path:
                print(f"Loading {key} from: {real_path}")
                self.data[key] = self._load_pickle_safe(real_path)
            else:
                print(f"Warning: {filename} not found. Using dummy data for {key}.")
                self.data[key] = (
                    [0.5] * Config.steps_per_day
                    if key == "price"
                    else [[0] * Config.steps_per_day] * 10
                )

        self.price_data = self.data["price"]
        self.pv_data = self.data["pv"]
        self.wind_data = self.data["wind"]

        self.demand_start_idx = 0
        self._synth_offset = 0
        self._rng = np.random.default_rng()

        # 与合成序列等长的需求模拟（原逻辑保留）
        total_steps = Config.steps_per_day * 365
        t = np.linspace(0, 365 * 2 * np.pi, total_steps)
        self.demand_data = np.abs(
            10 + 5 * np.sin(t) + np.random.normal(0, 2, total_steps)
        )

        self._synth_len = total_steps
        self.price_series, self.pv_series, self.wind_series = synthesize_series(
            self.price_data,
            self.pv_data,
            self.wind_data,
            self._synth_len,
            self._rng,
        )

    def _find_file(self, filename):
        if os.path.exists(filename):
            return filename
        basename = os.path.basename(filename)
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(curr_dir, basename),
            os.path.join(curr_dir, "..", basename),
            os.path.join(curr_dir, "data_file", basename),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def _load_pickle_safe(self, path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            with open(path, "rb") as f:
                return pickle.load(f, encoding="latin1")

    def reset(self, episode_index=None):
        """新 episode：电价/风光窗口。
        episode_index 非空时按索引确定窗口 → 同序号 episode 各算法面对相同外生情景，曲线更可比、震荡更小。
        为 None 时保持原随机滑动（兼容外部脚本）。"""
        span = Config.steps_per_day * 5
        if episode_index is not None:
            ei = int(episode_index)
            if len(self.demand_data) > span:
                hi = len(self.demand_data) - span - 1
                self.demand_start_idx = (ei * 7919) % (hi + 1)
            else:
                self.demand_start_idx = 0
            if self._synth_len >= Config.steps_per_day:
                hi2 = self._synth_len - Config.steps_per_day
                self._synth_offset = (ei * 7937) % (hi2 + 1)
            else:
                self._synth_offset = 0
            return

        if len(self.demand_data) > span:
            self.demand_start_idx = random.randint(0, len(self.demand_data) - span)
        else:
            self.demand_start_idx = 0

        if self._synth_len >= Config.steps_per_day:
            self._synth_offset = random.randint(
                0, self._synth_len - Config.steps_per_day
            )
        else:
            self._synth_offset = 0

    def get_step_data(self, step_idx):
        idx = (self._synth_offset + int(step_idx)) % self._synth_len

        price = float(self.price_series[idx])
        pv = float(self.pv_series[idx])
        wind = float(self.wind_series[idx])

        demand = self.demand_data[
            (self.demand_start_idx + step_idx) % len(self.demand_data)
        ]

        return {
            "wind": max(0.0, wind),
            "pv": max(0.0, pv),
            "price": max(0.0, price),
            "demand": max(0.0, float(demand)),
        }
