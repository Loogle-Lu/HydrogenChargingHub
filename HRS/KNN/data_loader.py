import numpy as np
import pickle
import os
import random
from config import Config


class DataLoader:
    def __init__(self):
        files = {
            "price": Config.path_price,
            "pv": Config.path_pv,
            "wind": Config.path_wind
        }

        self.data = {}
        for key, filename in files.items():
            real_path = self._find_file(filename)
            if real_path:
                print(f"Loading {key} from: {real_path}")
                self.data[key] = self._load_pickle_safe(real_path)
            else:
                print(f"Warning: {filename} not found. Using dummy data for {key}.")
                # Dummy fallback
                self.data[key] = [0.5] * 96 if key == 'price' else [[0] * 96] * 10

        self.price_data = self.data["price"]
        self.pv_data = self.data["pv"]
        self.wind_data = self.data["wind"]

        self.current_pv_day_idx = 0
        self.current_wind_day_idx = 0
        self.demand_start_idx = 0
        self.demand_multiplier = 1.0  # v3.7: 需求倍率 (用于场景化测试)

        # 模拟需求数据 (因为没有提供真实需求)
        # 保持正弦波模拟，但确保量级适合 1000kW 的电解槽
        total_steps = 96 * 365
        t = np.linspace(0, 365 * 2 * np.pi, total_steps)
        # 基准 10kg/h，波动 5kg/h。电解槽满载约 20kg/h，足以覆盖。
        self.demand_data = np.abs(10 + 5 * np.sin(t) + np.random.normal(0, 2, total_steps))

    def _find_file(self, filename):
        if os.path.exists(filename): return filename
        basename = os.path.basename(filename)
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(curr_dir, basename),
            os.path.join(curr_dir, "..", basename),
            os.path.join(curr_dir, "data_file", basename),
        ]
        for p in candidates:
            if os.path.exists(p): return p
        return None

    def _load_pickle_safe(self, path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except:
            with open(path, 'rb') as f:
                return pickle.load(f, encoding='latin1')

    def reset(self):
        """随机重置日索引 (原有行为)"""
        if len(self.pv_data) > 0:
            self.current_pv_day_idx = random.randint(0, len(self.pv_data) - 1)
        if len(self.wind_data) > 0:
            self.current_wind_day_idx = random.randint(0, len(self.wind_data) - 1)
        if len(self.demand_data) > 96 * 5:
            self.demand_start_idx = random.randint(0, len(self.demand_data) - 96 * 5)
        self.demand_multiplier = 1.0  # 随机模式下恢复默认倍率

    def reset_with_scenario(self, pv_day_idx, wind_day_idx, demand_start_idx,
                            demand_multiplier=1.0):
        """
        v3.7: 使用指定的日索引重置 (场景化测试)。

        Parameters
        ----------
        pv_day_idx : int
            PV 数据天索引
        wind_day_idx : int
            Wind 数据天索引
        demand_start_idx : int
            需求数据起始索引
        demand_multiplier : float
            需求倍率 (1.0=正常, 2.0=双倍需求)
        """
        self.current_pv_day_idx = pv_day_idx % len(self.pv_data)
        self.current_wind_day_idx = wind_day_idx % len(self.wind_data)
        self.demand_start_idx = demand_start_idx % max(1, len(self.demand_data) - 96 * 5)
        self.demand_multiplier = demand_multiplier

    def get_step_data(self, step_idx):
        time_idx = step_idx % 96

        p_len = len(self.price_data) if hasattr(self.price_data, '__len__') else 1
        price = self.price_data[time_idx % p_len] if p_len > 1 else 0.5

        pv_day = self.pv_data[self.current_pv_day_idx % len(self.pv_data)]
        pv = pv_day[time_idx % len(pv_day)]

        wd_day = self.wind_data[self.current_wind_day_idx % len(self.wind_data)]
        wind = wd_day[time_idx % len(wd_day)]

        demand = self.demand_data[(self.demand_start_idx + step_idx) % len(self.demand_data)]
        demand *= self.demand_multiplier  # v3.7: 场景化需求倍率

        return {
            "wind": max(0, float(wind)),
            "pv": max(0, float(pv)),
            "price": max(0, float(price)),
            "demand": max(0, float(demand))
        }
