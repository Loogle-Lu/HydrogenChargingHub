import numpy as np
import pickle
import os
import random
from config import Config


class DataLoader:
    N_PRICE_DAYS = 365
    PRICE_NOISE_SCALE = 0.03

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
                self.data[key] = [0.5] * 96 if key == 'price' else [[0] * 96] * 10

        self.price_data = self.data["price"]
        self.pv_data = self.data["pv"]
        self.wind_data = self.data["wind"]

        # Multi-day price tiling with Gaussian noise (plot_data.py style)
        # Fixed seed ensures all DataLoader instances produce identical price_days
        base_price = np.asarray(self.price_data).flatten()[:96]
        _price_rng = np.random.RandomState(2024)
        self.price_days = np.array([
            np.clip(base_price + _price_rng.normal(0, self.PRICE_NOISE_SCALE, 96),
                    0.01, 2.0)
            for _ in range(self.N_PRICE_DAYS)
        ])
        self.current_price_day_idx = 0

        self.current_pv_day_idx = 0
        self.current_wind_day_idx = 0
        self.demand_start_idx = 0

        total_steps = 96 * 365
        t = np.linspace(0, 365 * 2 * np.pi, total_steps)
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
        self.current_price_day_idx = random.randint(0, len(self.price_days) - 1)
        if len(self.pv_data) > 0:
            self.current_pv_day_idx = random.randint(0, len(self.pv_data) - 1)
        if len(self.wind_data) > 0:
            self.current_wind_day_idx = random.randint(0, len(self.wind_data) - 1)
        if len(self.demand_data) > 96 * 5:
            self.demand_start_idx = random.randint(0, len(self.demand_data) - 96 * 5)

    def get_step_data(self, step_idx):
        time_idx = step_idx % 96

        price = float(self.price_days[self.current_price_day_idx % len(self.price_days)][time_idx])

        pv_day = self.pv_data[self.current_pv_day_idx % len(self.pv_data)]
        pv = pv_day[time_idx % len(pv_day)]

        wd_day = self.wind_data[self.current_wind_day_idx % len(self.wind_data)]
        wind = wd_day[time_idx % len(wd_day)]

        demand = self.demand_data[(self.demand_start_idx + step_idx) % len(self.demand_data)]

        return {
            "wind": max(0, float(wind)),
            "pv": max(0, float(pv)),
            "price": max(0, float(price)),
            "demand": max(0, float(demand))
        }
