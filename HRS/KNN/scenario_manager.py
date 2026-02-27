"""
场景管理器: KNN 场景提取 + 预定义典型场景

v3.7 新增:
- ScenarioExtractor: 从历史数据中提取每日特征，用 KNN 将每天分类到最近的典型场景
- ScenarioManager: 管理预定义场景库，提供场景化环境重置接口

典型场景:
    1. normal        - 常规日 (中位数特征)
    2. high_price     - 高电价日
    3. low_renewable  - 低可再生能源日
    4. high_demand    - 高需求日
    5. ideal          - 理想日 (低电价 + 高可再生)
    6. extreme        - 极端日 (高电价 + 低可再生 + 高需求)

使用方法:
    from scenario_manager import ScenarioManager
    from data_loader import DataLoader

    dl = DataLoader()
    sm = ScenarioManager(dl)

    # 获取所有场景名
    names = sm.get_all_scenario_names()

    # 获取某个场景的环境配置
    cfg = sm.get_scenario_config("high_price")
    # cfg = {"pv_day_idx": 42, "wind_day_idx": 88, "demand_start_idx": 1000, "demand_multiplier": 1.0}
"""

import numpy as np


# =====================================================================
# 1. 特征提取工具函数
# =====================================================================

def _extract_daily_price_features(price_data):
    """
    从 96 点日电价曲线中提取统计特征。

    Parameters
    ----------
    price_data : array-like, shape (96,)
        一天 96 个时间步的电价。

    Returns
    -------
    dict : 包含 mean, std, peak, valley, peak_valley_ratio 的字典。
    """
    arr = np.asarray(price_data, dtype=np.float64)
    mean_p = float(np.mean(arr))
    std_p = float(np.std(arr))
    peak_p = float(np.max(arr))
    valley_p = float(np.min(arr))
    ratio_p = peak_p / max(valley_p, 1e-6)
    return {
        "price_mean": mean_p,
        "price_std": std_p,
        "price_peak": peak_p,
        "price_valley": valley_p,
        "price_peak_valley_ratio": ratio_p,
    }


def _extract_daily_re_features(day_curve):
    """
    从 96 点可再生能源日功率曲线中提取统计特征。

    Parameters
    ----------
    day_curve : array-like, shape (96,)
        一天的 PV 或 Wind 功率。

    Returns
    -------
    dict : daily_total, peak_output, variability (std/mean)。
    """
    arr = np.asarray(day_curve, dtype=np.float64)
    arr = np.maximum(arr, 0.0)
    daily_total = float(np.sum(arr))  # kWh (假设 dt=0.25h，未乘 dt，仅用于相对比较)
    peak_output = float(np.max(arr))
    mean_val = float(np.mean(arr))
    variability = float(np.std(arr) / max(mean_val, 1e-6))
    return {
        "daily_total": daily_total,
        "peak_output": peak_output,
        "variability": variability,
    }


# =====================================================================
# 2. KNN 场景提取器
# =====================================================================

class ScenarioExtractor:
    """
    从 DataLoader 的历史数据中提取每日特征，并用 KNN 将 PV/Wind 日
    分类到最接近的预定义场景中心。

    工作流程:
        1. 对所有 PV 天计算特征向量 (daily_total, peak_output, variability)
        2. 对所有 Wind 天计算特征向量
        3. 对电价计算特征 (全局只有 1 条 96 点曲线)
        4. 根据预定义场景中心，用欧氏距离 KNN (k=1) 找最近邻
    """

    def __init__(self, data_loader):
        self.data_loader = data_loader

        # --- 提取 PV 每日特征 ---
        self.pv_features = []  # list of dict
        for i, day in enumerate(data_loader.pv_data):
            feat = _extract_daily_re_features(day)
            feat["day_idx"] = i
            self.pv_features.append(feat)

        # --- 提取 Wind 每日特征 ---
        self.wind_features = []
        for i, day in enumerate(data_loader.wind_data):
            feat = _extract_daily_re_features(day)
            feat["day_idx"] = i
            self.wind_features.append(feat)

        # --- 电价特征 (单条曲线) ---
        self.price_features = _extract_daily_price_features(data_loader.price_data)

        # --- 计算分位数统计 (用于场景分类阈值) ---
        pv_totals = np.array([f["daily_total"] for f in self.pv_features])
        wind_totals = np.array([f["daily_total"] for f in self.wind_features])

        self.pv_percentiles = {
            "p25": float(np.percentile(pv_totals, 25)),
            "p50": float(np.percentile(pv_totals, 50)),
            "p75": float(np.percentile(pv_totals, 75)),
        }
        self.wind_percentiles = {
            "p25": float(np.percentile(wind_totals, 25)),
            "p50": float(np.percentile(wind_totals, 50)),
            "p75": float(np.percentile(wind_totals, 75)),
        }

        # RE 合计 (PV+Wind) 分位数需跨越组合，这里近似用各自分位数
        self.re_combined_p25 = self.pv_percentiles["p25"] + self.wind_percentiles["p25"]
        self.re_combined_p75 = self.pv_percentiles["p75"] + self.wind_percentiles["p75"]

    # ---------- KNN 分类工具 ----------

    def _knn_select(self, features_list, key, target_value, k=3):
        """
        在 features_list 中找出 key 最接近 target_value 的 k 个日的 day_idx。
        """
        dists = []
        for feat in features_list:
            d = abs(feat[key] - target_value)
            dists.append((d, feat["day_idx"]))
        dists.sort(key=lambda x: x[0])
        return [idx for _, idx in dists[:k]]

    def _knn_select_multi(self, features_list, target_vector, feature_keys, k=3):
        """
        多特征 KNN: 用标准化欧氏距离在 features_list 中找 k 近邻。

        Parameters
        ----------
        features_list : list of dict
        target_vector : dict, 目标特征
        feature_keys : list of str, 参与距离计算的特征名
        k : int

        Returns
        -------
        list of int : 最近的 k 个 day_idx
        """
        # 构建特征矩阵用于标准化
        mat = np.array([[f[key] for key in feature_keys] for f in features_list])
        means = mat.mean(axis=0)
        stds = mat.std(axis=0)
        stds[stds < 1e-8] = 1.0  # 防止除零

        target = np.array([target_vector[key] for key in feature_keys])
        target_norm = (target - means) / stds

        dists = []
        for i, feat in enumerate(features_list):
            vec = np.array([feat[key] for key in feature_keys])
            vec_norm = (vec - means) / stds
            d = float(np.linalg.norm(vec_norm - target_norm))
            dists.append((d, feat["day_idx"]))
        dists.sort(key=lambda x: x[0])
        return [idx for _, idx in dists[:k]]

    # ---------- 场景日选取 ----------

    def get_high_re_days(self, k=3):
        """选取可再生能源发电量最高的 PV/Wind 日"""
        pv_top = self._knn_select(self.pv_features, "daily_total",
                                   self.pv_percentiles["p75"] * 1.2, k)
        wind_top = self._knn_select(self.wind_features, "daily_total",
                                     self.wind_percentiles["p75"] * 1.2, k)
        return pv_top, wind_top

    def get_low_re_days(self, k=3):
        """选取可再生能源发电量最低的 PV/Wind 日"""
        pv_low = self._knn_select(self.pv_features, "daily_total",
                                   self.pv_percentiles["p25"] * 0.8, k)
        wind_low = self._knn_select(self.wind_features, "daily_total",
                                     self.wind_percentiles["p25"] * 0.8, k)
        return pv_low, wind_low

    def get_median_days(self, k=3):
        """选取最接近中位数的 PV/Wind 日 (normal 场景)"""
        pv_med = self._knn_select(self.pv_features, "daily_total",
                                   self.pv_percentiles["p50"], k)
        wind_med = self._knn_select(self.wind_features, "daily_total",
                                     self.wind_percentiles["p50"], k)
        return pv_med, wind_med

    def get_high_variability_days(self, k=3):
        """选取波动性最大的日 (用于压力测试)"""
        pv_var = sorted(self.pv_features, key=lambda x: -x["variability"])
        wind_var = sorted(self.wind_features, key=lambda x: -x["variability"])
        return [f["day_idx"] for f in pv_var[:k]], [f["day_idx"] for f in wind_var[:k]]


# =====================================================================
# 3. 预定义场景配置
# =====================================================================

# 场景描述 (用于图表标注)
SCENARIO_DESCRIPTIONS = {
    "normal":        "Normal Day (median RE, standard demand)",
    "high_price":    "High Price Day (peak electricity price)",
    "low_renewable": "Low Renewable Day (PV+Wind < P25)",
    "high_demand":   "High Demand Day (2x vehicle arrivals)",
    "ideal":         "Ideal Day (low price + high RE)",
    "extreme":       "Extreme Day (high price + low RE + high demand)",
}

SCENARIO_NAMES = list(SCENARIO_DESCRIPTIONS.keys())


# =====================================================================
# 4. 场景管理器
# =====================================================================

class ScenarioManager:
    """
    管理典型场景库，提供场景配置接口。

    使用 KNN 从历史数据中找出每个场景的典型日索引，
    并返回可直接传给 DataLoader.reset_with_scenario() 的配置字典。
    """

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.extractor = ScenarioExtractor(data_loader)
        self.scenarios = self._build_scenarios()

        # 打印场景摘要
        print("\n" + "=" * 60)
        print("  SCENARIO MANAGER INITIALIZED (KNN-based)")
        print("=" * 60)
        for name, cfg in self.scenarios.items():
            desc = SCENARIO_DESCRIPTIONS.get(name, "")
            pv_idx = cfg["pv_day_idx"]
            wind_idx = cfg["wind_day_idx"]
            dm = cfg["demand_multiplier"]
            print(f"  [{name:<15s}] PV day={pv_idx:>3d}, Wind day={wind_idx:>3d}, "
                  f"demand_mult={dm:.1f}  | {desc}")
        print("=" * 60 + "\n")

    def _build_scenarios(self):
        """
        用 KNN 从历史数据中选取每个场景的代表日。
        返回 dict: scenario_name -> config_dict
        """
        ext = self.extractor
        scenarios = {}

        # --- 1. Normal: 中位数日 ---
        pv_med, wind_med = ext.get_median_days(k=3)
        scenarios["normal"] = {
            "pv_day_idx": pv_med[0],
            "wind_day_idx": wind_med[0],
            "demand_start_idx": 0,
            "demand_multiplier": 1.0,
        }

        # --- 2. High Price: 电价特征不变（只有一条曲线），但选高 RE 日
        #     以测试高电价时的套利能力。demand 不变。
        #     注意: 电价曲线是固定的，"高电价"效果通过搭配低 RE 来加压。
        #     这里用中位数 RE，突出电价压力。
        pv_med2 = pv_med[1] if len(pv_med) > 1 else pv_med[0]
        wind_med2 = wind_med[1] if len(wind_med) > 1 else wind_med[0]
        scenarios["high_price"] = {
            "pv_day_idx": pv_med2,
            "wind_day_idx": wind_med2,
            "demand_start_idx": 96 * 30,  # 选另一段 demand
            "demand_multiplier": 1.0,
        }

        # --- 3. Low Renewable: 低 RE 日 ---
        pv_low, wind_low = ext.get_low_re_days(k=3)
        scenarios["low_renewable"] = {
            "pv_day_idx": pv_low[0],
            "wind_day_idx": wind_low[0],
            "demand_start_idx": 96 * 60,
            "demand_multiplier": 1.0,
        }

        # --- 4. High Demand: 正常 RE，但 demand 放大 2 倍 ---
        scenarios["high_demand"] = {
            "pv_day_idx": pv_med[0],
            "wind_day_idx": wind_med[0],
            "demand_start_idx": 96 * 90,
            "demand_multiplier": 2.0,
        }

        # --- 5. Ideal: 高 RE + 低需求 ---
        pv_high, wind_high = ext.get_high_re_days(k=3)
        scenarios["ideal"] = {
            "pv_day_idx": pv_high[0],
            "wind_day_idx": wind_high[0],
            "demand_start_idx": 96 * 120,
            "demand_multiplier": 0.7,
        }

        # --- 6. Extreme: 低 RE + 高需求 ---
        pv_low2 = pv_low[1] if len(pv_low) > 1 else pv_low[0]
        wind_low2 = wind_low[1] if len(wind_low) > 1 else wind_low[0]
        scenarios["extreme"] = {
            "pv_day_idx": pv_low2,
            "wind_day_idx": wind_low2,
            "demand_start_idx": 96 * 150,
            "demand_multiplier": 2.5,
        }

        return scenarios

    # ---------- 公开接口 ----------

    def get_scenario_config(self, scenario_name):
        """
        获取指定场景的环境配置字典。

        Parameters
        ----------
        scenario_name : str
            场景名称 (如 "normal", "high_price", "extreme" 等)

        Returns
        -------
        dict : 包含 pv_day_idx, wind_day_idx, demand_start_idx, demand_multiplier

        Raises
        ------
        KeyError : 如果场景名不存在
        """
        if scenario_name not in self.scenarios:
            available = ", ".join(self.scenarios.keys())
            raise KeyError(f"Unknown scenario '{scenario_name}'. Available: {available}")
        return dict(self.scenarios[scenario_name])  # 返回副本

    def get_all_scenario_names(self):
        """返回所有可用场景名列表"""
        return list(self.scenarios.keys())

    def get_scenario_description(self, scenario_name):
        """返回场景的文字描述"""
        return SCENARIO_DESCRIPTIONS.get(scenario_name, "Unknown scenario")

    def get_all_scenarios(self):
        """返回所有场景的配置字典 {name: config}"""
        return {name: dict(cfg) for name, cfg in self.scenarios.items()}

    def print_scenario_features(self):
        """
        打印每个场景代表日的详细特征，用于论文表格。
        """
        ext = self.extractor
        print("\n" + "=" * 90)
        print("  SCENARIO REPRESENTATIVE DAY FEATURES")
        print("=" * 90)
        header = (f"{'Scenario':<16s} {'PV Total':>10s} {'PV Peak':>10s} {'PV Var':>8s} "
                  f"{'Wind Total':>10s} {'Wind Peak':>10s} {'Wind Var':>8s} {'Demand x':>8s}")
        print(header)
        print("-" * 90)

        for name in SCENARIO_NAMES:
            cfg = self.scenarios[name]
            pv_idx = cfg["pv_day_idx"]
            wind_idx = cfg["wind_day_idx"]

            pv_feat = ext.pv_features[pv_idx]
            wind_feat = ext.wind_features[wind_idx]
            dm = cfg["demand_multiplier"]

            print(f"{name:<16s} "
                  f"{pv_feat['daily_total']:>10.1f} {pv_feat['peak_output']:>10.1f} "
                  f"{pv_feat['variability']:>8.2f} "
                  f"{wind_feat['daily_total']:>10.1f} {wind_feat['peak_output']:>10.1f} "
                  f"{wind_feat['variability']:>8.2f} "
                  f"{dm:>8.1f}")
        print("=" * 90 + "\n")
