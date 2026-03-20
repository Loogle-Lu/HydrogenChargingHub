"""
实验4: 多场景鲁棒性验证 (3×3×3 Scenario-Based Robustness Validation)

使用 K-Medoids 聚类从三个维度独立提取典型运行场景:
  1. Electricity Price (365 天, 高斯噪声 tiling) → K-Medoids K=3 → Low / Med / High
  2. Renewable Energy  (PV×Wind 历史组合)       → K-Medoids K=3 → Low / Med / High
  3. FCEV Demand       (到达率蒙特卡洛模拟)     → K-Medoids K=3 → Low / Med / High

3×3×3 = 27 个典型场景, 每场景重新训练 4 种系统配置:
  Proposed   = SAC + VSD + Bypass + Cooling + APC + 3-Stage CC
  w/o RL     = Random Policy + 4T + 3-Stage CC
  w/o 4T     = SAC + 3-Stage Naive
  w/o CC     = SAC + 1-Stage Naive

输出:
  Figure_4.0a_clustering_price.png   — 电价 K-Medoids 聚类
  Figure_4.0b_clustering_RE.png      — 可再生能源 K-Medoids 聚类
  Figure_4.0c_clustering_demand.png  — FCEV 需求 K-Medoids 聚类
  Figure_4.1a_grid_LowPrice.png      — Low Price 下 3×3 Grid (RE × Demand)
  Figure_4.1b_grid_MedPrice.png      — Med Price 下 3×3 Grid
  Figure_4.1c_grid_HighPrice.png     — High Price 下 3×3 Grid
  Figure_4.2_scenario_heatmap.png    — 4 方案 × 27 场景 性能热力图

使用方法:
    python exp4_scenario_robustness.py
"""

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from config import Config
from env import HydrogenEnv
from exp3_cascade_ablation import NaiveArchEnv
from SAC import SAC, ReplayBuffer
from data_loader import DataLoader


# ======================== 配置 ========================
NUM_EPISODES = 200
NUM_RANDOM_EPISODES = 50
WARMUP_STEPS = 500
BATCH_SIZE = 256
LR = 3e-4
MA_WINDOW = 20

N_CLUSTERS = 3
N_RE_SAMPLES = 2000
N_DEMAND_SAMPLES = 1000
DEMAND_RATE_RANGE = (4.0, 18.0)

PRICE_LABELS = ["Low Price", "Med Price", "High Price"]
RE_LABELS = ["Low RE", "Med RE", "High RE"]
DEMAND_LABELS = ["Low Demand", "Med Demand", "High Demand"]

CLUSTER_COLORS = ["#d62728", "#2ca02c", "#1f77b4"]

VARIANT_NAMES = ["Proposed", "w/o RL", "w/o 4T", "w/o CC"]
VARIANT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

_CFG_KEYS = [
    "enable_vsd", "enable_dynamic_cooling", "enable_bypass",
    "enable_adaptive_pressure", "enable_arbitrage_bonus",
    "base_vehicle_arrival_rate",
]


# ======================== 工具函数 ========================

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_cfg():
    return {k: getattr(Config, k) for k in _CFG_KEYS}


def _restore_cfg(saved):
    for k, v in saved.items():
        setattr(Config, k, v)


def _last_n_mean(arr, n=20):
    return float(np.mean(arr[-n:])) if len(arr) >= n else float(np.mean(arr))


# ======================== K-Medoids ========================

def _euclidean_cdist(A, B):
    A2 = (A ** 2).sum(axis=1, keepdims=True)
    B2 = (B ** 2).sum(axis=1, keepdims=True)
    return np.sqrt(np.maximum(A2 - 2 * A @ B.T + B2.T, 0.0))


def k_medoids(X, k, max_iter=100, random_state=42):
    rng = np.random.RandomState(random_state)
    n = len(X)
    medoids = rng.choice(n, k, replace=False)

    for _ in range(max_iter):
        D = _euclidean_cdist(X, X[medoids])
        labels = np.argmin(D, axis=1)

        new_medoids = np.empty(k, dtype=int)
        for j in range(k):
            members = np.where(labels == j)[0]
            if len(members) == 0:
                new_medoids[j] = medoids[j]
                continue
            intra = _euclidean_cdist(X[members], X[members]).sum(axis=1)
            new_medoids[j] = members[np.argmin(intra)]

        if np.array_equal(np.sort(new_medoids), np.sort(medoids)):
            break
        medoids = new_medoids

    D = _euclidean_cdist(X, X[medoids])
    labels = np.argmin(D, axis=1)
    return medoids, labels


# ======================== 场景提取: 电价 ========================

def extract_price_scenarios():
    """
    从 DataLoader 的 365 天噪声电价中提取 3 个典型电价场景。
    特征: (日均电价, 峰谷价差)
    """
    loader = DataLoader()
    price_days = loader.price_days

    features = np.column_stack([
        price_days.mean(axis=1),
        price_days.max(axis=1) - price_days.min(axis=1),
    ])

    mean, std = features.mean(0), features.std(0) + 1e-8
    features_norm = (features - mean) / std

    medoids, labels = k_medoids(features_norm, N_CLUSTERS)

    sort_key = features[medoids, 0]
    order = np.argsort(sort_key)

    scenarios = []
    reordered = np.full_like(labels, -1)

    print("\n  K-Medoids Price Scenario Extraction (K=3):")
    print(f"  {'Label':<12} {'Day Idx':>8} {'Mean $/kWh':>11} {'Spread':>8} {'Weight':>8}")
    print("  " + "-" * 50)

    for rank, oidx in enumerate(order):
        mi = medoids[oidx]
        mask = labels == oidx
        reordered[mask] = rank
        w = mask.sum() / len(labels)
        m_price = features[mi, 0]
        spread = features[mi, 1]
        scenarios.append(dict(
            label=PRICE_LABELS[rank], price_day_idx=int(mi),
            mean_price=float(m_price), spread=float(spread), weight=float(w),
        ))
        print(f"  {PRICE_LABELS[rank]:<12} {mi:>8d} {m_price:>11.4f} {spread:>8.4f} {w:>7.1%}")

    return scenarios, features, reordered


# ======================== 场景提取: 可再生能源 ========================

def extract_re_scenarios():
    """
    从 PV(100天) × Wind(150天) 中采样组合, K-Medoids 提取 3 个典型 RE 场景。
    特征: (日总 PV kWh, 日总 Wind kWh)
    """
    loader = DataLoader()
    pv_data = np.array(loader.pv_data)
    wind_data = np.array(loader.wind_data)

    pv_totals = pv_data.sum(axis=1) * Config.dt
    wind_totals = wind_data.sum(axis=1) * Config.dt

    n_pv, n_wind = len(pv_totals), len(wind_totals)
    rng = np.random.RandomState(42)
    n_samples = min(N_RE_SAMPLES, n_pv * n_wind)
    pv_idx_arr = rng.randint(0, n_pv, n_samples)
    wind_idx_arr = rng.randint(0, n_wind, n_samples)

    features = np.column_stack([pv_totals[pv_idx_arr], wind_totals[wind_idx_arr]])

    mean, std = features.mean(0), features.std(0) + 1e-8
    features_norm = (features - mean) / std

    medoids, labels = k_medoids(features_norm, N_CLUSTERS)

    total_re = features[medoids].sum(axis=1)
    order = np.argsort(total_re)

    scenarios = []
    reordered = np.full_like(labels, -1)

    print("\n  K-Medoids RE Scenario Extraction (K=3):")
    print(f"  {'Label':<10} {'PV Day':>7} {'Wind Day':>9} {'PV kWh':>9} {'Wind kWh':>10} {'Weight':>8}")
    print("  " + "-" * 56)

    for rank, oidx in enumerate(order):
        si = medoids[oidx]
        pi, wi = int(pv_idx_arr[si]), int(wind_idx_arr[si])
        mask = labels == oidx
        reordered[mask] = rank
        w = mask.sum() / n_samples
        pv_kwh = pv_totals[pi]
        wind_kwh = wind_totals[wi]

        scenarios.append(dict(
            label=RE_LABELS[rank], pv_idx=pi, wind_idx=wi,
            pv_kwh=float(pv_kwh), wind_kwh=float(wind_kwh), weight=float(w),
        ))
        print(f"  {RE_LABELS[rank]:<10} {pi:>7d} {wi:>9d} {pv_kwh:>9.0f} {wind_kwh:>10.0f} {w:>7.1%}")

    return scenarios, features, reordered


# ======================== 场景提取: FCEV 需求 ========================

def _simulate_daily_demand(arrival_rate, rng):
    """蒙特卡洛模拟单日 FCEV 到达, 复用 FCEVDemandGenerator 的时段系数。"""
    total = 0
    peak = 0
    for step in range(96):
        hour = int(step * Config.dt) % 24
        if hour in Config.peak_morning_hours:
            mult = Config.peak_arrival_multiplier * 1.2
        elif hour in Config.peak_evening_hours:
            mult = Config.peak_arrival_multiplier
        elif hour in Config.midday_hours:
            mult = Config.midday_multiplier
        elif hour in Config.offpeak_hours:
            mult = Config.offpeak_multiplier
        else:
            mult = 1.0
        mult *= rng.uniform(0.6, 1.4)
        mean_arr = arrival_rate * mult * Config.dt
        n = min(int(rng.poisson(mean_arr)), 15)
        total += n
        peak = max(peak, n)
    return total, peak


def extract_demand_scenarios():
    """
    蒙特卡洛模拟 1000 天 (到达率 ~ Uniform[4, 18]), K-Medoids 提取 3 个典型需求场景。
    特征: (日总到达量, 峰值步到达量)
    """
    rng = np.random.RandomState(42)
    arrival_rates = rng.uniform(*DEMAND_RATE_RANGE, N_DEMAND_SAMPLES)

    features_list = []
    for rate in arrival_rates:
        total, peak = _simulate_daily_demand(rate, rng)
        features_list.append([total, peak])

    features = np.array(features_list, dtype=float)
    mean, std = features.mean(0), features.std(0) + 1e-8
    features_norm = (features - mean) / std

    medoids, labels = k_medoids(features_norm, N_CLUSTERS)

    sort_key = features[medoids, 0]
    order = np.argsort(sort_key)

    scenarios = []
    reordered = np.full_like(labels, -1)

    print("\n  K-Medoids Demand Scenario Extraction (K=3):")
    print(f"  {'Label':<14} {'Rate':>6} {'Scale':>7} {'Total Arr.':>11} {'Peak':>6} {'Weight':>8}")
    print("  " + "-" * 56)

    for rank, oidx in enumerate(order):
        mi = medoids[oidx]
        mask = labels == oidx
        reordered[mask] = rank
        w = mask.sum() / N_DEMAND_SAMPLES
        rate = arrival_rates[mi]
        scale = rate / Config.base_vehicle_arrival_rate

        scenarios.append(dict(
            label=DEMAND_LABELS[rank], arrival_rate=float(rate),
            demand_scale=float(scale),
            total_arrivals=float(features[mi, 0]),
            peak_arrivals=float(features[mi, 1]), weight=float(w),
        ))
        print(f"  {DEMAND_LABELS[rank]:<14} {rate:>6.1f} {scale:>7.2f}× "
              f"{features[mi, 0]:>10.0f} {features[mi, 1]:>6.0f} {w:>7.1%}")

    return scenarios, features, reordered


# ======================== 聚类可视化 (三张独立图) ========================

def _plot_cluster(features, labels, scenarios, xlabel, ylabel,
                  label_key_x, label_key_y, dim_labels, title, savepath):
    """通用聚类散点图: ★ = Medoid。"""
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)

    for k in range(N_CLUSTERS):
        mask = labels == k
        ax.scatter(features[mask, 0], features[mask, 1],
                   c=CLUSTER_COLORS[k], alpha=0.15, s=14,
                   label=f"Cluster → {dim_labels[k]} (n={mask.sum()})")

    for k, sc in enumerate(scenarios):
        ax.scatter(sc[label_key_x], sc[label_key_y],
                   c=CLUSTER_COLORS[k], s=260, marker="*",
                   edgecolors="black", linewidth=1.4, zorder=5)
        ax.annotate(sc["label"], (sc[label_key_x], sc[label_key_y]),
                    textcoords="offset points", xytext=(10, 8),
                    fontsize=9, fontweight="bold", color=CLUSTER_COLORS[k])

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {savepath}")


def plot_clustering_price(features, labels, scenarios):
    _plot_cluster(
        features, labels, scenarios,
        xlabel="Mean Daily Price ($/kWh)", ylabel="Daily Price Spread ($/kWh)",
        label_key_x="mean_price", label_key_y="spread",
        dim_labels=PRICE_LABELS,
        title="K-Medoids: Electricity Price Scenarios (K=3)\n★ = Medoid",
        savepath="Figure_4.0a_clustering_price.png",
    )


def plot_clustering_re(features, labels, scenarios):
    _plot_cluster(
        features, labels, scenarios,
        xlabel="Daily PV Generation (kWh)", ylabel="Daily Wind Generation (kWh)",
        label_key_x="pv_kwh", label_key_y="wind_kwh",
        dim_labels=RE_LABELS,
        title="K-Medoids: Renewable Energy Scenarios (K=3)\n★ = Medoid",
        savepath="Figure_4.0b_clustering_RE.png",
    )


def plot_clustering_demand(features, labels, scenarios):
    _plot_cluster(
        features, labels, scenarios,
        xlabel="Total Daily FCEV Arrivals", ylabel="Peak Step Arrivals",
        label_key_x="total_arrivals", label_key_y="peak_arrivals",
        dim_labels=DEMAND_LABELS,
        title="K-Medoids: FCEV Demand Scenarios (K=3)\n★ = Medoid",
        savepath="Figure_4.0c_clustering_demand.png",
    )


# ======================== 场景环境子类 ========================

class ScenarioEnv(HydrogenEnv):
    """固定 Price/PV/Wind 日 + 缩放 FCEV 到达率的环境。"""

    def __init__(self, price_day_idx, pv_idx, wind_idx, demand_scale=1.0, **kw):
        self._fixed_price = price_day_idx
        self._fixed_pv = pv_idx
        self._fixed_wind = wind_idx
        self._demand_scale = demand_scale
        super().__init__(**kw)

    def reset(self):
        state = super().reset()
        self.data_loader.current_price_day_idx = self._fixed_price
        self.data_loader.current_pv_day_idx = self._fixed_pv
        self.data_loader.current_wind_day_idx = self._fixed_wind
        self.current_data = self.data_loader.get_step_data(0)
        re = self.current_data["wind"] + self.current_data["pv"]
        self.state[0] = self.current_data["price"] / Config.price_max
        self.state[1] = re / Config.ele_max_power
        return self.state

    def step(self, action):
        orig = Config.base_vehicle_arrival_rate
        Config.base_vehicle_arrival_rate = orig * self._demand_scale
        try:
            out = super().step(action)
        finally:
            Config.base_vehicle_arrival_rate = orig
        return out


class ScenarioNaiveEnv(NaiveArchEnv):
    """固定场景的 NaiveArchEnv。"""

    def __init__(self, arch, price_day_idx, pv_idx, wind_idx, demand_scale=1.0, **kw):
        self._fixed_price = price_day_idx
        self._fixed_pv = pv_idx
        self._fixed_wind = wind_idx
        self._demand_scale = demand_scale
        super().__init__(arch, **kw)

    def reset(self):
        state = super().reset()
        self.data_loader.current_price_day_idx = self._fixed_price
        self.data_loader.current_pv_day_idx = self._fixed_pv
        self.data_loader.current_wind_day_idx = self._fixed_wind
        self.current_data = self.data_loader.get_step_data(0)
        re = self.current_data["wind"] + self.current_data["pv"]
        self.state[0] = self.current_data["price"] / Config.price_max
        self.state[1] = re / Config.ele_max_power
        return self.state

    def step(self, action):
        orig = Config.base_vehicle_arrival_rate
        Config.base_vehicle_arrival_rate = orig * self._demand_scale
        try:
            out = super().step(action)
        finally:
            Config.base_vehicle_arrival_rate = orig
        return out


# ======================== 训练 / 评估 ========================

def train_sac(env_factory, num_episodes=NUM_EPISODES):
    env = env_factory()
    sd = env.observation_space.shape[0]
    ad = env.action_space.shape[0]
    agent = SAC(sd, ad, lr=LR)
    buf = ReplayBuffer(capacity=100_000)

    rewards, profits = [], []
    total_steps = 0

    for ep in range(num_episodes):
        s = env.reset()
        ep_r, ep_p = 0.0, 0.0
        done = False
        while not done:
            if total_steps < WARMUP_STEPS:
                a = env.action_space.sample()
            else:
                a = agent.select_action(s, evaluate=False)
            ns, r, done, info = env.step(a)
            buf.push(s, a, r, ns, float(done))
            if total_steps >= WARMUP_STEPS and len(buf) >= BATCH_SIZE:
                agent.update(buf, BATCH_SIZE)
            s = ns
            ep_r += r
            ep_p += info.get("profit", 0.0)
            total_steps += 1
        rewards.append(ep_r)
        profits.append(ep_p)

    del env
    return np.array(rewards), np.array(profits)


def eval_random(env_factory, num_episodes=NUM_RANDOM_EPISODES):
    env = env_factory()
    rewards, profits = [], []
    for _ in range(num_episodes):
        s = env.reset()
        ep_r, ep_p = 0.0, 0.0
        done = False
        while not done:
            s, r, done, info = env.step(env.action_space.sample())
            ep_r += r
            ep_p += info.get("profit", 0.0)
        rewards.append(ep_r)
        profits.append(ep_p)
    del env
    return np.array(rewards), np.array(profits)


# ======================== 可视化: 3×3 Grid (每个 Price Level 一张) ========================

def plot_grids(all_results):
    """生成 3 张 3×3 Grid (一张 / Price Level), 共享 y 轴范围。"""
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 8

    all_vals = []
    for key, res in all_results.items():
        for v in VARIANT_NAMES:
            all_vals.append(_last_n_mean(res[v][1]))
    y_lo = min(all_vals) * 1.15 if min(all_vals) < 0 else min(all_vals) * 0.70
    y_hi = max(all_vals) * 1.20

    for p_idx, p_label in enumerate(PRICE_LABELS):
        fig, axs = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
        fig.suptitle(
            f"Exp 4: Scenario Robustness under {p_label} — Daily Profit ($)\n"
            f"Rows: RE Level (Low → High)  |  Cols: Demand Level (Low → High)",
            fontsize=11, fontweight="bold",
        )

        for r_idx, re_l in enumerate(RE_LABELS):
            for d_idx, d_l in enumerate(DEMAND_LABELS):
                ax = axs[r_idx, d_idx]
                key = (p_label, re_l, d_l)
                res = all_results[key]
                vals = [_last_n_mean(res[v][1]) for v in VARIANT_NAMES]

                x = np.arange(len(VARIANT_NAMES))
                bars = ax.bar(x, vals, color=VARIANT_COLORS,
                              edgecolor="gray", linewidth=0.5, width=0.60)
                ax.set_xticks(x)
                ax.set_xticklabels(VARIANT_NAMES, rotation=30, ha="right", fontsize=6.5)
                ax.set_title(f"{re_l} × {d_l}", fontsize=8, fontweight="bold")
                ax.set_ylim(y_lo, y_hi)
                ax.grid(True, axis="y", alpha=0.3, linestyle="--")
                if d_idx == 0:
                    ax.set_ylabel("Avg Profit ($)", fontsize=8)

                for b, v in zip(bars, vals):
                    yoff = abs(y_hi - y_lo) * 0.012
                    ax.text(b.get_x() + b.get_width() / 2,
                            b.get_height() + (yoff if v >= 0 else -yoff),
                            f"${v:,.0f}", ha="center",
                            va="bottom" if v >= 0 else "top", fontsize=5.5)

        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in VARIANT_COLORS]
        fig.legend(handles, VARIANT_NAMES, loc="lower center", ncol=4,
                   fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.01))

        suffix = p_label.replace(" ", "")
        path = f"Figure_4.1{chr(97 + p_idx)}_grid_{suffix}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


# ======================== 可视化: 三面板热力图 ========================

def plot_heatmap(all_results):
    """3 并排子热力图 (一个 / Price Level), 行=方案, 列=RE×Demand 组合。"""
    fig, axes = plt.subplots(1, 3, figsize=(21, 5), constrained_layout=True)
    fig.suptitle("Exp 4: Profit Heatmap — System Variant × Operating Scenario",
                 fontsize=12, fontweight="bold")

    matrices, all_tags = [], []
    for p_label in PRICE_LABELS:
        M = np.zeros((len(VARIANT_NAMES), 9))
        tags = []
        for r_idx, re_l in enumerate(RE_LABELS):
            for d_idx, d_l in enumerate(DEMAND_LABELS):
                col = r_idx * 3 + d_idx
                key = (p_label, re_l, d_l)
                re_s = re_l.split()[0][0]
                d_s = d_l.split()[0][0]
                tags.append(f"{re_s}R-{d_s}D")
                for v_idx, v in enumerate(VARIANT_NAMES):
                    M[v_idx, col] = _last_n_mean(all_results[key][v][1])
        matrices.append(M)
        all_tags.append(tags)

    vmin = min(m.min() for m in matrices)
    vmax = max(m.max() for m in matrices)

    im = None
    for p_idx in range(3):
        ax = axes[p_idx]
        M = matrices[p_idx]
        tags = all_tags[p_idx]

        im = ax.imshow(M, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(9))
        ax.set_xticklabels(tags, fontsize=6, rotation=45, ha="right")
        ax.set_yticks(range(len(VARIANT_NAMES)))
        if p_idx == 0:
            ax.set_yticklabels(VARIANT_NAMES, fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.set_title(PRICE_LABELS[p_idx], fontsize=10, fontweight="bold")

        for i in range(len(VARIANT_NAMES)):
            for j in range(9):
                mid = (vmin + vmax) / 2
                txt_c = "white" if abs(M[i, j] - mid) > (vmax - vmin) * 0.35 else "black"
                ax.text(j, i, f"${M[i, j]:,.0f}", ha="center", va="center",
                        fontsize=5.5, color=txt_c, fontweight="bold")

    fig.colorbar(im, ax=axes.tolist(), label="Avg Daily Profit ($)",
                 shrink=0.85, pad=0.02)

    path = "Figure_4.2_scenario_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ======================== 汇总表 ========================

def print_summary(all_results):
    col_w = 14
    print("\n" + "=" * 105)
    print("  SUMMARY TABLE — Avg Daily Profit (Last 20 Episodes)")
    print("=" * 105)

    header = f"  {'Scenario':<36}"
    for v in VARIANT_NAMES:
        header += f" {v:>{col_w}}"
    header += f"  {'Δ vs w/o CC':>12}"
    print(header)
    print("  " + "-" * 102)

    proposed_wins = 0
    total_scenarios = 0

    for p_label in PRICE_LABELS:
        print(f"\n  ── {p_label} {'─' * 70}")
        for re_l in RE_LABELS:
            for d_l in DEMAND_LABELS:
                key = (p_label, re_l, d_l)
                tag = f"{p_label[:3]}P × {re_l} × {d_l}"
                res = all_results[key]
                vals = [_last_n_mean(res[v][1]) for v in VARIANT_NAMES]

                line = f"  {tag:<36}"
                for v in vals:
                    line += f" ${v:>{col_w - 1},.0f}"
                baseline = vals[-1]
                if abs(baseline) > 1e-3:
                    delta = (vals[0] - baseline) / abs(baseline) * 100
                    line += f"  {delta:>+10.1f}%"
                else:
                    line += f"  {'N/A':>12}"
                print(line)

                total_scenarios += 1
                if vals[0] >= max(vals[1:]):
                    proposed_wins += 1

    print("\n  " + "=" * 102)
    print(f"  Proposed ranks #1 in {proposed_wins}/{total_scenarios} scenarios.")
    print("  Δ vs w/o CC: improvement of Proposed (RL+4T+CC) over single-stage naive.\n")


# ======================== 主函数 ========================

def main():
    saved_cfg = _save_cfg()
    print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    print("=" * 70)
    print("  Exp 4: 3×3×3 Scenario-Based Robustness Validation")
    print("  K-Medoids on (Price × RE × Demand) + 4 system variants")
    print("=" * 70)

    # ── Phase 1: 三维场景提取 ──
    print("\n  Phase 1: Extracting representative scenarios ...")
    price_scenarios, price_feat, price_labels = extract_price_scenarios()
    re_scenarios, re_feat, re_labels = extract_re_scenarios()
    demand_scenarios, demand_feat, demand_labels = extract_demand_scenarios()

    plot_clustering_price(price_feat, price_labels, price_scenarios)
    plot_clustering_re(re_feat, re_labels, re_scenarios)
    plot_clustering_demand(demand_feat, demand_labels, demand_scenarios)

    # ── Phase 2: 逐场景训练 / 评估 (27 scenarios × 4 variants) ──
    total = len(PRICE_LABELS) * len(RE_LABELS) * len(DEMAND_LABELS)
    print(f"\n  Phase 2: Training & evaluation across {total} scenarios × 4 variants ...")
    all_results = {}
    count = 0

    for p_info in price_scenarios:
        for re_info in re_scenarios:
            for d_info in demand_scenarios:
                count += 1
                key = (p_info["label"], re_info["label"], d_info["label"])
                pri = p_info["price_day_idx"]
                pi = re_info["pv_idx"]
                wi = re_info["wind_idx"]
                ds = d_info["demand_scale"]

                print(f"\n{'=' * 70}")
                print(f"  [{count}/{total}] {key[0]} × {key[1]} × {key[2]}  "
                      f"(price day {pri}, PV {pi}, Wind {wi}, demand ×{ds:.2f})")
                print(f"{'=' * 70}")

                res = {}

                # (1) Proposed
                _restore_cfg(saved_cfg)
                Config.enable_vsd = True
                Config.enable_dynamic_cooling = True
                Config.enable_bypass = True
                Config.enable_adaptive_pressure = True
                print("  [1/4] Proposed ...")
                set_seed(42)
                r, p = train_sac(
                    lambda _pr=pri, _pi=pi, _wi=wi, _ds=ds:
                        ScenarioEnv(_pr, _pi, _wi, _ds, enable_i2s_constraint=True)
                )
                res["Proposed"] = (r, p)
                print(f"        → Profit MA = ${_last_n_mean(p):,.0f}")

                # (2) w/o RL
                print("  [2/4] w/o RL ...")
                set_seed(42)
                r, p = eval_random(
                    lambda _pr=pri, _pi=pi, _wi=wi, _ds=ds:
                        ScenarioEnv(_pr, _pi, _wi, _ds, enable_i2s_constraint=True)
                )
                res["w/o RL"] = (r, p)
                print(f"        → Profit mean = ${np.mean(p):,.0f}")

                # (3) w/o 4T
                _restore_cfg(saved_cfg)
                print("  [3/4] w/o 4T ...")
                set_seed(42)
                r, p = train_sac(
                    lambda _pr=pri, _pi=pi, _wi=wi, _ds=ds:
                        ScenarioNaiveEnv("naive_3stage", _pr, _pi, _wi, _ds,
                                         enable_i2s_constraint=True)
                )
                res["w/o 4T"] = (r, p)
                print(f"        → Profit MA = ${_last_n_mean(p):,.0f}")

                # (4) w/o CC
                _restore_cfg(saved_cfg)
                print("  [4/4] w/o CC ...")
                set_seed(42)
                r, p = train_sac(
                    lambda _pr=pri, _pi=pi, _wi=wi, _ds=ds:
                        ScenarioNaiveEnv("naive_1stage", _pr, _pi, _wi, _ds,
                                         enable_i2s_constraint=True)
                )
                res["w/o CC"] = (r, p)
                print(f"        → Profit MA = ${_last_n_mean(p):,.0f}")

                all_results[key] = res

    _restore_cfg(saved_cfg)

    # ── Phase 3: 可视化 ──
    print("\n  Phase 3: Generating figures ...")
    plot_grids(all_results)
    plot_heatmap(all_results)
    print_summary(all_results)


if __name__ == "__main__":
    main()
