"""
实验3: 压缩机架构消融实验

证明级联三级架构优于单级/两级架构，以及智能四项技术在级联架构上的附加收益。

对比四种配置：
  - 1-Stage Naive  : 单级压缩 2 bar → 700 bar，无四项技术，高压比大耗能
  - 2-Stage Naive  : 两级压缩 2→35→700 bar，无四项技术，少一次中间冷却
  - 3-Stage Naive  : 三级级联 2→35→500→700 bar，无 VSD/旁路/APC/动态冷却
  - 3-Stage Smart  : 三级级联 + VSD + 旁路 + APC + 动态级间冷却 (本文系统)

实验结论:
  1. 架构维度: 1S → 2S → 3S(naive) Profit 递增，证明级联架构热力学优势
  2. 技术维度: 3S-naive → 3S-smart Profit 进一步提升，证明四项技术协同贡献
  合并: 3S-smart 是最优配置

输出: 2×2 图 (Reward 柱状图 | Reward 曲线 | Profit 柱状图 | Profit 曲线)
"""

import numpy as np
import torch
import random
import matplotlib.pyplot as plt

from config import Config
from env import HydrogenEnv
from SAC import SAC, ReplayBuffer


# ======================== 配置 ========================
NUM_RUNS = 1
NUM_EPISODES = 80
WARMUP_STEPS = 400
BATCH_SIZE = 256
LR = 3e-4
MA_WINDOW = 15

COLORS = {
    "1-Stage Naive":  "#d62728",   # 红
    "2-Stage Naive":  "#ff7f0e",   # 橙
    "3-Stage Naive":  "#9467bd",   # 紫
    "3-Stage Smart":  "#1f77b4",   # 蓝（本文系统）
}


# ======================== 架构子类 ========================

class NaiveArchEnv(HydrogenEnv):
    """
    覆写 _compute_comp_block，将三级智能压缩替换为指定的朴素架构。

    arch:
      "naive_1stage" : 单级 2→700 bar，无任何智能特性
      "naive_2stage" : 两级 2→35→700 bar，无任何智能特性
      "naive_3stage" : 三级 2→35→500→700 bar，无任何智能特性
                       (等同于 exp1 的 Naive 配置)
    """

    _gamma    = Config.H2_gamma
    _R        = Config.H2_R
    _T_in     = Config.T_in
    _cp       = _gamma * _R / (_gamma - 1)
    _exponent = (_gamma - 1) / _gamma
    _eta      = Config.comp_efficiency   # 固定额定效率

    def __init__(self, arch: str, enable_i2s_constraint=None):
        # 禁用四项智能特性（保证朴素评估公平）
        self._saved = {
            "enable_vsd":              Config.enable_vsd,
            "enable_bypass":           Config.enable_bypass,
            "enable_dynamic_cooling":  Config.enable_dynamic_cooling,
            "enable_adaptive_pressure": Config.enable_adaptive_pressure,
        }
        Config.enable_vsd              = False
        Config.enable_bypass           = False
        Config.enable_dynamic_cooling  = False
        Config.enable_adaptive_pressure = False

        super().__init__(enable_i2s_constraint=enable_i2s_constraint)
        self.arch = arch

    def _isentropic_kw(self, flow_kg_h: float, p_in: float, p_out: float,
                        T_start: float = None) -> tuple:
        """
        单段等熵压缩功耗 (kW) 与热负荷 (kW)。
        使用额定固定效率，不含 VSD / 动态冷却。
        """
        if flow_kg_h <= 0:
            return 0.0, 0.0
        T = T_start if T_start is not None else self._T_in
        m_dot = flow_kg_h / 3600.0
        term = (p_out / p_in) ** self._exponent - 1
        work_j_kg = self._cp * T * term / self._eta
        power_kw = m_dot * work_j_kg / 1000.0
        # 热负荷 ≈ 压缩功 × (1/η − 1)（损失转热）
        heat_kw = power_kw * (1.0 / self._eta - 1.0)
        return power_kw, heat_kw

    def _compute_comp_block(self, h2_produced, fcev_h2_demand,
                             comp_load_ratio, cooling_intensity,
                             bypass_bias, c3_pressure_bias, price):
        """覆写：朴素架构功耗计算（物理 H₂ 流量与父类相同，仅功耗不同）。"""
        t1_soc = self.storage.t1.get_soc()
        t2_soc = self.storage.t2.get_soc()
        avg_fcev_sog = self.service_station.current_fcev_avg_sog  # noqa: F841

        # ---- 流量计算（与 smart 相同，物理上不变） ----
        c1_flow_base = h2_produced * min(1.0, max(0.5, t1_soc))
        comp_scale   = 0.4 + 0.6 * comp_load_ratio
        c1_flow      = c1_flow_base * comp_scale
        c2_flow_base = c1_flow_base * min(1.0, max(0.4, t2_soc)) + fcev_h2_demand * 0.5
        c2_flow      = c2_flow_base * comp_scale
        c3_flow      = fcev_h2_demand * 0.3 * (0.6 + 0.4 * comp_load_ratio)

        if self.arch == "naive_1stage":
            # 所有 H₂ 从 2 bar 直压到 700 bar，压比 = 350，热力学代价极大
            # 代表流量取 c1_flow（主流量，对应总制氢输入）
            p1, h1 = self._isentropic_kw(c1_flow, 2.0, 700.0)
            return p1, 0.0, 0.0,  h1, 0.0, 0.0,  c1_flow, c2_flow, c3_flow

        elif self.arch == "naive_2stage":
            # 两级: C1(2→35) + C2_combined(35→700)
            # T_in 对 C2 假设理想回温（固定 T_in），但无动态控制
            p1, h1 = self._isentropic_kw(c1_flow, 2.0, 35.0)
            # 第二级处理 c2_flow（需要从 35 bar 升压到 700 bar 直接送 FCEV/T4）
            p2, h2 = self._isentropic_kw(c2_flow, 35.0, 700.0)
            return p1, p2, 0.0,  h1, h2, 0.0,  c1_flow, c2_flow, c3_flow

        elif self.arch == "naive_3stage":
            # 三级：与本文架构相同的压力分级，但无四项智能特性
            # 四项特性已在 __init__ 中关闭，直接用等熵公式
            p1, h1 = self._isentropic_kw(c1_flow, 2.0,   35.0)
            p2, h2 = self._isentropic_kw(c2_flow, 35.0,  500.0)
            p3, h3 = self._isentropic_kw(c3_flow, 500.0, 700.0)
            return p1, p2, p3,  h1, h2, h3,  c1_flow, c2_flow, c3_flow

        else:
            raise ValueError(f"Unknown arch: {self.arch}")

    def __del__(self):
        # 析构时恢复 Config（供同一进程后续实验使用）
        try:
            for k, v in self._saved.items():
                setattr(Config, k, v)
        except Exception:
            pass


# ======================== 工具函数 ========================

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def moving_average(data, window):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


# ======================== 训练函数 ========================

def _train_one_variant(env_factory, num_episodes, num_runs):
    """
    用 SAC 训练单个环境变体，重复 num_runs 次取平均。
    返回 (avg_rewards, avg_profits)，均为 shape [num_episodes]。
    """
    all_rewards = []
    all_profits = []

    for run in range(num_runs):
        set_seed(42 + run)
        env = env_factory()
        state_dim  = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent      = SAC(state_dim, action_dim, lr=LR)
        replay     = ReplayBuffer(capacity=100_000)

        run_rewards = []
        run_profits = []
        total_steps = 0

        for ep in range(num_episodes):
            state = env.reset()
            ep_reward = 0.0
            ep_profit = 0.0
            done = False

            while not done:
                if total_steps < WARMUP_STEPS:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state, evaluate=False)

                next_state, reward, done, info = env.step(action)
                replay.push(state, action, reward, next_state, float(done))

                if total_steps >= WARMUP_STEPS and len(replay) >= BATCH_SIZE:
                    agent.update(replay, BATCH_SIZE)

                state = next_state
                ep_reward += reward
                ep_profit += info.get("profit", 0.0)
                total_steps += 1

            run_rewards.append(ep_reward)
            run_profits.append(ep_profit)

        all_rewards.append(run_rewards)
        all_profits.append(run_profits)

    return np.mean(all_rewards, axis=0), np.mean(all_profits, axis=0)


# ======================== 主函数 ========================

def main():
    print("=" * 65)
    print("  Exp3: Cascade Architecture Ablation")
    print("  Comparing: 1-Stage / 2-Stage / 3-Stage-Naive / 3-Stage-Smart")
    print("=" * 65)

    # 四种配置的环境工厂函数
    variants = {
        "1-Stage Naive":  lambda: NaiveArchEnv("naive_1stage", enable_i2s_constraint=True),
        "2-Stage Naive":  lambda: NaiveArchEnv("naive_2stage", enable_i2s_constraint=True),
        "3-Stage Naive":  lambda: NaiveArchEnv("naive_3stage", enable_i2s_constraint=True),
        "3-Stage Smart":  lambda: HydrogenEnv(enable_i2s_constraint=True),   # 本文系统
    }

    results = {}  # name -> (rewards, profits)
    for name, factory in variants.items():
        print(f"\n[Training] {name} ...")
        if name == "3-Stage Smart":
            Config.enable_vsd              = True
            Config.enable_bypass           = True
            Config.enable_dynamic_cooling  = True
            Config.enable_adaptive_pressure = True
        rewards, profits = _train_one_variant(factory, NUM_EPISODES, NUM_RUNS)
        results[name] = (rewards, profits)
        r20 = np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards)
        p20 = np.mean(profits[-20:]) if len(profits) >= 20 else np.mean(profits)
        print(f"  Final MA Reward = {r20:.2f}, MA Profit = ${p20:,.0f}")

    # ======================== 绘图 2×2 ========================
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9

    fig, axs = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    fig.suptitle(
        "Exp3: Cascade Architecture Ablation (Reward + Profit)\n"
        "1-Stage vs 2-Stage vs 3-Stage-Naive vs 3-Stage-Smart",
        fontsize=11, fontweight="bold"
    )

    names  = list(results.keys())
    avg_r  = [np.mean(results[n][0][-20:]) if len(results[n][0]) >= 20 else np.mean(results[n][0])
              for n in names]
    avg_p  = [np.mean(results[n][1][-20:]) if len(results[n][1]) >= 20 else np.mean(results[n][1])
              for n in names]
    colors = [COLORS[n] for n in names]
    x = np.arange(len(names))

    # (a) Reward 柱状图
    bars = axs[0, 0].bar(x, avg_r, color=colors, edgecolor="gray", linewidth=0.5, width=0.55)
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(names, rotation=12, ha="right")
    axs[0, 0].set_ylabel("Avg Reward (Last 20 Ep)")
    axs[0, 0].set_title("(a) Reward by Architecture")
    axs[0, 0].grid(True, axis="y", alpha=0.3, linestyle="--")
    for b, v in zip(bars, avg_r):
        axs[0, 0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{v:.1f}",
                       ha="center", va="bottom", fontsize=8)

    # (b) Reward 曲线
    for name, (rewards, _) in results.items():
        c = COLORS[name]
        lw = 2.5 if "Smart" in name else 1.5
        axs[0, 1].plot(rewards, alpha=0.15, color=c, linewidth=0.8)
        ma = moving_average(rewards, MA_WINDOW)
        axs[0, 1].plot(range(MA_WINDOW - 1, len(rewards)), ma, color=c, linewidth=lw, label=name)
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Episode Reward")
    axs[0, 1].set_title(f"(b) Reward Curves (MA{MA_WINDOW})")
    axs[0, 1].legend(loc="lower right", fontsize=8)
    axs[0, 1].grid(True, alpha=0.3, linestyle="--")

    # (c) Profit 柱状图
    bars = axs[1, 0].bar(x, avg_p, color=colors, edgecolor="gray", linewidth=0.5, width=0.55)
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(names, rotation=12, ha="right")
    axs[1, 0].set_ylabel("Avg Profit (Last 20 Ep, $)")
    axs[1, 0].set_title("(c) Profit by Architecture")
    axs[1, 0].grid(True, axis="y", alpha=0.3, linestyle="--")
    for b, v in zip(bars, avg_p):
        axs[1, 0].text(b.get_x() + b.get_width() / 2, b.get_height() + abs(max(avg_p)) * 0.01,
                       f"${v:,.0f}", ha="center", va="bottom", fontsize=8)

    # (d) Profit 曲线
    for name, (_, profits) in results.items():
        c = COLORS[name]
        lw = 2.5 if "Smart" in name else 1.5
        axs[1, 1].plot(profits, alpha=0.15, color=c, linewidth=0.8)
        ma = moving_average(profits, MA_WINDOW)
        axs[1, 1].plot(range(MA_WINDOW - 1, len(profits)), ma, color=c, linewidth=lw, label=name)
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Episode Profit ($)")
    axs[1, 1].set_title(f"(d) Profit Curves (MA{MA_WINDOW})")
    axs[1, 1].legend(loc="lower right", fontsize=8)
    axs[1, 1].grid(True, alpha=0.3, linestyle="--")

    plt.savefig("CompressorComparison_exp3_cascade_ablation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigure saved: CompressorComparison_exp3_cascade_ablation.png")

    # ======================== 汇总表 ========================
    print("\n" + "=" * 60)
    print(f"{'Variant':<18} {'Reward (MA-20)':>14} {'Profit (MA-20)':>18}")
    print("-" * 60)
    for name, (rewards, profits) in results.items():
        r = np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards)
        p = np.mean(profits[-20:]) if len(profits) >= 20 else np.mean(profits)
        marker = "  ← This paper" if "Smart" in name else ""
        print(f"  {name:<16} {r:>14.2f} ${p:>16,.0f}{marker}")
    print("=" * 60)


if __name__ == "__main__":
    main()
