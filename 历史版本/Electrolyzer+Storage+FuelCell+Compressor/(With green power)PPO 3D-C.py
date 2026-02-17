# ==========================================
# 0. 安装依赖
# ==========================================
try:
    import gymnasium
    import stable_baselines3
except ImportError:
    print("正在安装依赖库...")
    !pip install gymnasium stable-baselines3 shimmy matplotlib pandas tqdm > /dev/null 2>&1
    print("依赖安装完成！")

# ==========================================
# 1. 导入库与数据读取
# ==========================================
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import os
import pickle
import copy
import random
from google.colab import drive
from tqdm.notebook import tqdm

# --- 全局数据容器 ---
PRICE_DATA = None
PV_DATA = None
WD_DATA = None

def load_data():
    global PRICE_DATA, PV_DATA, WD_DATA
    print("正在初始化... 尝试挂载 Google Drive 读取价格数据...")
    try:
        # 尝试挂载 Drive
        if not os.path.exists('/content/drive'):
            drive.mount('/content/drive')
    except:
        pass

    # 定义可能的路径
    possible_paths = [
        "/content/drive/MyDrive/price_after_MAD_96.pkl",
        "/content/drive/MyDrive/RL_environment/price_after_MAD_96.pkl",
        "./price_after_MAD_96.pkl"
    ]
    
    # 1. 读取价格 (严格遵循你的要求)
    price_loaded = False
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                PRICE_DATA = np.array(pickle.load(f))
            print(f"✅ 价格数据加载成功: {path}")
            price_loaded = True
            break
    
    if not price_loaded:
        print("⚠️ 未找到价格文件，使用合成正弦波数据。")
        t = np.linspace(0, 2*np.pi, 96)
        PRICE_DATA = np.clip(20 + 10*np.sin(t), 1, 50)

    # 2. 读取光伏和风电 (尝试寻找类似路径)
    base_dirs = ["/content/drive/MyDrive/", "/content/drive/MyDrive/RL_environment/Data-file", "./"]
    
    for base in base_dirs:
        pv_p = os.path.join(base, "pv_power_100.pkl")
        if os.path.exists(pv_p) and PV_DATA is None:
            with open(pv_p, 'rb') as f: PV_DATA = pickle.load(f)
            print(f"✅ PV数据加载成功: {pv_p}")
            
        wd_p = os.path.join(base, "wd_power_150.pkl")
        if os.path.exists(wd_p) and WD_DATA is None:
            with open(wd_p, 'rb') as f: WD_DATA = pickle.load(f)
            print(f"✅ Wind数据加载成功: {wd_p}")

load_data()

# ==========================================
# 2. 训练监控回调 (带进度条集成)
# ==========================================
class TQDMCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.ep_rewards = []
        self.last_mean_reward = -np.inf

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="训练进度")

    def _on_step(self):
        self.pbar.update(1)
        
        # 记录奖励
        for info in self.locals['infos']:
            if 'episode' in info.keys():
                self.ep_rewards.append(info['episode']['r'])
        
        # 每1000步更新一次描述
        if self.n_calls % 1000 == 0 and self.ep_rewards:
            mean_reward = np.mean(self.ep_rewards[-50:]) # 最近50个episode的平均
            self.pbar.set_postfix({'Mean Reward': f'{mean_reward:.2f}'})
            self.last_mean_reward = mean_reward
            
        return True

    def _on_training_end(self):
        if self.pbar: self.pbar.close()

# ==========================================
# 3. 物理组件 (保持不变)
# ==========================================
class Electrolyzer:
    def __init__(self, max_power, efficiency=0.02): self.max, self.eff = max_power, efficiency
    def run(self, action): return (p := (action + 1) / 2 * self.max), p * self.eff

class FuelCell:
    def __init__(self, max_power, efficiency=0.06): self.max, self.eff = max_power, efficiency
    def run(self, action): return (p := (action + 1) / 2 * self.max), p * self.eff

class Compressor:
    def __init__(self, max_flow, energy_cost=1.5): self.max, self.cost = max_flow, energy_cost
    def run(self, action, limit): 
        real_flow = min((action + 1) / 2 * self.max, limit)
        return real_flow * self.cost, real_flow

class HydrogenTank:
    def __init__(self, cap, soc): self.cap, self.lvl = cap, cap * soc
    def update(self, flow): self.lvl = np.clip(self.lvl + flow, 0, self.cap); return self.lvl

# ==========================================
# 4. 核心环境 (深度优化版)
# ==========================================
class HybridChargingHubEnv(gym.Env):
    def __init__(self):
        super(HybridChargingHubEnv, self).__init__()
        self.steps_per_day = 96
        self.days = 5
        self.max_steps = self.steps_per_day * self.days
        self.dt = 0.25
        
        # 绿色能源参数
        self.scale_pv = 3.0 
        self.scale_wd = 1.0
        
        # 经济参数 (假设你的价格数据单位是 cents/kWh, 这里转换为 $/kWh 需要除以100，或者调整其他参数)
        # 如果 PRICE_DATA 是 20-50，假设这是 cents。那么 20 cents = $0.20。
        # 我们这里为了保持数值敏感度，统一把单位视为 "Money Unit"
        self.params = {
            'ev_fee': 10.0,       # 充电服务费 (在基础电价上叠加)
            'h2_price': 15.0,     # 氢气售价 (调高一点以激励制氢)
            'pen_i2s': 100.0,     # I2S 约束惩罚
            'pen_miss': 20.0,     # 拒载惩罚
        }
        
        self.ely = Electrolyzer(300.0)
        self.fc = FuelCell(100.0)
        self.comp = Compressor(5.0)
        self.lp = HydrogenTank(20.0, 0.0)
        self.hp = HydrogenTank(200.0, 0.5)
        self.init_hp_lvl = self.hp.lvl

        self.action_space = spaces.Box(-1, 1, (3,), np.float32)
        # 观察空间: [HP, LP, Price_Forecast, H2_Dem, EV_Dem, PV_Forecast, WD_Forecast]
        self.observation_space = spaces.Box(0, 5000, (7,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.lp.lvl = 0
        self.hp.lvl = self.init_hp_lvl
        self.profit = 0
        self._gen_scenario_refined()
        return self._get_obs(), {}

    def _gen_scenario_refined(self):
        """精细化建模：预测 (Forecast) vs 实际 (Actual) + 你的价格数据"""
        len_needed = self.max_steps + 1
        
        # 1. 价格数据 (使用全局 PRICE_DATA)
        # 将 96 步的数据平铺到所需的长度
        base_price = PRICE_DATA
        if len(base_price) < 96: # 简单的长度检查
             base_price = np.resize(base_price, 96)
        
        full_price = np.tile(base_price[:96], self.days + 1)[:len_needed]
        # 电价通常比较确定，但也加一点微小的随机性
        self.ac_price = np.clip(full_price + np.random.normal(0, 0.5, len_needed), 0.1, 100)
        self.fc_price = full_price # Agent 看到的是标准曲线

        # 2. 光伏 & 风电 (真实数据采样 + 自相关误差)
        raw_pv, raw_wd = [], []
        for _ in range(self.days + 1):
            if PV_DATA: raw_pv.extend(np.array(random.choice(PV_DATA))[:96])
            else: raw_pv.extend(300 * np.maximum(0, np.sin((np.linspace(0,24,96)-6)*np.pi/12)))
            
            if WD_DATA: raw_wd.extend(np.array(random.choice(WD_DATA))[:96])
            else: raw_wd.extend(100 * np.abs(np.sin(np.linspace(0,24,96)/5)))

        ac_pv_base = np.array(raw_pv[:len_needed]) * self.scale_pv
        ac_wd_base = np.array(raw_wd[:len_needed]) * self.scale_wd

        # 生成“自相关”预测误差 (更真实的精细化建模)
        # 意思是：如果现在云层遮挡了，接下来几小时可能都会遮挡
        def correlated_noise(n, rho=0.9):
            x = np.zeros(n)
            for i in range(1, n):
                x[i] = rho * x[i-1] + np.random.normal(0, 0.1)
            return np.exp(x) # 乘法噪声

        # 实际值 = 基础曲线 * 自相关噪声
        self.ac_pv = ac_pv_base * correlated_noise(len_needed)
        self.ac_wd = ac_wd_base * correlated_noise(len_needed, rho=0.8)
        
        # 预测值 = 基础曲线 (Agent 以为天气是完美的)
        self.fc_pv = ac_pv_base
        self.fc_wd = ac_wd_base
        
        # 3. 需求
        t = np.linspace(0, 24*self.days, len_needed)
        self.dem_ev = np.clip(200*(np.exp(-((t%24-9)**2)/5)+np.exp(-((t%24-18)**2)/5)), 20, 400)
        self.dem_h2 = np.random.choice([0, 5.0], p=[0.9, 0.1], size=len_needed)

    def _get_obs(self):
        i = self.step_idx
        # Agent 看到的是 Forecast (fc_)
        return np.array([self.hp.lvl, self.lp.lvl, self.fc_price[i], self.dem_h2[i], 
                         self.dem_ev[i], self.fc_pv[i], self.fc_wd[i]], dtype=np.float32)

    def step(self, action):
        i = self.step_idx
        # 环境执行用 Actual (ac_)
        price = self.ac_price[i]
        real_pv = self.ac_pv[i]
        real_wd = self.ac_wd[i]
        
        # --- 物理流 ---
        p_fc_tgt, h2_fc_in = self.fc.run(action[2])
        h2_fc_real = min(h2_fc_in, self.hp.lvl)
        p_fc_real = p_fc_tgt * (h2_fc_real / (h2_fc_in + 1e-6))
        
        p_ely, h2_ely = self.ely.run(action[0])
        avail_h2_lp = self.lp.lvl + h2_ely
        p_comp, h2_moved = self.comp.run(action[1], avail_h2_lp)
        
        # --- 储罐 ---
        self.lp.update(h2_ely - h2_moved)
        h2_sales = min(self.dem_h2[i], self.hp.lvl - h2_fc_real)
        self.hp.update(h2_moved - (h2_fc_real + h2_sales))
        
        # --- 能量平衡 (核心逻辑) ---
        load = self.dem_ev[i] + p_ely + p_comp
        gen = real_pv + real_wd + p_fc_real
        grid_import = max(0, load - gen)
        ren_consumed = min(load, gen) # 自发自用量
        
        # --- 财务 ---
        # 成本
        cost = grid_import * price * self.dt
        # 收入 (EV充电 + 氢气销售)
        # 注意: 这里的 price 是你的原始数据 (20-50)，加上 ev_fee (10)
        income = (self.dem_ev[i]*(price + self.params['ev_fee']) + h2_sales*self.params['h2_price']) * self.dt
        step_profit = income - cost
        self.profit += step_profit
        
        # 奖励整形
        reward = step_profit * 0.01 # 缩放以稳定训练
        reward -= (self.dem_h2[i] - h2_sales) * self.params['pen_miss']
        
        self.step_idx += 1
        done = self.step_idx >= self.max_steps
        
        if self.step_idx % 96 == 0:
            reward -= abs(self.hp.lvl - self.init_hp_lvl) * self.params['pen_i2s']
            
        info = {
            'profit': step_profit, 'hp': self.hp.lvl, 'grid': grid_import, 
            'ren': ren_consumed, 'ev': self.dem_ev[i], 'ely': p_ely,
            'pv': real_pv, 'wd': real_wd, 'price': price
        }
        
        return self._get_obs(), reward, done, False, info

# ==========================================
# 5. 训练与可视化
# ==========================================
def train_and_visualize():
    # 必须用 Monitor 包裹环境，否则无法记录 Episode 奖励
    env = DummyVecEnv([lambda: Monitor(HybridChargingHubEnv())])
    
    total_steps = 500000 # 按照你的要求改为 500k
    
    print(f">>> 开始训练 ({total_steps} 步)...")
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4)
    model.learn(total_timesteps=total_steps, callback=TQDMCallback(total_steps))
    print("\n>>> 训练结束!")

    print(">>> 开始测试...")
    test_env = HybridChargingHubEnv()
    obs, _ = test_env.reset()
    
    H = {k: [] for k in ['profit', 'cum_profit', 'hp', 'grid', 'ren', 'ev', 'ely', 'pv', 'wd', 'price']}
    
    done = False
    cum_p = 0
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _, info = test_env.step(action)
        
        cum_p += info['profit']
        H['cum_profit'].append(cum_p)
        for k in info: 
            if k in H: H[k].append(info[k])

    # === 可视化 ===
    fig, ax = plt.subplots(5, 1, figsize=(12, 22), sharex=True)
    steps = range(len(H['hp']))
    
    # 1. 利润
    ax[0].plot(steps, H['cum_profit'], 'g', lw=2)
    ax[0].set_title('Cumulative Profit', fontweight='bold')
    ax[0].grid(True, alpha=0.3)
    
    # 2. 储氢状态
    ax[1].plot(steps, H['hp'], 'b', label='Tank Level')
    ax[1].axhline(100, c='r', ls='--', label='Initial Target')
    ax[1].set_title(f"H2 Level (End: {H['hp'][-1]:.1f})")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    # 3. 功率堆叠 (解释网购电量)
    # 灰色=EV需求, 橙色=制氢需求. 如果橙色很高且没有绿色(新能源), 就要买电(红色)
    ax[2].stackplot(steps, H['ev'], H['ely'], labels=['EV Demand', 'H2 Prod (Ely)'], colors=['grey', 'orange'], alpha=0.6)
    total_ren = np.array(H['pv']) + np.array(H['wd'])
    ax[2].plot(steps, total_ren, 'g--', lw=1.5, label='Renewable Available')
    ax[2].legend(loc='upper right')
    ax[2].set_title('Load Breakdown vs Renewable Supply')
    
    # 4. 能源来源 (自用 vs 网购)
    ax[3].stackplot(steps, H['ren'], H['grid'], labels=['Renewable Consumed', 'Grid Purchased'], colors=['#90EE90', '#FF6347'])
    ax[3].legend(loc='upper right')
    ax[3].set_title('Energy Source Breakdown')
    
    # 5. 你的价格数据
    ax[4].plot(steps, H['price'], color='purple', linestyle='-')
    ax[4].set_title('Grid Electricity Price (From Data)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_and_visualize()
