"""
SAC (Soft Actor-Critic) — 针对 HRS 环境优化版

相比朴素版的关键改动 (均有学术依据):
1. entropy_scale 0.5 (原 1.0): 降低过度探索, 加速收敛到高利润策略
2. hidden_dim 512 (原 256): 更大网络 + 熵正则 = 更好泛化 (Haarnoja et al. 2018)
3. 增加 Critic 网络深度: 3 hidden layers, 更好拟合复杂 Q 函数
4. tau 0.005 → 0.005 (保持): 稳定的软目标更新
5. 学习率分离: Actor lr = 1e-4 (慢), Critic lr = 3e-4 (快), 防止 Actor 领先 Critic
6. Reward normalization in replay buffer: 稳定 Critic 训练
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class FastReplayBuffer:
    """
    【极速版 GPU 经验回放池 + Reward 归一化】
    新增: 存入时用 running mean/std 归一化 reward → Critic 训练更稳定
    """
    def __init__(self, state_dim, action_dim, capacity=200000, device="cuda"):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.state = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.action = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.reward = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_state = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.done = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

        # Reward normalization (Welford online algorithm)
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_count = 0

    def _update_reward_stats(self, reward):
        self._reward_count += 1
        delta = reward - self._reward_mean
        self._reward_mean += delta / self._reward_count
        delta2 = reward - self._reward_mean
        self._reward_var += delta * delta2

    def _normalize_reward(self, reward):
        if self._reward_count < 2:
            return reward
        std = np.sqrt(self._reward_var / self._reward_count + 1e-8)
        return (reward - self._reward_mean) / (std + 1e-8)

    def push(self, state, action, reward, next_state, done):
        self._update_reward_stats(float(reward))
        norm_reward = self._normalize_reward(float(reward))

        self.state[self.ptr] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.action[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.reward[self.ptr] = torch.as_tensor([norm_reward], dtype=torch.float32, device=self.device)
        self.next_state[self.ptr] = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        self.done[self.ptr] = torch.as_tensor([done], dtype=torch.float32, device=self.device)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.state[ind],
            self.action[ind],
            self.reward[ind],
            self.next_state[ind],
            self.done[ind]
        )

    def __len__(self):
        return self.size


class GaussianActor(nn.Module):
    """
    高斯策略网络 (SAC Actor) — 加宽至 512 hidden
    """
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()
        action_tanh = torch.tanh(z)
        action_scaled = (action_tanh + 1.0) / 2.0
        log_prob = dist.log_prob(z) - torch.log(0.5 * (1.0 - action_tanh.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action_scaled, log_prob

    def get_deterministic_action(self, state):
        mean, _ = self.forward(state)
        action_tanh = torch.tanh(mean)
        action_scaled = (action_tanh + 1.0) / 2.0
        return action_scaled


class QNetwork(nn.Module):
    """Q值网络 — 加深至 3 hidden layers (512)"""
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class SAC:
    """
    Soft Actor-Critic (SAC) — HRS 优化版

    关键改动:
    1. entropy_scale=0.5: 目标熵 = -0.5 × action_dim = -3.0
       (原 -6.0 过度探索, 后期 profit 追不上确定性策略)
    2. hidden_dim=512, Critic 3 layers: 更好拟合复杂 Q 函数
    3. Actor lr=1e-4, Critic lr=3e-4: 防止 Actor 领先 Critic (训练更稳)
    4. Replay buffer capacity=200000: 存更多经验, 减少遗忘
    5. Reward normalization: 在 buffer 中归一化, Critic 训练更稳定
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
        hidden_dim=512,          # ← 改: 256 → 512
        entropy_scale=0.5,       # ← 改: 1.0 → 0.5 (降低过度探索)
        max_grad_norm=1.0,
        actor_lr=None,           # ← 新增: Actor 单独学习率
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        self.actor = GaussianActor(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.critic1_target = QNetwork(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.critic2_target = QNetwork(state_dim, action_dim, hidden_dim=hidden_dim).to(device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Actor 学习率更低 → 等 Critic 估值稳定后再更新策略
        _actor_lr = actor_lr if actor_lr is not None else lr * 0.33
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=_actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)

        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -entropy_scale * float(action_dim)
            self.log_alpha = torch.tensor([np.log(alpha)], requires_grad=True,
                                          dtype=torch.float32, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha

        self.total_updates = 0
        self.max_grad_norm = max_grad_norm

    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                action = self.actor.get_deterministic_action(state_t)
            else:
                action, _ = self.actor.sample(state_t)
        return action.cpu().numpy().flatten()

    def update(self, replay_buffer, batch_size=256):
        if len(replay_buffer) < batch_size:
            return {}

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # 1. 更新 Critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        if self.max_grad_norm and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.max_grad_norm)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        if self.max_grad_norm and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.max_grad_norm)
        self.critic2_optimizer.step()

        # 2. 更新 Actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # 3. 更新 Alpha
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            if self.max_grad_norm and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_([self.log_alpha], self.max_grad_norm)
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # 4. Soft Target 更新
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_updates += 1

        return {
            'critic_loss': (critic1_loss.item() + critic2_loss.item()) / 2,
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha
        }