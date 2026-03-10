import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=100000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state, dtype=np.float32),
                np.array(action, dtype=np.float32),
                np.array(reward, dtype=np.float32).reshape(-1, 1),
                np.array(next_state, dtype=np.float32),
                np.array(done, dtype=np.float32).reshape(-1, 1))

    def __len__(self):
        return len(self.buffer)


class DeterministicActor(nn.Module):
    """
    确定性策略网络 (TD3 Actor)
    输出确定性动作 ∈ [0, 1]
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        action = self.net(state)
        action = torch.tanh(action)  # [-1, 1]
        action = (action + 1.0) / 2.0  # [0, 1]
        return action


class QNetwork(nn.Module):
    """Q值网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
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


class TD3:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) 算法

    核心特性:
    1. Twin Q-Networks: 双Q网络取最小值，缓解过估计
    2. Delayed Policy Update: 每隔d步才更新Actor
    3. Target Policy Smoothing: 给目标动作加噪声，平滑Q值
    4. Exploration Noise: 训练时给动作加高斯噪声

    优势:
    - Off-Policy: 样本效率高
    - 确定性策略: 适合连续控制
    - 训练稳定性强 (Twin + Delay)
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.1,
        noise_clip=0.25,
        policy_delay=2,
        exploration_noise=0.1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.exploration_noise = exploration_noise
        self.action_dim = action_dim

        # Actor
        self.actor = DeterministicActor(state_dim, action_dim).to(device)
        self.actor_target = DeterministicActor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Twin Critics
        self.critic1 = QNetwork(state_dim, action_dim).to(device)
        self.critic2 = QNetwork(state_dim, action_dim).to(device)
        self.critic1_target = QNetwork(state_dim, action_dim).to(device)
        self.critic2_target = QNetwork(state_dim, action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)

        self.total_updates = 0

    def select_action(self, state, evaluate=False):
        """选择动作 (训练时加探索噪声)"""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().flatten()

        if not evaluate:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = np.clip(action + noise, 0.0, 1.0)

        return action

    def update(self, replay_buffer, batch_size=256):
        """TD3更新: Critic每步 → Actor延迟更新"""
        if len(replay_buffer) < batch_size:
            return {}

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # --- 1. 更新 Critic (每步) ---
        with torch.no_grad():
            # Target Policy Smoothing
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(0.0, 1.0)

            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            q_target = rewards + (1 - dones) * self.gamma * q_next

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        self.total_updates += 1

        # --- 2. 延迟更新 Actor + Target Networks ---
        if self.total_updates % self.policy_delay == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft Target 更新 (Actor + Critics)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_loss': (critic1_loss.item() + critic2_loss.item()) / 2,
        }
