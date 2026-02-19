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
    """确定性策略网络"""
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
        action = torch.tanh(action)
        action = (action + 1.0) / 2.0
        return action


class QNetwork(nn.Module):
    """Q 值网络 (单个，区别于 TD3 的 Twin)"""
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


class DDPG:
    """
    Deep Deterministic Policy Gradient (DDPG) 算法

    核心特性:
    1. 确定性策略: Actor 输出确定性动作
    2. 单 Critic: 只有一个 Q 网络 (区别于 TD3 的 Twin)
    3. 每步更新: Actor 和 Critic 同步更新 (无延迟)
    4. 软目标更新: Target networks 缓慢跟踪

    与 TD3 的区别:
    - 单 Q 网络 (容易过估计)
    - 无 Delayed Policy Update
    - 无 Target Policy Smoothing
    - 训练更快但稳定性较差
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        exploration_noise=0.1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.exploration_noise = exploration_noise
        self.action_dim = action_dim

        # Actor
        self.actor = DeterministicActor(state_dim, action_dim).to(device)
        self.actor_target = DeterministicActor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Single Critic (区别于 TD3 的 Twin Critics)
        self.critic = QNetwork(state_dim, action_dim).to(device)
        self.critic_target = QNetwork(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.total_updates = 0

    def select_action(self, state, evaluate=False):
        """选择动作 (训练时加高斯探索噪声)"""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().flatten()

        if not evaluate:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = np.clip(action + noise, 0.0, 1.0)

        return action

    def update(self, replay_buffer, batch_size=256):
        """DDPG 更新: Critic + Actor 同步，每步都更新"""
        if len(replay_buffer) < batch_size:
            return {}

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # --- 1. 更新 Critic ---
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            # 单 Q 网络，无 min 操作 (区别于 TD3)
            q_next = self.critic_target(next_states, next_actions)
            q_target = rewards + (1 - dones) * self.gamma * q_next

        q_current = self.critic(states, actions)
        critic_loss = F.mse_loss(q_current, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- 2. 更新 Actor (每步都更新，无延迟) ---
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- 3. 软目标更新 ---
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_updates += 1

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
        }
