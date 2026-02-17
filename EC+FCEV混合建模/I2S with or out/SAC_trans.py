"""
SAC with Transformer - 基于 Transformer Encoder 的序列特征提取
用于建模状态序列的时序依赖
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import random
from collections import deque


class SequenceReplayBuffer:
    """序列经验回放缓冲区 (用于 Transformer-SAC)"""
    def __init__(self, capacity=100000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state_seq, action, reward, next_state_seq, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state_seq, action, reward, next_state_seq, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_seq, action, reward, next_state_seq, done = zip(*batch)
        return (np.array(state_seq, dtype=np.float32),
                np.array(action, dtype=np.float32),
                np.array(reward, dtype=np.float32).reshape(-1, 1),
                np.array(next_state_seq, dtype=np.float32),
                np.array(done, dtype=np.float32).reshape(-1, 1))

    def __len__(self):
        return len(self.buffer)


class TransformerFeatureExtractor(nn.Module):
    """序列特征提取器 (Transformer Encoder)"""
    def __init__(self, state_dim, seq_len, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Linear(state_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (B, seq_len, state_dim)
        h = self.embed(x) + self.pos_embed
        h = self.encoder(h)
        return h[:, -1, :]


class TransformerGaussianActor(nn.Module):
    """Transformer版高斯策略网络 (SAC Actor)"""
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, state_dim, action_dim, seq_len, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1):
        super().__init__()
        self.encoder = TransformerFeatureExtractor(
            state_dim=state_dim,
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_ff=dim_ff,
            dropout=dropout
        )
        self.mean_head = nn.Linear(d_model, action_dim)
        self.log_std_head = nn.Linear(d_model, action_dim)

    def forward(self, state_seq):
        x = self.encoder(state_seq)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state_seq):
        mean, log_std = self.forward(state_seq)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()
        action_tanh = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        action = (action_tanh + 1.0) / 2.0
        return action, log_prob

    def get_deterministic_action(self, state_seq):
        mean, _ = self.forward(state_seq)
        action = torch.tanh(mean)
        action = (action + 1.0) / 2.0
        return action


class TransformerQNetwork(nn.Module):
    """Transformer版Q网络"""
    def __init__(self, state_dim, action_dim, seq_len, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1):
        super().__init__()
        self.encoder = TransformerFeatureExtractor(
            state_dim=state_dim,
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_ff=dim_ff,
            dropout=dropout
        )
        self.q_head = nn.Sequential(
            nn.Linear(d_model + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state_seq, action):
        features = self.encoder(state_seq)
        x = torch.cat([features, action], dim=-1)
        return self.q_head(x)


class SAC_Transformer:
    """
    Soft Actor-Critic with Transformer
    使用 Transformer Encoder 对状态序列编码，建模时序依赖
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
        device="cuda" if torch.cuda.is_available() else "cpu",
        seq_len=8,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_ff=128,
        dropout=0.1
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.state_buffer = deque(maxlen=seq_len)

        self.actor = TransformerGaussianActor(
            state_dim, action_dim, seq_len,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_ff=dim_ff, dropout=dropout
        ).to(device)
        self.critic1 = TransformerQNetwork(
            state_dim, action_dim, seq_len,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_ff=dim_ff, dropout=dropout
        ).to(device)
        self.critic2 = TransformerQNetwork(
            state_dim, action_dim, seq_len,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_ff=dim_ff, dropout=dropout
        ).to(device)
        self.critic1_target = TransformerQNetwork(
            state_dim, action_dim, seq_len,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_ff=dim_ff, dropout=dropout
        ).to(device)
        self.critic2_target = TransformerQNetwork(
            state_dim, action_dim, seq_len,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_ff=dim_ff, dropout=dropout
        ).to(device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)

        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha

        self.total_updates = 0

    def reset_state_buffer(self):
        self.state_buffer.clear()

    def append_state(self, state):
        self.state_buffer.append(np.array(state, dtype=np.float32))

    def get_state_seq(self):
        if len(self.state_buffer) == 0:
            raise ValueError("state_buffer is empty. Call append_state() first.")
        seq = list(self.state_buffer)
        if len(seq) < self.seq_len:
            pad = [seq[0]] * (self.seq_len - len(seq))
            seq = pad + seq
        return np.array(seq, dtype=np.float32)

    def select_action(self, state, evaluate=False):
        """选择动作 (内部维护 state_buffer)"""
        if len(self.state_buffer) == 0:
            self.append_state(state)
        state_seq = self.get_state_seq()
        return self._select_action_from_seq(state_seq, evaluate=evaluate)

    def _select_action_from_seq(self, state_seq, evaluate=False):
        state_seq_t = torch.FloatTensor(state_seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                action = self.actor.get_deterministic_action(state_seq_t)
            else:
                action, _ = self.actor.sample(state_seq_t)
        return action.cpu().numpy().flatten()

    def update(self, replay_buffer, batch_size=256):
        """SAC更新: Critic → Actor → Alpha → Soft Target (序列输入)"""
        if len(replay_buffer) < batch_size:
            return {}

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        # states/next_states: (B, seq_len, state_dim)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

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
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

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
