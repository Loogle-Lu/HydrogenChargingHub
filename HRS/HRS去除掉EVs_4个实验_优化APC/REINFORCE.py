"""
REINFORCE - 原始策略梯度算法 (Vanilla Policy Gradient)

Monte Carlo 策略梯度，使用完整 episode 回报。
作为 baseline 与 PPO/A2C 等改进算法对比，证明 GAE、Clipping 等技术的价值。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class PolicyNetwork(nn.Module):
    """REINFORCE 策略网络 (Actor + Value 作为 Baseline)"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        features = self.shared(state)
        action_mean = torch.tanh(self.actor_mean(features))
        action_std = torch.exp(self.actor_log_std)
        value = self.critic(features)
        return action_mean, action_std, value

    def act(self, state):
        action_mean, action_std, value = self.forward(state)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        action = (action + 1.0) / 2.0
        action = torch.clamp(action, 0.0, 1.0)
        return action, log_prob, value

    def evaluate(self, state, action):
        action_mean, action_std, value = self.forward(state)
        action_normalized = action * 2.0 - 1.0
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action_normalized).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, value, entropy


class REINFORCE:
    """
    REINFORCE (Vanilla Policy Gradient)

    核心特性:
    1. Monte Carlo 回报: G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...
    2. Value Baseline: Advantage = G_t - V(s_t)，降低方差
    3. 单次更新: 每 Episode 结束后更新一次
    4. 无 GAE、无 Clipping: 原始策略梯度形式

    与 PPO/A2C 对比:
    - 方差大 (MC 回报 vs GAE)
    - 无策略约束，易发散
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        c1=0.5,
        c2=0.01,
        max_grad_norm=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.gamma = gamma
        self.c1 = c1
        self.c2 = c2
        self.max_grad_norm = max_grad_norm

        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.95
        )

        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

        self.total_updates = 0

    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                action_mean, _, _ = self.policy(state_t)
                action = (action_mean + 1.0) / 2.0
                action = torch.clamp(action, 0.0, 1.0)
                return action.cpu().numpy().flatten()
            else:
                action, log_prob, value = self.policy.act(state_t)
                self.log_probs.append(log_prob.item())
                self.values.append(value.item())
                return action.cpu().numpy().flatten()

    def store_transition(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def _compute_returns(self):
        """Monte Carlo 回报: G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}"""
        returns = []
        G = 0
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)
        return np.array(returns)

    def update(self):
        """
        REINFORCE 更新: Monte Carlo + Value Baseline
        advantage_t = G_t - V(s_t)
        """
        if len(self.states) == 0:
            return {}

        states = np.array(self.states)
        actions = np.array(self.actions)
        returns = self._compute_returns()
        old_values = np.array(self.values)

        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        log_probs, values, entropy = self.policy.evaluate(states_t, actions_t)

        actor_loss = -(log_probs * advantages_t).mean()
        value_loss = F.mse_loss(values.squeeze(), returns_t)
        entropy_loss = -entropy.mean()

        loss = actor_loss + self.c1 * value_loss + self.c2 * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.total_updates += 1

        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': value_loss.item(),
            'entropy': -entropy_loss.item()
        }

    def step_scheduler(self):
        self.scheduler.step()
