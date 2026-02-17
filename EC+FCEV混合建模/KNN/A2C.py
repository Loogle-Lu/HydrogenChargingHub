import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """
    A2C 的 Actor-Critic 网络
    与 PPO 结构相同，但更新规则不同（无 clipping）
    """
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


class A2C:
    """
    Advantage Actor-Critic (A2C) 算法

    核心特性:
    1. 同步 Actor-Critic: 共享特征层，Actor 和 Critic 同步更新
    2. Advantage Function: A(s,a) = R + γV(s') - V(s)，降低方差
    3. 单次更新: 每个 Episode 只更新一次（不像 PPO 多轮）
    4. 熵正则化: 鼓励探索

    与 PPO 的区别:
    - 无 Clipped Surrogate Objective（直接策略梯度）
    - 无 Multiple Epochs（单次遍历）
    - 训练更快但稳定性较差
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        c1=0.5,
        c2=0.01,
        max_grad_norm=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.c1 = c1
        self.c2 = c2
        self.max_grad_norm = max_grad_norm

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.95
        )

        # On-policy buffer
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

    def _compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        values_ext = values + [next_value]
        for t in reversed(range(len(rewards))):
            nv = next_value if t == len(rewards) - 1 else values_ext[t + 1]
            delta = rewards[t] + self.gamma * nv * (1 - dones[t]) - values_ext[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self):
        """
        A2C 更新: 单次遍历，无 clipping
        loss = -log_prob * advantage + c1 * value_loss - c2 * entropy
        """
        if len(self.states) == 0:
            return {}

        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = self.rewards
        dones = self.dones
        old_values = self.values

        # 计算 GAE
        with torch.no_grad():
            last_state = torch.FloatTensor(states[-1]).unsqueeze(0).to(self.device)
            _, _, last_value = self.policy(last_state)
            last_value = last_value.item()

        advantages = self._compute_gae(rewards, old_values, dones, last_value)
        advantages = np.array(advantages)
        returns = advantages + np.array(old_values)

        # 标准化 advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 转 tensor
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # 单次前向 + 反向 (A2C 核心: 无 clipping, 无 multiple epochs)
        log_probs, values, entropy = self.policy.evaluate(states_t, actions_t)

        # Actor loss: 直接策略梯度 (无 clipping)
        actor_loss = -(log_probs * advantages_t).mean()

        # Critic loss
        value_loss = F.mse_loss(values.squeeze(), returns_t)

        # Entropy bonus
        entropy_loss = -entropy.mean()

        loss = actor_loss + self.c1 * value_loss + self.c2 * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.total_updates += 1

        # 清空 buffer
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
