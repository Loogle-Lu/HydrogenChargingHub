import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """
    PPO的Actor-Critic网络
    
    Actor: 输出动作的均值和标准差 (高斯策略)
    Critic: 输出状态值函数 V(s)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # 共享特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor分支 (策略网络)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))  # 可学习的log标准差
        
        # Critic分支 (价值网络)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # 正交初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        """前向传播，返回动作分布和状态值"""
        features = self.shared_layers(state)
        
        # Actor输出
        action_mean = torch.tanh(self.actor_mean(features))  # 限制在[-1,1]
        action_std = torch.exp(self.actor_log_std)
        
        # Critic输出
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def act(self, state):
        """选择动作（用于训练时采样）"""
        action_mean, action_std, value = self.forward(state)
        
        # 创建高斯分布
        dist = Normal(action_mean, action_std)
        
        # 采样动作
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        
        # 将动作映射到[0, 1]范围
        action = (action + 1.0) / 2.0
        action = torch.clamp(action, 0.0, 1.0)
        
        return action, action_log_prob, value
    
    def evaluate(self, state, action):
        """评估状态-动作对（用于更新时计算log_prob和entropy）"""
        action_mean, action_std, value = self.forward(state)
        
        # 将动作从[0,1]映射回[-1,1]
        action_normalized = action * 2.0 - 1.0
        
        # 创建高斯分布
        dist = Normal(action_mean, action_std)
        
        # 计算log概率和熵
        action_log_prob = dist.log_prob(action_normalized).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action_log_prob, value, entropy


class RolloutBuffer:
    """
    PPO的经验回放缓冲区（on-policy）
    存储一个完整的rollout
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
    
    def get(self):
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.dones),
            np.array(self.log_probs),
            np.array(self.values)
        )


class PPO:
    """
    Proximal Policy Optimization (PPO) 算法
    
    核心特性:
    1. Clipped Surrogate Objective: 限制策略更新幅度
    2. GAE (Generalized Advantage Estimation): 优势函数估计
    3. Multiple Epochs: 使用同一批数据多次更新
    4. Value Function Clipping: 限制价值函数更新
    
    优势:
    - 实现简单，易于调试
    - 训练稳定，样本效率较好
    - 适合连续控制任务
    - OpenAI等机构广泛使用
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        c1=0.5,  # value loss coefficient
        c2=0.01,  # entropy coefficient
        max_grad_norm=0.5,
        update_epochs=10,
        minibatch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        
        # 创建Actor-Critic网络
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.95
        )
        
        # 经验缓冲区
        self.buffer = RolloutBuffer()
        
        # 统计信息
        self.total_updates = 0
        
        # 奖励归一化（running mean and std）
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 0
        
    def select_action(self, state, evaluate=False):
        """
        选择动作
        
        参数:
            state: 当前状态
            evaluate: 是否为评估模式（True时使用确定性策略）
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                # 评估模式：使用均值作为动作
                action_mean, _, _ = self.policy(state)
                action = (action_mean + 1.0) / 2.0  # 映射到[0,1]
                action = torch.clamp(action, 0.0, 1.0)
                return action.cpu().numpy().flatten()
            else:
                # 训练模式：从分布中采样
                action, log_prob, value = self.policy.act(state)
                
                # 存储到buffer（用于后续更新）
                self.buffer.log_probs.append(log_prob.item())
                self.buffer.values.append(value.item())
                
                return action.cpu().numpy().flatten()
    
    def store_transition(self, state, action, reward, done):
        """存储经验"""
        # 奖励归一化
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_var += delta * delta2
        
        reward_std = np.sqrt(max(self.reward_var / self.reward_count, 1e-6))
        normalized_reward = (reward - self.reward_mean) / (reward_std + 1e-8)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.rewards.append(normalized_reward)
        self.buffer.dones.append(done)
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        计算广义优势估计 (Generalized Advantage Estimation, GAE)
        
        GAE平衡了偏差和方差:
        - lambda=0: 只使用TD误差（低方差，高偏差）
        - lambda=1: 使用MC回报（高方差，低偏差）
        """
        advantages = []
        gae = 0
        
        values = values + [next_value]
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_value
            else:
                next_value = values[t + 1]
            
            # TD误差
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE累积
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self):
        """
        PPO更新步骤
        
        1. 计算GAE和回报
        2. 多轮更新（使用同一批数据）
        3. 使用minibatch进行更新
        """
        # 获取buffer中的数据
        states, actions, rewards, dones, old_log_probs, values = self.buffer.get()
        
        # 计算最后一个状态的value（用于GAE计算）
        with torch.no_grad():
            last_state = torch.FloatTensor(states[-1]).unsqueeze(0).to(self.device)
            _, _, last_value = self.policy(last_state)
            last_value = last_value.item()
        
        # 计算GAE
        advantages = self.compute_gae(rewards.tolist(), values.tolist(), dones.tolist(), last_value)
        advantages = np.array(advantages)
        
        # 计算returns (value targets)
        returns = advantages + values
        
        # 标准化advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 多轮更新
        for epoch in range(self.update_epochs):
            # 创建minibatch索引
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start_idx in range(0, len(states), self.minibatch_size):
                end_idx = min(start_idx + self.minibatch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                # 提取minibatch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 评估当前策略
                log_probs, values, entropy = self.policy.evaluate(batch_states, batch_actions)
                
                # 计算ratio = π_θ(a|s) / π_θ_old(a|s)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Surrogate loss (clipped)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (MSE with clipping)
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy bonus (鼓励探索)
                entropy_loss = -entropy.mean()
                
                # 总损失
                loss = actor_loss + self.c1 * value_loss + self.c2 * entropy_loss
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                self.total_updates += 1
        
        # 清空buffer
        self.buffer.clear()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': value_loss.item(),
            'entropy': -entropy_loss.item()
        }
    
    def step_scheduler(self):
        """学习率调度"""
        self.scheduler.step()
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'total_updates': self.total_updates,
            'reward_mean': self.reward_mean,
            'reward_var': self.reward_var,
            'reward_count': self.reward_count
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.total_updates = checkpoint['total_updates']
        self.reward_mean = checkpoint['reward_mean']
        self.reward_var = checkpoint['reward_var']
        self.reward_count = checkpoint['reward_count']
        print(f"Model loaded from {filepath}")
