import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((capacity, state_dim))
        self.action = np.zeros((capacity, action_dim))
        self.reward = np.zeros((capacity, 1))
        self.next_state = np.zeros((capacity, state_dim))
        self.done = np.zeros((capacity, 1))

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]),
            torch.FloatTensor(self.action[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.done[ind])
        )


class Actor(nn.Module):
    """TD3 Actor - 确定性策略网络 (Deterministic Policy)"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """正交初始化 - 更稳定的训练"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # 最后一层使用更小的初始化
        nn.init.uniform_(self.l3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.l3.bias, -3e-3, 3e-3)

    def forward(self, state):
        """确定性输出动作 (无随机性)"""
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        # TD3关键: 输出确定性动作，使用tanh限制到[-1, 1]
        action = self.max_action * torch.tanh(self.l3(x))
        return action


class Critic(nn.Module):
    """TD3 Critic - Twin Q-Networks (双Q网络)"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Q1 network
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        
        # Q2 network
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """正交初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        # Q1
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        # Q2
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        """只返回Q1 (用于actor更新)"""
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)
    
    核心改进 (相比DDPG/SAC):
    1. Twin Q-Networks: 双Q网络减少过估计
    2. Delayed Policy Update: 延迟策略更新，每2次critic更新才更新1次actor
    3. Target Policy Smoothing: 目标策略平滑，给目标动作添加噪声
    
    优势:
    - 比SAC更稳定 (确定性策略 + 三大技巧)
    - 训练速度快 (无熵项计算)
    - 超参数鲁棒性强
    - 适合连续控制 (电解槽/燃料电池功率管理)
    """
    def __init__(self, state_dim, action_dim, gamma=0.99, tau=0.005, lr=3e-4,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        """
        参数:
            state_dim: 状态维度 (8维)
            action_dim: 动作维度 (2维)
            gamma: 折扣因子
            tau: 软更新系数 (TD3通常用0.005，比SAC的0.01更小)
            lr: 学习率 (TD3通常用3e-4)
            policy_noise: 目标策略平滑噪声标准差
            noise_clip: 噪声裁剪范围
            policy_freq: 策略更新频率 (每policy_freq次critic更新才更新1次actor)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化网络
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # TD3超参数
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        # 奖励归一化参数
        self.reward_scale = 0.01
        self.running_reward_mean = 0.0
        self.running_reward_std = 1.0
        self.reward_beta = 0.99
        
        # 学习率调度器
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=50, gamma=0.95)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=50, gamma=0.95)
        
        # 更新计数器 (用于延迟策略更新)
        self.total_it = 0
        
        # 探索噪声 (训练时添加)
        self.expl_noise = 0.1  # 探索噪声标准差

    def select_action(self, state, add_noise=True):
        """
        选择动作
        
        参数:
            state: 当前状态
            add_noise: 是否添加探索噪声 (训练时True，测试时False)
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        # TD3: 训练时添加高斯噪声进行探索
        if add_noise:
            noise = np.random.normal(0, self.expl_noise, size=action.shape)
            action = (action + noise).clip(-1, 1)
        
        return action

    def update(self, replay_buffer, batch_size=64):
        """
        更新网络
        
        TD3核心逻辑:
        1. 每次都更新Critic (双Q网络)
        2. 每policy_freq次才更新Actor (延迟策略更新)
        3. 目标动作添加噪声 (目标策略平滑)
        """
        self.total_it += 1
        
        # 从replay buffer采样
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        
        # 奖励归一化
        reward_np = reward.numpy()
        self.running_reward_mean = self.reward_beta * self.running_reward_mean + (1 - self.reward_beta) * reward_np.mean()
        self.running_reward_std = self.reward_beta * self.running_reward_std + (1 - self.reward_beta) * reward_np.std()
        reward_normalized = (reward - self.running_reward_mean) / (self.running_reward_std + 1e-8)
        reward_normalized = torch.clamp(reward_normalized, -10, 10).to(self.device)
        
        with torch.no_grad():
            # TD3 Trick 3: Target Policy Smoothing
            # 给目标动作添加噪声，减少Q值过估计
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)
            
            # TD3 Trick 1: Twin Q-Networks
            # 使用两个Q网络的最小值作为目标，减少过估计
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward_normalized + (1 - done) * self.gamma * target_Q
        
        # 更新Critic
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # TD3 Trick 2: Delayed Policy Update
        # 每policy_freq次才更新一次actor，让critic更稳定
        if self.total_it % self.policy_freq == 0:
            # 更新Actor (最大化Q1)
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # 软更新目标网络
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def step_schedulers(self):
        """每个episode结束时调用，降低学习率"""
        self.actor_scheduler.step()
        self.critic_scheduler.step()
    
    def save(self, filename):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filename)
    
    def load(self, filename):
        """加载模型"""
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
