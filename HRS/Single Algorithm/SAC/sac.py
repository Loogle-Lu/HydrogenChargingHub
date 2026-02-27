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
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # [改进] 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """改进的权重初始化 - 更稳定的训练"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # 最后一层使用更小的初始化
        nn.init.uniform_(self.mean.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std.weight, -3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
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
        
        # [改进] 权重初始化 - 减少初始震荡
        self._initialize_weights()
    
    def _initialize_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        # Q1
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)  # [修复] 最后一层不用激活函数
        
        # Q2
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)  # [修复] 最后一层不用激活函数
        
        return q1, q2


class SAC:
    def __init__(self, state_dim, action_dim, gamma=0.99, tau=0.01, alpha=0.1, lr=1e-4):
        self.actor = Actor(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        
        # [改进] 自动熵调优 (Automatic Entropy Tuning)
        self.target_entropy = -action_dim  # 目标熵
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        # 奖励归一化参数 (减少震荡)
        self.reward_scale = 0.01  # 将奖励缩放到更小范围
        self.running_reward_mean = 0.0
        self.running_reward_std = 1.0
        self.reward_beta = 0.99  # 移动平均系数
        
        # [新增] 学习率调度器
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=50, gamma=0.95)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=50, gamma=0.95)
        
        # [新增] 更新计数器 (用于延迟策略更新)
        self.update_counter = 0
        self.policy_update_freq = 2  # 每2次critic更新，更新1次actor

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=64):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        # [新增] 奖励归一化 - 减少震荡
        # 更新移动平均和标准差
        reward_np = reward.numpy()
        self.running_reward_mean = self.reward_beta * self.running_reward_mean + (1 - self.reward_beta) * reward_np.mean()
        self.running_reward_std = self.reward_beta * self.running_reward_std + (1 - self.reward_beta) * reward_np.std()
        
        # 归一化奖励
        reward_normalized = (reward - self.running_reward_mean) / (self.running_reward_std + 1e-8)
        reward_normalized = torch.clamp(reward_normalized, -10, 10)  # 限制范围

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
            target_Q = reward_normalized + (1 - done) * self.gamma * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # [优化] 更严格的梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # [改进] 延迟策略更新 (Delayed Policy Update)
        # 每policy_update_freq次critic更新才更新一次actor
        # 这样可以让critic更稳定，减少actor的震荡
        self.update_counter += 1
        
        if self.update_counter % self.policy_update_freq == 0:
            action_new, log_prob_new = self.actor.sample(state)
            Q1_new, Q2_new = self.critic(state, action_new)
            Q_new = torch.min(Q1_new, Q2_new)
            actor_loss = (self.alpha * log_prob_new - Q_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # [优化] 更严格的梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # [新增] 自动熵调优 (Automatic Entropy Tuning)
            # 动态调整探索-利用平衡
            alpha_loss = -(self.log_alpha * (log_prob_new + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # [优化] 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def step_schedulers(self):
        """每个episode结束时调用，降低学习率"""
        self.actor_scheduler.step()
        self.critic_scheduler.step()