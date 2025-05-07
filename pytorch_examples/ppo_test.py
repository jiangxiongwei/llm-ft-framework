import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# 策略网络（包含值函数）
class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        action_logits = self.fc2(x)
        value = self.value_head(x)
        return Categorical(logits=action_logits), value

# 超参数
gamma = 0.99
epsilon = 0.2
learning_rate = 1e-3  # 提高学习率
num_epochs = 1000
max_steps = 500
value_loss_coef = 0.1  # 降低值损失权重
num_updates = 10  # 增加每次更新的次数

# 初始化环境和网络
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
policy = Policy(state_size, action_size)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    states, actions, rewards, values, log_probs = [], [], [], [], []
    state, _ = env.reset()
    state = torch.FloatTensor(state)
    done = False
    step = 0

    # 数据收集
    while not done and step < max_steps:
        dist, value = policy(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        states.append(state)
        actions.append(action)
        values.append(value)
        log_probs.append(log_prob)

        next_state, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        state = torch.FloatTensor(next_state)
        rewards.append(reward)
        step += 1

    # 计算回报
    returns = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.FloatTensor(returns)

    # 标准化回报（避免除以零）
    if returns.std() > 1e-6:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # 转换为张量
    states = torch.stack(states)
    actions = torch.stack(actions)
    values = torch.stack(values).squeeze(-1)  # 确保值形状正确
    log_probs = torch.stack(log_probs)

    # 计算优势
    advantages = returns - values.detach()
    if advantages.std() > 1e-6:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # PPO 更新
    policy_loss_total = 0.0
    value_loss_total = 0.0
    for _ in range(num_updates):
        dist, value = policy(states)
        new_log_probs = dist.log_prob(actions)
        ratios = torch.exp(new_log_probs - log_probs.detach())

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.MSELoss()(value.squeeze(-1), returns)

        loss = policy_loss + value_loss_coef * value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)  # 放宽梯度裁剪
        optimizer.step()

        policy_loss_total += policy_loss.item()
        value_loss_total += value_loss.item()

    # 打印详细日志
    episode_length = len(rewards)
    avg_return = sum(rewards)  # 直接使用累积奖励
    if epoch % 50 == 0:
        print(f"Epoch {epoch:4d}, Episode Length: {episode_length:3d}, Avg Return: {avg_return:.2f}, "
              f"Policy Loss: {policy_loss_total/num_updates:.4f}, Value Loss: {value_loss_total/num_updates:.4f}")

env.close()