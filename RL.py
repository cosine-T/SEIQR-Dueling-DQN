import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import City, calculate_economic_loss, do_migration


# 强化学习环境类
class SEIREnv:
    def __init__(self, cities_config, migration_matrix, time_window=5, max_weeks=52):
        self.time_window = time_window
        self.max_weeks = max_weeks
        self.current_week = 0
        self.cities_config = cities_config
        self.migration_matrix = migration_matrix
        self.ema_alpha = 0.1
        self.reset()

    def _create_cities(self):
        """创建带随机初始化的城市"""
        cities = []
        for config in self.cities_config:
            city_config = config.copy()
            # 参数随机化
            for param in ['beta_day', 'gamma_day', 'sigma_day']:
                if isinstance(city_config[param], (tuple, list)):
                    city_config[param] = np.random.uniform(*city_config[param])
            # 初始感染人数随机化
            city_config['I0'] = np.random.randint(50, 200)
            cities.append(City(**city_config))
        return cities

    def reset(self):
        """完全重置环境"""
        self.current_week = 0
        self.cities = self._create_cities()
        self.init_history()
        self.last_action = np.zeros(3)
        self.ema_action = np.zeros(3)
        return self.get_state()

    def init_history(self):
        self.I_history = {city.name: deque(maxlen=self.time_window) for city in self.cities}
        self.economy_history = {city.name: deque(maxlen=self.time_window) for city in self.cities}
        for city in self.cities:
            for _ in range(self.time_window):
                self.I_history[city.name].append(city.I)
                self.economy_history[city.name].append(0)

    def get_state(self):
        state_features = []
        total_pop = sum(c.total_population() for c in self.cities)
        for city in self.cities:
            infected = np.array(self.I_history[city.name]) / city.total_population()
            economy = np.array(list(self.economy_history[city.name])) / city.economy_base
            features = [
                *infected.tolist(),
                *economy.tolist(),
                city.type / 3.0,
                self.last_action[0],
                self.last_action[1],
                self.last_action[2]
            ]
            state_features.extend(features)
        return np.array(state_features, dtype=np.float32)

    def step(self, action):
        # 带惯性的动作更新
        smoothed_action = 0.6 * self.last_action + 0.4 * np.clip(action, 0, 1)
        self.last_action = smoothed_action
        self.ema_action = (1 - self.ema_alpha) * self.ema_action + self.ema_alpha * smoothed_action

        params = {'mask': smoothed_action[0],
                  'home': smoothed_action[1],
                  'med': smoothed_action[2]}

        total_reward = 0
        weekly_infections = 0
        weekly_economy = 0

        # 模拟一周发展
        for day in range(7):
            for city in self.cities:
                city.update(params)
                loss = calculate_economic_loss(city, **params)
                self.economy_history[city.name].append(loss)
                weekly_economy += loss
                weekly_infections += city.I

            # 更新感染历史
            for city in self.cities:
                self.I_history[city.name].append(city.I)

        # 人口迁移
        total_infected = sum(c.I for c in self.cities)
        alpha = 0.6 if total_infected > 50000 else 0.0
        do_migration(self.cities, self.migration_matrix, alpha, self.current_week)

        # 重新设计的奖励函数
        infection_ratio = total_infected / sum(c.total_population() for c in self.cities)
        economy_ratio = weekly_economy / sum(c.economy_base for c in self.cities)

        reward = (
                - 10 * infection_ratio  # 感染惩罚项
                - 5 * economy_ratio  # 经济惩罚项
                - 0.1 * np.sum(self.ema_action)  # 干预强度惩罚
        )

        self.current_week += 1
        done = self.current_week >= self.max_weeks

        return self.get_state(), reward, done, {}


# 增强的策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(128),
        )
        self.mu_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )
        self.log_std_head = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        features = self.feature_net(x)
        mu = self.mu_head(features)
        std = torch.exp(self.log_std_head)
        return torch.distributions.Normal(mu, std)


# 改进的强化学习智能体
class RLAgent:
    def __init__(self, env, lr=3e-5, gamma=0.97):
        self.env = env
        self.gamma = gamma
        state_dim = len(env.get_state())
        self.policy = PolicyNetwork(state_dim)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state, explore=True):
        state = torch.FloatTensor(state)
        dist = self.policy(state)
        action = dist.sample() if explore else dist.mean
        self.saved_log_probs.append(dist.log_prob(action).sum())
        return action.detach().numpy()

    def update_policy(self):
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        del self.rewards[:]
        del self.saved_log_probs[:]


# 确保 cities_config 是一个有效的列表
cities_config = [
    {
        'name': "平原城市",
        'S0': 999800, 'E0': 150, 'I0': 50, 'R0': 0,
        'beta_day': (0.18, 0.22),
        'gamma_day': (0.08, 0.12),
        'sigma_day': (0.45, 0.55),
        'city_type': 1,
        'economy_base': 1e5
    },
    {
        'name': "丘陵城市",
        'S0': 799900, 'E0': 80, 'I0': 20, 'R0': 0,
        'beta_day': (0.16, 0.20),
        'gamma_day': (0.10, 0.14),
        'sigma_day': (0.35, 0.45),
        'city_type': 2,
        'economy_base': 8e4
    },
    {
        'name': "高原城市",
        'S0': 499900, 'E0': 50, 'I0': 50, 'R0': 0,
        'beta_day': (0.22, 0.28),
        'gamma_day': (0.06, 0.10),
        'sigma_day': (0.25, 0.35),
        'city_type': 3,
        'economy_base': 5e4
    }
]

# 确保 migration_matrix 是一个有效的 NumPy 数组
migration_matrix = np.array([
    [0.00, 0.05, 0.02],
    [0.04, 0.00, 0.03],
    [0.01, 0.06, 0.00],
])

# 初始化环境
env = SEIREnv(cities_config, migration_matrix)

# 训练参数
episodes = 800
print_interval = 50


# 训练过程（带多个随机种子）
def train_with_seeds(seeds=[42, 123, 999]):
    all_rewards = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        agent = RLAgent(env)
        episode_rewards = []

        for ep in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.rewards.append(reward)
                total_reward += reward
                state = next_state

            agent.update_policy()
            episode_rewards.append(total_reward)

            if ep % print_interval == 0:
                avg_reward = np.mean(episode_rewards[-print_interval:])
                print(f"Seed {seed} | Episode {ep:4d} | Avg Reward: {avg_reward:7.2f}")

        all_rewards.append(episode_rewards)

    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    for i, rewards in enumerate(all_rewards):
        plt.plot(rewards, label=f'Seed {seeds[i]}')
    plt.title('Training Progress with Different Seeds')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

    return agent


# 运行多种子训练
trained_agent = train_with_seeds()


# 增强可视化函数
def visualize_policy(agent, num_simulations=3):
    plt.figure(figsize=(15, 18))

    for sim in range(num_simulations):
        # 运行模拟
        state = env.reset()
        done = False
        records = {
            'infections': [[] for _ in env.cities],
            'economy': [[] for _ in env.cities],
            'actions': []
        }

        while not done:
            with torch.no_grad():
                action = agent.policy(torch.FloatTensor(state)).mean.numpy()
            records['actions'].append(action)

            # 记录状态
            for i, city in enumerate(env.cities):
                records['infections'][i].append(city.I)
                # 修复：将 deque 转换为列表后再切片
                economy_history = list(env.economy_history[city.name])
                records['economy'][i].append(sum(economy_history[-7:]))

            state, _, done, _ = env.step(action)

        # 绘制感染趋势
        plt.subplot(num_simulations, 3, sim * 3 + 1)
        for i, city in enumerate(env.cities):
            plt.plot(records['infections'][i], label=city.name)
        plt.title(f'Simulation {sim + 1} - Infections')
        plt.xlabel('Week')
        plt.ylabel('Infected')
        plt.legend()

        # 绘制经济损失
        plt.subplot(num_simulations, 3, sim * 3 + 2)
        for i, city in enumerate(env.cities):
            normalized_loss = [x / city.economy_base for x in records['economy'][i]]
            plt.plot(normalized_loss, label=city.name)
        plt.title(f'Simulation {sim + 1} - Economic Loss')
        plt.xlabel('Week')
        plt.ylabel('Loss Ratio')
        plt.legend()

        # 绘制干预措施
        plt.subplot(num_simulations, 3, sim * 3 + 3)
        actions = np.array(records['actions'])
        labels = ['Mask', 'Home', 'Medical']
        for i in range(3):
            plt.plot(actions[:, i], label=labels[i])
        plt.title(f'Simulation {sim + 1} - Interventions')
        plt.xlabel('Week')
        plt.ylabel('Intensity')
        plt.legend()

    plt.tight_layout()
    plt.show()


# 运行可视化
visualize_policy(trained_agent)