import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import City, calculate_economic_loss, do_migration


class SEIREnv:
    def __init__(self, cities_config, migration_matrix, time_window=5, max_weeks=52):
        self.num_cities = len(cities_config)
        self.time_window = time_window
        self.max_weeks = max_weeks
        self.current_week = 0
        self.cities_config = cities_config
        self.migration_matrix = migration_matrix
        self.ema_alpha = 0.1
        self.reset()

    def _create_cities(self):
        cities = []
        for config in self.cities_config:
            city_config = config.copy()
            for param in ['beta_day', 'gamma_day', 'sigma_day']:
                if isinstance(city_config[param], (tuple, list)):
                    city_config[param] = np.random.uniform(*city_config[param])
            city_config['I0'] = np.random.randint(50, 200)
            cities.append(City(**city_config))
        return cities

    def reset(self):
        self.current_week = 0
        self.cities = self._create_cities()
        self.init_history()
        self.last_action = np.zeros(3 * self.num_cities)  # 修正初始化维度
        self.ema_action = np.zeros(3 * self.num_cities)
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
        for i, city in enumerate(self.cities):
            infected = np.array(self.I_history[city.name]) / city.total_population()
            economy = np.array(list(self.economy_history[city.name])) / city.economy_base
            city_action = self.last_action[i * 3: (i + 1) * 3]
            features = [
                *infected.tolist(),
                *economy.tolist(),
                city.type / 3.0,
                *city_action
            ]
            state_features.extend(features)
        return np.array(state_features, dtype=np.float32)

    def step(self, action):
        # 策略平滑处理
        smoothed_action = 0.6 * self.last_action + 0.4 * np.clip(action, 0, 1)
        self.last_action = smoothed_action
        self.ema_action = (1 - self.ema_alpha) * self.ema_action + self.ema_alpha * smoothed_action

        # 拆分策略到各个城市
        city_actions = np.reshape(smoothed_action, (self.num_cities, 3))
        total_reward = 0
        weekly_infections = 0
        weekly_economy = 0

        # 模拟一周发展
        for day in range(7):
            for city, params in zip(self.cities, city_actions):
                # 更新城市状态
                city.update({
                    'mask': params[0],
                    'home': params[1],
                    'med': params[2]
                })

                # 计算经济损失（修正调用方式）
                loss = calculate_economic_loss(
                    city,
                    mask=params[0],
                    home=params[1],
                    med=params[2]
                )
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
                -10 * infection_ratio  # 感染惩罚项
                - 5 * economy_ratio  # 经济惩罚项
                - 0.1 * np.sum(self.ema_action)  # 干预强度惩罚
        )

        self.current_week += 1
        done = self.current_week >= self.max_weeks

        return self.get_state(), reward, done, {}


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, num_cities=3):
        super().__init__()
        self.num_cities = num_cities
        self.terrain_embed = nn.Embedding(4, 32)  # 0-3对应不同地形

        # 特征提取层
        self.shared_feature = nn.Sequential(
            nn.Linear((input_dim // num_cities) - 1, 256),  # 修正输入维度
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256)
        )

        # 地形特定策略头
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256 + 32, 64),
                nn.Tanh(),
                nn.Linear(64, 3),
                nn.Sigmoid()
            ) for _ in range(num_cities)
        ])

        self.log_std = nn.Parameter(torch.zeros(3 * num_cities))

    def forward(self, x):
        batch_size = x.shape[0]
        city_states = x.view(batch_size, self.num_cities, -1)  # [batch_size, num_cities, state_dim_per_city]

        mus = []
        for i in range(self.num_cities):
            # 提取地形特征
            terrain = (city_states[:, i, -1] * 3).round().long().clamp(0, 3)
            embed = self.terrain_embed(terrain)

            # 拼接特征
            features = torch.cat([
                self.shared_feature(city_states[:, i, :-1]),  # 去掉地形特征
                embed
            ], dim=1)

            # 生成策略
            mus.append(self.heads[i](features))

        mu = torch.cat(mus, dim=1)  # [batch_size, num_cities * 3]
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mu, std)


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
        state = torch.FloatTensor(state).unsqueeze(0)
        dist = self.policy(state)
        action = dist.sample() if explore else dist.mean
        self.saved_log_probs.append(dist.log_prob(action).sum())
        return action.squeeze(0).detach().numpy()

    def update_policy(self):
        R = 0
        returns = []
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


# 城市配置（保持相同）
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
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.show()
    return agent


def visualize_policy(agent, num_simulations=3):
    plt.rcParams.update({'font.size': 12})
    for sim in range(num_simulations):
        plt.figure(figsize=(18, 12))
        gs = plt.GridSpec(4, 3, height_ratios=[1.2, 1.2, 1.2, 0.8])

        state = env.reset()
        done = False
        records = {
            'infections': [[] for _ in env.cities],
            'economy': [[] for _ in env.cities],
            'actions': []
        }

        while not done:
            # 修正输入形状为 [1, state_dim]
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 增加 batch 维度
            with torch.no_grad():
                action = agent.policy(state_tensor).mean.numpy()[0]  # 去掉 batch 维度
            records['actions'].append(action)

            # 应用策略并记录状态
            city_actions = np.reshape(action, (env.num_cities, 3))  # 将动作重塑为 [num_cities, 3]
            for i, city in enumerate(env.cities):
                # 计算经济损失
                loss = calculate_economic_loss(
                    city,
                    mask=city_actions[i][0],
                    home=city_actions[i][1],
                    med=city_actions[i][2]
                )
                records['economy'][i].append(loss)
                records['infections'][i].append(city.I)

            state, _, done, _ = env.step(action)

        # 将 actions 转换为三维数组 [num_steps, num_cities, 3]
        actions_array = np.array(records['actions'])  # 形状为 [num_steps, num_cities * 3]
        actions_array = actions_array.reshape(-1, env.num_cities, 3)  # 重塑为 [num_steps, num_cities, 3]

        # 感染趋势
        ax1 = plt.subplot(gs[0, :])
        for i, city in enumerate(env.cities):
            ax1.plot(np.array(records['infections'][i]) / city.total_population(),
                     label=f'{city.name} ({city.type})')
        ax1.set_title(f'Simulation {sim + 1} - Normalized Infection Rate')
        ax1.legend()

        # 经济损失
        ax2 = plt.subplot(gs[1, :])
        for i, city in enumerate(env.cities):
            ax2.plot(np.array(records['economy'][i]) / city.economy_base,
                     label=f'{city.name}')
        ax2.set_title('Normalized Economic Loss')
        ax2.legend()

        # 分城市策略
        for i, city in enumerate(env.cities):
            ax = plt.subplot(gs[2, i])
            for j in range(3):
                ax.plot(actions_array[:, i, j],  # 使用 actions_array
                        label=['Mask Policy', 'Home Policy', 'Medical Policy'][j])
            ax.set_title(f'{city.name} Policies')
            ax.set_ylim(0, 1)
            if i == 0:
                ax.legend()

        # 策略热力图
        ax4 = plt.subplot(gs[3, :])
        im = ax4.imshow(actions_array.transpose(1, 0, 2).reshape(-1, actions_array.shape[0]),
                        aspect='auto', cmap='YlGnBu')
        plt.colorbar(im, ax=ax4)
        ax4.set_yticks(np.arange(len(env.cities) * 3))
        ax4.set_yticklabels([f'{c.name}\n{p}'
                             for c in env.cities
                             for p in ['Mask', 'Home', 'Med']])
        ax4.set_title('Policy Intensity Evolution')
        ax4.set_xlabel('Weeks')

        plt.tight_layout()
        plt.show()


# 初始化环境和训练
env = SEIREnv(cities_config, migration_matrix)
trained_agent = train_with_seeds()
visualize_policy(trained_agent)