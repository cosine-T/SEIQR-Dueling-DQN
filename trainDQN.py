import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from model import City, calculate_economic_loss, do_migration


# 强化学习环境类（修改动作空间处理）
class SEIREnv:
    def __init__(self, cities_config, migration_matrix, action_file='actions.xlsx', time_window=5, max_weeks=52):
        self.time_window = time_window
        self.max_weeks = max_weeks
        self.current_week = 0
        self.cities_config = cities_config
        self.migration_matrix = migration_matrix

        # 读取离散动作空间
        self.actions_df = pd.read_excel(action_file)
        self.actions = self.actions_df[['med', 'home', 'mask']].values
        self.num_actions = len(self.actions)

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
        self.last_action_params = np.zeros(3)  # 记录上次动作参数
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
        for city in self.cities:
            infected = np.array(self.I_history[city.name]) / city.total_population()
            economy = np.array(list(self.economy_history[city.name])) / city.economy_base
            features = [
                *infected.tolist(),
                *economy.tolist(),
                city.type / 3.0,
                self.last_action_params[0],
                self.last_action_params[1],
                self.last_action_params[2]
            ]
            state_features.extend(features)
        return np.array(state_features, dtype=np.float32)

    def step(self, action_idx):
        # 获取离散动作对应的参数
        action_params = self.actions[action_idx]
        self.last_action_params = action_params  # 更新最后动作参数

        total_reward = 0
        weekly_infections = 0
        weekly_economy = 0

        # 应用参数到所有城市
        params = {'mask': action_params[0], 'home': action_params[1], 'med': action_params[2]}

        for day in range(7):
            for city in self.cities:
                city.update(params)
                loss = calculate_economic_loss(city, **params)
                self.economy_history[city.name].append(loss)
                weekly_economy += loss
                weekly_infections += city.I

            for city in self.cities:
                self.I_history[city.name].append(city.I)

        # 人口迁移
        total_infected = sum(c.I for c in self.cities)
        alpha = 0.6 if total_infected > 50000 else 0.0
        do_migration(self.cities, self.migration_matrix, alpha, self.current_week)

        # 计算奖励
        infection_ratio = total_infected / sum(c.total_population() for c in self.cities)
        economy_ratio = weekly_economy / sum(c.economy_base for c in self.cities)
        reward = (
            -100 * infection_ratio
            - 5 * economy_ratio
            - 0.1 * np.sum(action_params)
        )  # 修复：补全右括号

        self.current_week += 1
        done = self.current_week >= self.max_weeks

        return self.get_state(), reward, done, {}


# Dueling Double DQN网络（新增）
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if x.dim() == 1:  # 如果输入是 1 维的（单个样本）
            x = x.unsqueeze(0)  # 添加批次维度
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# DQN智能体（替换原策略梯度方法）
class DQNAgent:
    def __init__(self, env, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.state_dim = len(env.get_state())
        self.action_dim = env.num_actions

        self.policy_net = DuelingDQN(self.state_dim, self.action_dim)
        self.target_net = DuelingDQN(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.update_target_every = 100
        self.steps_done = 0

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 修复：将列表转换为 numpy 数组后再转换为张量
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # 计算当前Q值
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值（Double DQN）
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1))
            target_q = rewards + (1 - dones) * self.gamma * next_q.squeeze()

        # 计算损失
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        # 优化步骤
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # 更新目标网络
        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.update_epsilon()

cities_config_model = [
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
migration_matrix_model = np.array([
    [0.00, 0.05, 0.02],
    [0.04, 0.00, 0.03],
    [0.01, 0.06, 0.00],
])
episodes = 100
print_interval = 50
# 训练过程（修改为DQN训练逻辑）
def train_with_seeds(seeds=[42, 123, 999]):
    all_rewards = []
    cities_config = cities_config_model  # 保持原配置
    migration_matrix =migration_matrix_model  # 保持原配置

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        env = SEIREnv(cities_config, migration_matrix)
        agent = DQNAgent(env)
        episode_rewards = []

        for ep in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.learn()
                total_reward += reward
                state = next_state

            episode_rewards.append(total_reward)

            if ep % print_interval == 0:
                avg_reward = np.mean(episode_rewards[-print_interval:])
                print(f"Seed {seed} | Episode {ep} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

        all_rewards.append(episode_rewards)

    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    for i, rewards in enumerate(all_rewards):
        plt.plot(rewards, label=f'Seed {seeds[i]}')
    plt.title('Training Progress with D3QN')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.show()
    return agent


# 增强可视化与统计分析
def analyze_results(agent, num_simulations=10):
    env = SEIREnv(cities_config_model, migration_matrix_model)
    city_types = ['1', '2', '3']
    stats = {t: {'infection': [], 'economy': [], 'actions': []} for t in city_types}

    for _ in range(num_simulations):
        state = env.reset()
        done = False

        while not done:
            action_idx = agent.select_action(state, training=False)
            action_params = env.actions[action_idx]

            # 记录每个城市的数据
            for i, city in enumerate(env.cities):
                stats[city_types[i]]['infection'].append(city.I)
                stats[city_types[i]]['economy'].append(sum(list(env.economy_history[city.name])[-7:]))
                stats[city_types[i]]['actions'].append(action_params)

            state, _, done, _ = env.step(action_idx)

    # 可视化统计结果
    plt.figure(figsize=(15, 8))

    # 感染人数分布
    plt.subplot(2, 2, 1)
    for t in city_types:
        plt.hist(stats[t]['infection'], bins=30, alpha=0.5, label=t)
    plt.title('Infection Distribution')
    plt.legend()

    # 经济损夫分布
    plt.subplot(2, 2, 2)
    for t in city_types:
        plt.hist(np.array(stats[t]['economy']) / 1e4, bins=30, alpha=0.5, label=t)
    plt.title('Economic Loss (10k) Distribution')
    plt.legend()

    # 各城市动作参数分布
    plt.subplot(2, 2, 3)
    param_names = ['Med', 'Home', 'Mask']
    for i, param in enumerate(param_names):
        plt.bar([t + f'_{param}' for t in city_types],
                [np.mean([a[i] for a in stats[t]['actions']]) for t in city_types],
                alpha=0.6)
    plt.title('Average Action Parameters')

    plt.tight_layout()
    plt.show()


# 运行训练与可视化
trained_agent = train_with_seeds()
analyze_results(trained_agent)