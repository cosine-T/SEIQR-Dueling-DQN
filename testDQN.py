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
    def __init__(self, cities_config, migration_matrix, action_file='actions.xlsx', time_window=7, max_weeks=52):
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
        self.I_history = {city.name: [city.I] for city in self.cities}
        self.economy_history = {city.name: [0] for city in self.cities}

    def get_state(self):
        state_features = []
        for city in self.cities:
            infected = np.array(self.I_history[city.name][-self.time_window:]) / city.total_population()
            economy = np.array(self.economy_history[city.name][-self.time_window:]) / city.economy_base
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
        )

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
        if x.dim() == 1:
            x = x.unsqueeze(0)
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

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1))
            target_q = rewards + (1 - dones) * self.gamma * next_q.squeeze()

        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.update_epsilon()

# 城市配置和迁移矩阵保持不变
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

migration_matrix_model = np.array([
    [0.00, 0.05, 0.02],
    [0.04, 0.00, 0.03],
    [0.01, 0.06, 0.00],
])

episodes = 100
print_interval = 50

def train_with_seeds(seeds=[42, 123, 999]):
    all_rewards = []
    cities_config = cities_config_model
    migration_matrix = migration_matrix_model

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

    plt.figure(figsize=(10, 6))
    for i, rewards in enumerate(all_rewards):
        plt.plot(rewards, label=f'Seed {seeds[i]}')
    plt.title('Training Progress with D3QN')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.show()
    return agent

def analyze_results(agent, num_simulations=10):
    env = SEIREnv(cities_config_model, migration_matrix_model)
    city_types = ['1', '2', '3']
    max_weeks = env.max_weeks

    # 初始化数据结构
    infection_over_time = {t: np.zeros((num_simulations, max_weeks)) for t in city_types}
    economy_over_time = {t: np.zeros((num_simulations, max_weeks)) for t in city_types}
    action_over_time = np.zeros((num_simulations, max_weeks, 3))  # med, home, mask

    for sim in range(num_simulations):
        state = env.reset()
        done = False
        while not done:
            action_idx = agent.select_action(state, training=False)
            next_state, reward, done, _ = env.step(action_idx)
            action_params = env.actions[action_idx]

            current_week = env.current_week - 1  # 调整周索引
            if current_week >= max_weeks or current_week < 0:
                break

            action_over_time[sim, current_week, :] = action_params

            for city in env.cities:
                t = str(city.city_type)
                # 感染人数为当前周最后一天的值
                infection = city.I
                # 计算该周的经济损失
                start_day = current_week * 7
                end_day = (current_week + 1) * 7
                economy_history = env.economy_history[city.name]
                if end_day > len(economy_history):
                    end_day = len(economy_history)
                economy = sum(economy_history[start_day:end_day])

                infection_over_time[t][sim, current_week] = infection
                economy_over_time[t][sim, current_week] = economy

            state = next_state

    # 计算平均值
    avg_infection = {t: np.mean(infection_over_time[t], axis=0) for t in city_types}
    avg_economy = {t: np.mean(economy_over_time[t], axis=0) for t in city_types}
    avg_actions = np.mean(action_over_time, axis=0)  # 各参数的平均值

    # 可视化
    plt.figure(figsize=(15, 12))

    # 感染人数时间序列
    plt.subplot(3, 2, 1)
    for t in city_types:
        plt.plot(avg_infection[t], label=f'Type {t}')
    plt.title('Average Infection Over Time')
    plt.xlabel('Week')
    plt.ylabel('Infected Population')
    plt.legend()

    # 经济损失时间序列
    plt.subplot(3, 2, 2)
    for t in city_types:
        plt.plot(avg_economy[t], label=f'Type {t}')
    plt.title('Average Economic Loss Over Time')
    plt.xlabel('Week')
    plt.ylabel('Economic Loss')
    plt.legend()

    # 干预强度时间序列
    plt.subplot(3, 2, 3)
    for param_idx, param_name in enumerate(['Med', 'Home', 'Mask']):
        plt.plot(avg_actions[:, param_idx], label=param_name)
    plt.title('Average Intervention Intensity Over Time')
    plt.xlabel('Week')
    plt.ylabel('Intensity')
    plt.legend()

    # 感染人数分布
    plt.subplot(3, 2, 4)
    for t in city_types:
        plt.hist(infection_over_time[t].flatten(), bins=30, alpha=0.5, label=t)
    plt.title('Infection Distribution')
    plt.legend()

    # 经济损失分布
    plt.subplot(3, 2, 5)
    for t in city_types:
        plt.hist(economy_over_time[t].flatten() / 1e4, bins=30, alpha=0.5, label=t)
    plt.title('Economic Loss (10k) Distribution')
    plt.legend()

    # 动作参数分布
    plt.subplot(3, 2, 6)
    param_names = ['Med', 'Home', 'Mask']
    for i, param in enumerate(param_names):
        plt.bar([f'Type {t}' for t in city_types],
                [np.mean(action_over_time[:, :, i])]*3,  # 全局平均值
                alpha=0.6)
    plt.title('Average Action Parameters')
    plt.xticks(rotation=45)
    plt.legend(param_names)

    plt.tight_layout()
    plt.show()

# 运行训练与可视化
trained_agent = train_with_seeds()
analyze_results(trained_agent)