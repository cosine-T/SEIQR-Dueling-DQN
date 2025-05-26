import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from SEIQR import City, calculate_economic_loss, do_migration, load_weather_data, cities_config_model


class SEIQREnv:
    def __init__(self,
                 cities_config,
                 migration_matrix,
                 weather_data,
                 action_file='actions.xlsx',
                 time_window=5,
                 max_weeks=52):
        """
        :param cities_config: 每个城市的配置
        :param migration_matrix: 迁移矩阵
        :param action_file: 包含 [mask, home, med] 三列的 Excel 文件
        :param time_window: 状态历史窗口
        :param max_weeks: 最大模拟周数
        :param weather_data: dict，每个城市对应 {'ah': xx, 'tmean': xx}
        """
        self.time_window = time_window
        self.max_weeks = max_weeks
        self.current_week = 0
        self.cities_config = cities_config
        self.migration_matrix = migration_matrix
        self.weather_data = weather_data

        # 读取离散动作空间（将每一行转为一个字典）
        actions_df = pd.read_excel(action_file)
        self.actions = []
        for _, row in actions_df.iterrows():
            self.actions.append({
                'med':  float(row['med']),
                'home': float(row['home']),
                'mask': float(row['mask'])
            })

        self.num_actions = len(self.actions)
        self.num_cities = len(cities_config)

        self.reset()
        print(self.actions)

    def _create_cities(self):
        """根据 config 生成 City 实例。"""
        cities = []
        for cfg in self.cities_config:
            city = City(
                name=cfg['name'],
                city_type=cfg['city_type'],
                population=cfg['population'],
                squared_km2=cfg['squared_km2'],
                initcial_exposed_rate=cfg['init_exposed_rate'],
                initcial_infected_rate=cfg['init_infected_rate'],
                economy_base=cfg['economy_base'],
                bed_base=cfg['bed_base']
            )
            cities.append(city)
        return cities

    def reset(self):
        self.current_week = 0
        self.cities = self._create_cities()
        self._init_history()
        # last_action_params 同样改为字典列表，初始化为 0
        self.last_action_params = [
            {'med': 0.0, 'home': 0.0, 'mask': 0.0} for _ in range(self.num_cities)
        ]
        return self.get_state()

    def _init_history(self):
        self.I_history = {city.name: deque(maxlen=self.time_window) for city in self.cities}
        self.Q_history = {city.name: deque(maxlen=self.time_window) for city in self.cities}
        self.economy_history = {city.name: deque(maxlen=self.time_window) for city in self.cities}

        for city in self.cities:
            for _ in range(self.time_window):
                self.I_history[city.name].append(city.I)
                self.Q_history[city.name].append(city.Q)
                self.economy_history[city.name].append(0)

    def get_state(self):
        """
        将所有城市的关键信息拼接成一个 state 向量。
        """
        state_features = []
        for i, city in enumerate(self.cities):
            pop = max(city.total_population(), 1e-9)
            last_I = np.array(self.I_history[city.name]) / pop
            last_Q = np.array(self.Q_history[city.name]) / pop
            last_econ = np.array(self.economy_history[city.name]) / city.economy_base

            # 这里可以继续添加其他特征
            # city.city_type 替换 city.type
            features = [
                *last_I.tolist(),
                *last_Q.tolist(),
                *last_econ.tolist(),
                city.type / 3.0
            ]

            # 将上一次动作存入状态
            # 把字典中的 med, home, mask 按固定顺序拼接
            features.extend([
                self.last_action_params[i]['med'],
                self.last_action_params[i]['home'],
                self.last_action_params[i]['mask']
            ])

            state_features.extend(features)

        return np.array(state_features, dtype=np.float32)

    def step(self, action_indices):
        """
        修改后的 step：每周只更新一次城市状态、只做一次迁移
        """
        # 记录上一周的动作
        prev_action_params = [p.copy() for p in self.last_action_params]

        # 当前动作（字典列表）
        action_params_list = [self.actions[idx] for idx in action_indices]
        #print("本周动作:", action_params_list)
        self.last_action_params = action_params_list

        # 本周经济损失（不再按天累加，直接本周一次）
        weekly_losses = [0.0] * self.num_cities

        # ---- 每个城市：本周更新一次 ----
        for i, city in enumerate(self.cities):
            # 当周动作参数
            params = {
                'mask': action_params_list[i]['mask'],
                'home': action_params_list[i]['home'],
                'med': action_params_list[i]['med']
            }

            # 假设每天使用同样的干预参数
            for day in range(7):
                city_weather_list = self.weather_data[city.name]
                daily_weather = city_weather_list[self.current_week]
                city.update(daily_weather, params)

            # 经济损失
            loss = calculate_economic_loss(city, **params)
            weekly_losses[i] = loss

        # ---- 每周只做一次迁移 ----
        do_migration(self.cities, self.migration_matrix)

        # ---- 更新历史记录 ----
        for i, city in enumerate(self.cities):
            self.I_history[city.name].append(city.I)
            self.Q_history[city.name].append(city.Q)
            self.economy_history[city.name].append(weekly_losses[i])

        # ---- 计算奖励（示例逻辑）----
        city_rewards = []
        for i, city in enumerate(self.cities):
            pop = city.total_population()
            inf_ratio = city.I / (pop + 1e-9)
            Q_ratio = city.Q / (pop + 1e-9)
            econ_ratio = weekly_losses[i] / city.economy_base

            # 动作变化量惩罚 (L1范数示例)
            action_change = abs(action_params_list[i]['med'] - prev_action_params[i]['med']) \
                            + abs(action_params_list[i]['home'] - prev_action_params[i]['home']) \
                            + abs(action_params_list[i]['mask'] - prev_action_params[i]['mask'])
            # 动作强度
            action_strenth = sum(action_params_list[i].values())

            # 示例：特定条件下的“过度反应”惩罚
            false_alarm01 = ((inf_ratio < 0.01) and (action_strenth > 0))
            false_alarm02 = ((inf_ratio < 0.03) and (action_strenth > 0.2))

            reward_i = - 350 * inf_ratio \
                       - 10 * Q_ratio \
                       - 15 * econ_ratio \
                       - 15 * (action_change ** 2) \
                       - 20 * false_alarm01 \
                       - 8 * false_alarm02

            city_rewards.append(reward_i)

        total_reward = sum(city_rewards)
        self.current_week += 1
        done = (self.current_week >= self.max_weeks)

        # 信息输出，可根据需求自行添加
        info = {
            'city_rewards': city_rewards,
            'action_changes': [abs(a['med'] - b['med']) +
                               abs(a['home'] - b['home']) +
                               abs(a['mask'] - b['mask'])
                               for a, b in zip(action_params_list, prev_action_params)]
        }

        return self.get_state(), total_reward, done, info


# -----------------------------
# 2. Dueling DQN 和 DQNAgent (下方内容基本与原始一致，只是注意动作维度)
# -----------------------------
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


class DQNAgent:
    def __init__(self, env, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.state_dim = len(env.get_state())
        # 动作数（仍然是离散的不同组合）
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
        # epsilon-greedy
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

        # 当前 Q
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            # Double DQN
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
