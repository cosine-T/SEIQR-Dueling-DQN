import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from SEIQR import City, calculate_economic_loss, do_migration


class SEIQREnv:
    def __init__(self,
                 cities_config,
                 migration_matrix,
                 action_file='actions.xlsx',
                 time_window=5,
                 max_weeks=52,
                 weather_data=None):
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
        self.weather_data = weather_data if weather_data else {}

        # 读取离散动作空间
        self.actions_df = pd.read_excel(action_file)
        self.actions = self.actions_df[['mask', 'home', 'med']].values
        self.num_actions = len(self.actions)
        self.num_cities = len(cities_config)

        self.reset()

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
        self.last_action_params = [np.zeros(3) for _ in range(self.num_cities)]
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
            features = [
                *last_I.tolist(),
                *last_Q.tolist(),
                *last_econ.tolist(),
                city.type / 3.0
            ]
            # 将上一次动作存入状态
            features.extend(self.last_action_params[i].tolist())

            state_features.extend(features)

        return np.array(state_features, dtype=np.float32)

    def step(self, action_indices):
        """
        修改后的 step：每周只更新一次城市状态、只做一次迁移
        """
        # 记录上一周的动作
        prev_action_params = [p.copy() for p in self.last_action_params]

        # 当前动作
        action_params_list = [self.actions[idx] for idx in action_indices]
        self.last_action_params = action_params_list

        # 本周经济损失（不再按天累加，直接本周一次）
        weekly_losses = [0.0] * self.num_cities

        # ---- 每个城市：本周更新一次 ----
        for i, city in enumerate(self.cities):
            # 取天气数据
            if city.name in self.weather_data:
                weather = self.weather_data[city.name]
            else:
                weather = {'ah': 0.7, 'tmean': 18}  # 若无数据给个默认

            params = {
                'med': action_params_list[i][0],
                'home': action_params_list[i][1],
                'mask':  action_params_list[i][2]
            }

            for day in range(7):
                city.update(weather, params)

                loss = calculate_economic_loss(city, **params)
            # 更新 SEIQR
            #city.update(weather, params)

            # 经济损失
            #loss = calculate_economic_loss(city, **params)
            weekly_losses[i] = loss

        # ---- 每周只做一次迁移 ----
        do_migration(self.cities, self.migration_matrix, self.current_week)

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


            # 动作变化量惩罚
            action_change = np.linalg.norm(action_params_list[i] - prev_action_params[i],ord=1)
            action_strenth = np.sum(action_params_list[i])
            false_alarm01 = ((inf_ratio < 0.01) & (action_strenth > 0))
            false_alarm02 = ((inf_ratio < 0.03) & (action_strenth > 0.2))

            # 简单示例：负向惩罚
            reward_i = - 100 * inf_ratio \
                       - 10 * Q_ratio \
                       - 1 * econ_ratio \
                       - 8 * action_change ** 2 \
                       - 8 * false_alarm01 \
                       - 3 * false_alarm02 \


            city_rewards.append(reward_i)

        total_reward = sum(city_rewards)
        self.current_week += 1
        done = (self.current_week >= self.max_weeks)

        return self.get_state(), total_reward, done, {
            'city_rewards': city_rewards,
            'action_changes': [np.linalg.norm(a - b) for a, b in zip(action_params_list, prev_action_params)]
        }


# -----------------------------
# 2. Dueling DQN 和 DQNAgent (保持不变)
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
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
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
