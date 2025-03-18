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
        """创建带随机参数的城市实例"""
        cities = []
        for config in self.cities_config:
            city_config = config.copy()
            # 增强参数随机性
            city_config['beta_day'] = np.random.uniform(*config['beta_day']) * 1.2
            city_config['gamma_day'] = np.random.uniform(*config['gamma_day']) * 0.8
            city_config['sigma_day'] = np.random.uniform(*config['sigma_day'])
            city_config['I0'] = np.random.randint(50, 200)
            cities.append(City(**city_config))
        return cities

    def reset(self):
        """完全重置环境状态"""
        self.current_week = 0
        self.cities = self._create_cities()  # 每次生成新参数
        self.init_history()
        self.last_action = np.zeros(3 * self.num_cities)
        self.ema_action = np.zeros(3 * self.num_cities)
        return self.get_state()

    def init_history(self):
        """初始化历史记录队列"""
        self.I_history = {city.name: deque(maxlen=self.time_window) for city in self.cities}
        self.economy_history = {city.name: deque(maxlen=self.time_window) for city in self.cities}
        for city in self.cities:
            for _ in range(self.time_window):
                self.I_history[city.name].append(city.I)
                self.economy_history[city.name].append(0)

    def get_state(self):
        """构建包含所有城市特征的状态向量"""
        state_features = []
        total_pop = sum(c.total_population() for c in self.cities)
        for i, city in enumerate(self.cities):
            infected = np.array(self.I_history[city.name]) / city.total_population()
            economy = np.array(list(self.economy_history[city.name])) / city.economy_base
            city_action = self.last_action[i * 3:(i + 1) * 3]
            features = [
                *infected.tolist(),
                *economy.tolist(),
                city.type / 3.0,
                *city_action
            ]
            state_features.extend(features)
        return np.array(state_features, dtype=np.float32)

    def step(self, action):
        """环境步进函数"""
        # 动作平滑处理
        smoothed_action = 0.8 * self.last_action + 0.2 * np.clip(action, 0, 1)
        self.last_action = smoothed_action
        self.ema_action = (1 - self.ema_alpha) * self.ema_action + self.ema_alpha * smoothed_action

        # 拆分动作到各城市
        city_actions = np.reshape(smoothed_action, (self.num_cities, 3))
        total_reward = 0
        weekly_economy = 0
        weekly_infections = 0

        # 模拟7天发展
        for day in range(7):
            for i, (city, params) in enumerate(zip(self.cities, city_actions)):
                # 更新城市参数
                city.update({
                    'mask': params[0],
                    'home': params[1],
                    'med': params[2]
                })

                # 计算城市个体奖励
                infection_ratio = city.I / city.total_population()
                economy_loss = calculate_economic_loss(city, *params)
                action_cost = np.sum(params)

                # 个性化奖励权重
                city_weight = 1 + (city.economy_base / 1e5)  # 经济规模越大权重越高
                reward = (
                        -15 * (infection_ratio ** 1.5) * city_weight  # 非线性感染惩罚
                        - economy_loss / city.economy_base  # 标准化经济损失
                        - 0.05 * action_cost  # 干预成本
                )
                total_reward += reward

                self.economy_history[city.name].append(economy_loss)
                weekly_economy += economy_loss
                weekly_infections += city.I

            # 更新感染历史
            for city in self.cities:
                self.I_history[city.name].append(city.I)

        # 动态迁移策略
        total_infected = sum(c.I for c in self.cities)
        alpha = 0.6 * (total_infected / 5e5) ** 2  # 非线性迁移系数
        do_migration(self.cities, self.migration_matrix, alpha, self.current_week)

        self.current_week += 1
        done = self.current_week >= self.max_weeks

        return self.get_state(), total_reward, done, {}


class PolicyNetwork(nn.Module):
    """支持多城市独立策略的网络结构"""

    def __init__(self, input_dim, num_cities=3):
        super().__init__()
        self.num_cities = num_cities
        self.terrain_embed = nn.Embedding(4, 64)  # 地形嵌入层

        # 共享特征提取层
        self.shared_feature = nn.Sequential(
            nn.Linear((input_dim // num_cities) - 1, 512),
            nn.SiLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.LayerNorm(256)
        )

        # 城市独立策略头
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256 + 64, 128),
                nn.Tanh(),
                nn.Linear(128, 3),
                nn.Sigmoid()
            ) for _ in range(num_cities)
        ])

        # 可学习的动作方差
        self.log_std = nn.Parameter(torch.ones(3 * num_cities) * 0.5 )

    def forward(self, x):
        batch_size = x.shape[0]
        city_states = x.view(batch_size, self.num_cities, -1)

        mus = []
        for i in range(self.num_cities):
            # 地形特征提取
            terrain = (city_states[:, i, -1] * 3).round().long().clamp(0, 3)
            embed = self.terrain_embed(terrain)

            # 特征拼接
            features = torch.cat([
                self.shared_feature(city_states[:, i, :-1]),
                embed
            ], dim=1)

            # 生成策略
            mus.append(self.heads[i](features))

        mu = torch.cat(mus, dim=1)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mu, std)


class RLAgent:
    def __init__(self, env, lr=1e-4, gamma=0.99):
        self.env = env
        self.gamma = gamma
        state_dim = len(env.get_state())
        self.policy = PolicyNetwork(state_dim, env.num_cities)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=500)
        self.entropy_bonus = 0.02  # 熵奖励系数
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state, explore=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        dist = self.policy(state)
        action = dist.sample() if explore else dist.mean
        log_prob = dist.log_prob(action).sum()
        entropy = dist.entropy().mean()

        # 添加熵奖励促进探索
        self.saved_log_probs.append(log_prob)
        self.rewards.append(self.entropy_bonus * entropy.item())
        return action.squeeze(0).detach().numpy()

    def update_policy(self):
        """带优势归一化的策略更新"""
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        policy_loss = torch.stack([
            -log_prob * R for log_prob, R in zip(self.saved_log_probs, returns)
        ]).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        self.scheduler.step()

        del self.rewards[:]
        del self.saved_log_probs[:]


def train_with_seeds(seeds=[42, 123, 999], num_episodes=800):
    """多种子训练函数"""
    all_rewards = []
    policy_variances = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        env = SEIREnv(cities_config, migration_matrix)
        agent = RLAgent(env)
        episode_rewards = []
        variances = []

        for ep in range(num_episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.rewards.append(reward)
                total_reward += reward
                state = next_state

            # 记录策略方差
            with torch.no_grad():
                sample_actions = [agent.select_action(env.reset()) for _ in range(10)]
                variances.append(np.var(sample_actions))

            agent.update_policy()
            episode_rewards.append(total_reward)

            if ep % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:] if ep >= 50 else episode_rewards)
                print(f"Seed {seed} | Ep {ep} | Avg.R: {avg_reward:.1f} | Var: {variances[-1]:.3f}")

        all_rewards.append(episode_rewards)
        policy_variances.append(variances)

    # 训练过程可视化
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    for i, rewards in enumerate(all_rewards):
        axs[0].plot(rewards, label=f'Seed {seeds[i]}')
    axs[0].set_title('Training Curves')
    axs[0].set_ylabel('Total Reward')
    axs[0].legend()

    for i, var in enumerate(policy_variances):
        axs[1].plot(var, label=f'Seed {seeds[i]}')
    axs[1].set_title('Policy Variance')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Action Variance')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

    return agent


def visualize_policy(agent, num_simulations=3):
    """增强的可视化函数"""
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12})

    for sim in range(num_simulations):
        env = SEIREnv(cities_config, migration_matrix)
        state = env.reset()
        done = False
        records = {
            'infections': [[] for _ in env.cities],
            'economy': [[] for _ in env.cities],
            'actions': []
        }

        while not done:
            with torch.no_grad():
                action = agent.select_action(state, explore=False)
            records['actions'].append(action)

            # 记录各城市状态
            for i, city in enumerate(env.cities):
                records['infections'][i].append(city.I / city.total_population())
                records['economy'][i].append(
                    calculate_economic_loss(city, *action[i * 3:(i + 1) * 3]) / city.economy_base
                )

            state, _, done, _ = env.step(action)

        # 创建可视化面板
        fig = plt.figure(figsize=(18, 6 * num_simulations))
        for i, city in enumerate(env.cities):
            # 感染趋势
            ax = plt.subplot(num_simulations, 3, sim * 3 + 1)
            for cid, city_data in enumerate(records['infections']):
                ax.plot(city_data, label=env.cities[cid].name)
            ax.set_title(f'Sim {sim + 1} - Infection Rates')
            ax.legend()

            # 经济损失
            ax = plt.subplot(num_simulations, 3, sim * 3 + 2)
            for cid, econ_data in enumerate(records['economy']):
                ax.plot(econ_data, label=env.cities[cid].name)
            ax.set_title(f'Sim {sim + 1} - Economic Loss')
            ax.legend()

            # 分城市策略
            ax = plt.subplot(num_simulations, 3, sim * 3 + 3)
            actions = np.array(records['actions'])
            city_actions = actions[:, i * 3:(i + 1) * 3]
            for j in range(3):
                ax.plot(city_actions[:, j], label=['Mask', 'Home', 'Med'][j])
            ax.set_title(f'{env.cities[i].name} Policies')
            ax.set_ylim(0, 1)
            ax.legend()

        plt.tight_layout()
        plt.show()


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

if __name__ == "__main__":
    trained_agent = train_with_seeds(seeds=[42, 123, 456], num_episodes=800)
    visualize_policy(trained_agent, num_simulations=3)