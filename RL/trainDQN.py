import numpy as np
import torch
from modelDQN import SEIQREnv, DQNAgent
from SEIQR import cities_config_model, migration_matrix_model, weather_data_model




episodes = 200
print_interval = 5
##################################
#  4.  训练/测试流程示例
##################################
def save_models(agents, city_types, path='saved_models'):
    import os
    os.makedirs(path, exist_ok=True)
    for agent, c_type in zip(agents, city_types):
        torch.save(agent.policy_net.state_dict(), f'{path}/policy_net_{c_type}.pth')


if __name__ == "__main__":
    # 假设有 3 个城市的配置

    # 环境初始化
    env = SEIQREnv(
        cities_config=cities_config_model,
        migration_matrix=migration_matrix_model,
        action_file='actions.xlsx',  # 请确保有此 Excel 文件
        time_window=5,
        max_weeks=52,
        weather_data=weather_data_model
    )

    # 创建多智能体，每个城市一个
    agents = [DQNAgent(env) for _ in range(env.num_cities)]

    # 训练参数

    episode_rewards = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # 各智能体选动作（这里示例：每个智能体都用同一个 state；
            # 如果你想拆分城市状态，可将环境拆为多个子Env或者拆分state）
            action_indices = [agent.select_action(state) for agent in agents]

            next_state, reward, done, info = env.step(action_indices)

            # 记录并学习
            for i, agent in enumerate(agents):
                # 每个智能体各自存储自己的 reward
                agent.remember(state, action_indices[i], info['city_rewards'][i], next_state, done)
                agent.learn()

            total_reward += reward
            state = next_state

        episode_rewards.append(total_reward)

        if ep % print_interval == 0:
            avg_reward = np.mean(episode_rewards[-print_interval:])
            print(f"Episode {ep}, Avg Reward: {avg_reward:.2f}")

    # 保存模型
    save_models(agents, city_types=[c['city_type'] for c in cities_config_model])
