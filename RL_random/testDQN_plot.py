import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

# 从主模型脚本中，导入你定义好的环境和 Agent 类（若放在同目录可直接引用）
# 假设文件名是 modelDQN.py 或 SEIQR.py 等，你可以根据实际文件名进行修改
from modelDQN import SEIQREnv, DQNAgent
from SEIQR_random import cities_config_model, migration_matrix_model, load_weather_data,get_population,get_eco_base



weather_data = load_weather_data(cities_config_model) # 取对应天气，但其实这个用不到

def load_trained_agents(env, model_path='saved_models'):
    """
    为每个城市加载已经训练好的模型权重，
    返回一个按城市顺序排列的 DQNAgent list。
    """
    agents = []
    for i, cfg in enumerate(cities_config_model):
        agent = DQNAgent(env)
        city_type = cfg['city_type']
        # 加载已经保存的策略网络参数
        model_file = f"{model_path}/policy_net_{city_type}.pth"
        agent.policy_net.load_state_dict(torch.load(model_file))
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        # 测试阶段通常不需要探索，直接用贪心策略
        agent.epsilon = 0.0
        agents.append(agent)
    return agents


def run_evaluation(env, agents, max_weeks=52):
    """
    在给定环境和若干城市智能体(agents)上跑一个完整的流感季(默认 52 周)。
    返回一个保存测试记录的 DataFrame，列包含:
      [city, week, mask, home, med, infection, economic].
    """
    state = env.reset()

    results = []
    done = False
    week = 0

    while not done and week < max_weeks:
        # 每个城市都用同一个 state(拼接了所有城市信息)
        action_indices = []
        for i, agent in enumerate(agents):
            action_idx = agent.select_action(state, training=False)
            action_indices.append(action_idx)

        next_state, reward, done, info = env.step(action_indices)

        # 将当前周每个城市的结果记录下来
        for i, city in enumerate(env.cities):
            city_name = city.name
            mask_val = env.actions[action_indices[i]]['mask']  # [1,3](@ref)
            home_val = env.actions[action_indices[i]]['home']
            med_val = env.actions[action_indices[i]]['med']
            infection = city.I

            economic_loss = env.economy_history[city.name][-1]

            results.append({
                'city': city_name,
                'week': week,
                'mask': mask_val,
                'home': home_val,
                'med':  med_val,
                'infection': infection,
                'economic': economic_loss,
                'infection_rate':infection / get_population(city_name,cities_config_model) / 10000,
                'r':city.R / get_population(city_name,cities_config_model) / 10000
            })

        state = next_state
        week += 1

    df_result = pd.DataFrame(results)
    return df_result


def run_no_intervention(env, max_weeks=52):
    """
    不采取任何干预(mask=0, home=0, med=0)的对照试验。
    在同一个环境里跑 52 周，并记录结果。
    """
    state = env.reset()

    # 因为我们不需要 Agent 动作，这里直接固定动作索引:
    #   1) 要么从 env.actions 中找到 (0,0,0) 所在的行索引
    #   2) 要么绕过 step() 的 action_indices 机制，直接改 city.update(...) 也可以
    #
    # 这里示例方式1：先找到无干预那一行
    #   注意如果 excel 里并没有 (0,0,0), 需要你自行添加一行
    no_int_index = None
    for idx, action in enumerate(env.actions):
        if (action['mask'] == 0.0 and
                action['home'] == 0.0 and
                action['med'] == 0.0):
            no_int_index = idx
            print("找到无干预动作的索引: %d" % no_int_index)
            break

    if no_int_index is None:
        raise ValueError("actions.xlsx 中未找到 (mask=0,home=0,med=0) 的无干预动作，请先添加一行。")

    results = []
    done = False
    week = 0

    while not done and week < max_weeks:
        # 所有城市都选同一个动作: no_int_index
        action_indices = [no_int_index] * env.num_cities

        next_state, reward, done, info = env.step(action_indices)

        # 记录
        for i, city in enumerate(env.cities):
            city_name = city.name
            mask_val = env.actions[no_int_index]['mask']
            home_val = env.actions[no_int_index]['home']
            med_val = env.actions[no_int_index]['med']
            infection = city.I
            economic_loss = env.economy_history[city.name][-1]

            results.append({
                'city': city_name,
                'week': week,
                'mask': mask_val,
                'home': home_val,
                'med':  med_val,
                'infection': infection,
                'economic': economic_loss
            })

        state = next_state
        week += 1

    df_result = pd.DataFrame(results)
    return df_result


def plot_test_results(df_all):
    """
    根据测试结果 df_all 可视化：不同城市的感染人数、策略变化、经济损失等。
    df_all 含列: [city, week, mask, home, med, infection, economic, scenario]
    """
    # 场景列表
    scenarios = df_all['scenario'].unique()
    cities = df_all['city'].unique()

    # 1) 感染人数随时间变化
    plt.figure(figsize=(8, 6))
    for scenario in scenarios:
        df_s = df_all[df_all['scenario'] == scenario]
        for city in cities:
            city_data = df_s[df_s['city'] == city]
            plt.plot(city_data['week'], city_data['infection']/get_population(city,cities_config_model),
                     label=f"{city}-{scenario}")
    plt.title("各城市 Infection 随时间变化 (含对照与RL策略)")
    plt.xlabel("Week")
    plt.ylabel("Infection")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2) 经济损失随时间变化
    plt.figure(figsize=(8, 6))
    for scenario in scenarios:
        df_s = df_all[df_all['scenario'] == scenario]
        for city in cities:
            city_data = df_s[df_s['city'] == city]
            plt.plot(city_data['week'], city_data['economic']/get_eco_base(city,cities_config_model),
                     label=f"{city}-{scenario}")
    plt.title("各城市 每周经济损失 随时间变化 (含对照与RL策略)")
    plt.xlabel("Week")
    plt.ylabel("Weekly Economic Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3) 策略随时间变化（以mask为例）
    df_rl = df_all[df_all['scenario'] == 'RL']

    # 2. 创建画布和三个子图（水平放置），并让它们共享 y 轴
    fig, axs = plt.subplots(3, 1, figsize=(15, 5), sharey=True)

    # 3. 遍历每个城市，在对应的子图中绘制 mask、home、med 曲线
    for i, city in enumerate(cities):
        city_data = df_rl[df_rl['city'] == city]
        axs[i].plot(city_data['week'], city_data['mask'], label='mask')
        axs[i].plot(city_data['week'], city_data['home'], label='home')
        axs[i].plot(city_data['week'], city_data['med'], label='med')

        axs[i].set_title(f"{city} - RL")
        axs[i].set_xlabel("Week")
        axs[i].legend()
        axs[i].grid(True)

    # 4. 设置第一个子图的 y 轴标签（因为其他子图共享 y 轴）
    axs[0].set_ylabel("策略强度值")

    # 5. 调整整体布局并显示
    plt.tight_layout()
    plt.show()


def main():
    # 1. 创建测试环境（与训练时保持一致）
    env = SEIQREnv(
        cities_config=cities_config_model,
        migration_matrix=migration_matrix_model,
        action_file='actions.xlsx',
        time_window=5,
        max_weeks=52,
        weather_data=weather_data
    )

    # 2. 先跑一遍“无干预”做对照
    df_no_int = run_no_intervention(env, max_weeks=52)
    df_no_int['scenario'] = 'NoIntervention'

    # 3. 加载已经训练好的智能体并跑一遍“RL策略”
    #    注意：跑完无干预后，还要 reset环境 或 新建一份环境，保证初始一致
    env_rl = SEIQREnv(
        cities_config=cities_config_model,
        migration_matrix=migration_matrix_model,
        action_file='actions.xlsx',
        time_window=5,
        max_weeks=52,
        weather_data=weather_data
    )
    agents = load_trained_agents(env_rl)
    df_rl = run_evaluation(env_rl, agents, max_weeks=52)
    df_rl['scenario'] = 'RL'

    # 4. 合并两种情形到一个 DataFrame
    df_all = pd.concat([df_no_int, df_rl], ignore_index=True)
    df_all.sort_values(by=['city', 'week', 'scenario'], inplace=True)

    # 导出到 Excel
    output_file = "test_results_with_noInt.xlsx"
    df_all.to_excel(output_file, index=False)
    print(f"测试结果(含无干预对照)已保存到 {output_file}")

    # 5. 可视化
    plot_test_results(df_all)


if __name__ == "__main__":
    main()
