import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体


class City:
    def __init__(self, name, city_type, population, squared_km2,
                 initcial_exposed_rate, initcial_infected_rate,
                 economy_base, bed_base):
        # 环境与传播参数
        self.ah_a = 0.8947065898679676
        self.ah_c = 0.02890372039629151
        self.ah_score = 61.70051869599825
        self.tmp_c = 18.57608998569208
        self.tmp_score = 0.3827618737395368
        self.tmp_a = -0.21151705827632955
        self.sigma_rho =0.06454950351559771
        self.rho_c = 1.4711351635054517
        self.rho_a = 0.9717321214728396
        self.eta_score = 0.2
        self.epsilon_0 = 0.125153167

        # 传播动力学参数
        self.beta0 = 0.211581379
        self.gamma0 = 0.246464429
        self.lambda0 = 0.0656252708302435
        self.kappa0 = 0.577415497
        self.d = 0.1

        self.name = name
        self.type = city_type
        # 注意 population 若以“万人”为单位，这里乘以 1e4 得到实际人数
        self.S = population * (1 - initcial_exposed_rate - initcial_infected_rate) * 1e4
        self.E = population * initcial_exposed_rate * 1e4
        self.I = population * initcial_infected_rate * 1e4
        self.Q = 0
        self.R = 0

        self.economy_base = economy_base
        self.rho = population / squared_km2 * 10  # 人口密度，乘 10 仅作缩放
        self.B_base = bed_base  # 床位数

    def total_population(self):
        return self.S + self.E + self.I + self.Q + self.R

    def update(self, weather, params):
        """
        根据每日天气和策略参数更新 SEIQR 状态
        weather: {'ah': float, 'tmean': float}
        params: {'mask': float, 'home': float, 'med': float}
        """
        ah = weather['ah']
        tmp = weather['tmean']
        m = params['mask']
        med = params['med']
        home = params['home']

        # 计算环境因子
        H_factor = (self.ah_score * (ah - self.ah_c) ** 2 + self.ah_a) * ((self.tmp_c / tmp) ** self.tmp_score + self.tmp_a)

        density_factor = self.rho_a + self.sigma_rho * (self.rho / self.rho_c)
        mask_factor = 1 - self.eta_score * m * (4 - self.type)

        # 若想考虑所有因子，则：
        beta = self.beta0 * H_factor * density_factor * mask_factor
        # 这里为了演示，把 H_factor 注释起来，仅保留原先的用法:
        #beta = self.beta0 * density_factor * mask_factor
        #beta = self.beta0 * H_factor
        #beta = self.beta0 * H_factor * density_factor
        #beta = self.beta0



        # 计算医疗资源影响
        B_effective = self.B_base * (1 + med*15) * self.type # 医疗资源影响恢复率
        I_val = max(self.I, 1e-9)
        resource_ratio = min(0.3, self.lambda0 * B_effective / I_val)
        resource_ratio = 0
        gamma = self.gamma0 * (1 + resource_ratio)

        # 隔离强度
        delta = home * (4 - self.type) * 0.2 # home参数直接影响隔离率
        #delta = 0

        # 微分方程
        N = self.total_population()
        dS = -beta * self.S * (self.I + self.kappa0 * self.E) / (N + 1e-9)
        dE = beta * self.S * (self.I + self.kappa0 * self.E) / (N + 1e-9) - self.epsilon_0 * self.E
        dI = self.epsilon_0 * self.E - (gamma + delta) * self.I
        dQ = delta * self.I - self.d * self.Q
        dR = gamma * self.I + self.d * self.Q

        # 更新状态（非负约束）
        self.S = max(self.S + dS, 0)
        self.E = max(self.E + dE, 0)
        self.I = max(self.I + dI, 0)
        self.Q = max(self.Q + dQ, 0)
        self.R = max(self.R + dR, 0)


def calculate_economic_loss(city, mask, home, med):
    """经济影响计算"""
    coefficients = {
        1: (15, 0.1, 0.3),  # 平原城市：居家影响大
        2: (12, 0.12,0.3),  # 丘陵城市：均衡影响
        3: (9, 0.15, 0.5)  # 高原城市：医疗投入影响大
    }
    c1, c2, c3 = coefficients[city.type]
    # 每日损失，这里简单按 city.economy_base 比例缩放
    return city.economy_base * (c1 * home + c2 * mask + c3 * med) / 7  # 按天计算


def do_migration(cities, migration_matrix):
    """人口迁移逻辑（每周执行一次）"""
    num_cities = len(cities)

    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            # 计算基础迁移率
            exchange_rate = min(migration_matrix[i, j], migration_matrix[j, i])
            base_i = cities[i].total_population() * exchange_rate
            base_j = cities[j].total_population() * exchange_rate
            exchange_base = min(base_i, base_j)

            # 双向迁移
            for src, dest in [(i, j), (j, i)]:
                total = cities[src].total_population()
                if total <= 0:
                    continue

                # 计算迁移比例
                ratios = [
                    cities[src].S / total,
                    cities[src].E / total,
                    cities[src].I / total,
                    cities[src].R / total
                ]

                mig_total = exchange_base
                S_mig = mig_total * ratios[0]
                E_mig = mig_total * ratios[1]
                I_mig = mig_total * ratios[2]
                R_mig = mig_total * ratios[3]

                # 更新人口
                cities[src].S -= S_mig
                cities[src].E -= E_mig
                cities[src].I -= I_mig
                cities[src].R -= R_mig

                cities[dest].S += S_mig
                cities[dest].E += E_mig
                cities[dest].I += I_mig
                cities[dest].R += R_mig


def get_strategy_params(strategy, cities):
    """给定策略编号，生成不同的干预参数 (mask, home, med) """
    params = {}

    if strategy == 1:  # 无干预
        for city in cities:
            params[city.name] = {'mask': 0, 'home': 0, 'med': 0}

    elif strategy == 2:  # 动态响应（示例，阈值可调）
        for city in cities:
            inf_ratio = city.I / (city.total_population())
            if inf_ratio > 0.075 :
                params[city.name] = {'mask': 0.3, 'home': 0.2, 'med': 0.3}
            elif inf_ratio > 0.05 :
                params[city.name] = {'mask': 0.2, 'home': 0.1, 'med': 0.2}
            elif inf_ratio > 0.03 :
                params[city.name] = {'mask': 0.1, 'home': 0.0, 'med': 0.1}
            else:
                params[city.name] = {'mask': 0, 'home': 0, 'med': 0}

    return params


# -------------------------
# 模型配置
# -------------------------
"""
cities_config_model = [
    {
        'name': "平原城市_成都",
        'city_type': 1,
        'population': 2140.3,  # 万
        'squared_km2': 14332.3,
        'init_exposed_rate': 0.002,
        'init_infected_rate': 0.001,
        'economy_base': 19916.98,
        'bed_base': 153663
    },
    {
        'name': "丘陵城市_宜宾",
        'city_type': 2,
        'population': 462.8,
        'squared_km2': 13266.2,
        'init_exposed_rate': 0.002,
        'init_infected_rate': 0.001,
        'economy_base': 3148.08,
        'bed_base': 36215
    },
    {
        'name': "高原城市_阿坝",
        'city_type': 3,
        'population': 82.5,
        'squared_km2': 83016.3,
        'init_exposed_rate': 0.002,
        'init_infected_rate': 0.001,
        'economy_base': 449.63,
        'bed_base': 5381
    }
]
"""

cities_config_model = [
    {
        'name': "ganzi",
        'city_type': 3,
        'population': 110.6,  # 万
        'squared_km2': 149599.3,
        'init_exposed_rate': 0.002,
        'init_infected_rate': 0.001,
        'economy_base': 447.04,
        'bed_base': 5606
    },
    {
        'name': "luzhou",
        'city_type': 2,
        'population': 426.7 ,
        'squared_km2': 12236.2,
        'init_exposed_rate': 0.002,
        'init_infected_rate': 0.001,
        'economy_base': 2406.10,
        'bed_base': 34608
    },
    {
        'name': "mianyang",
        'population': 491.1,
        'city_type': 2,
        'squared_km2': 20248.4,
        'init_exposed_rate': 0.002,
        'init_infected_rate': 0.001,
        'economy_base': 3350.29,
        'bed_base': 40666
    }
]


migration_matrix_model = np.array([
    [0.00, 0.05, 0.02],
    [0.04, 0.00, 0.03],
    [0.01, 0.06, 0.00]
])


def load_weather_data(cities_config):
    file_name = 'weatherSichuan.xlsx'
    if os.path.exists(file_name):
        df = pd.read_excel(file_name)
    elif os.path.exists(os.path.join('..', file_name)):
        df = pd.read_excel(os.path.join('..', file_name))
    # df = pd.read_excel('weatherSichuan.xlsx')
    # 假设 Excel 全部都是同一个城市的记录
    weather_data = {}
    for cfg in cities_config:
        city_name = cfg['name']
        # 如果 Excel 里并没有按不同城市拆分数据，而是一个文件只对应一个城市，
        # 那就直接把整张表当做 weather_data[city_name]
        day_list = []
        for _, row in df.iterrows():
            T = row['tmean']  # 温度（摄氏度）
            P = row['pressure']  # 压力（百帕）
            RH = row['humidity']  # 相对湿度（百分比）
            e_s = 6.112 * np.exp((17.67 * T) / (T + 243.5))
            e_a = (RH / 100) * e_s
            AH = (e_a * 2.1674) / P
            day_list.append({
                'ah': AH,
                'tmean':T
            })
        weather_data[city_name] = day_list
    #print(weather_data)
    return weather_data



def get_population(city_name,cities_config_model):
    # 获取配置中城市的总人口 (万人)
    for city in cities_config_model:
        if city['name'] == city_name:
            return city['population']
    return None

def get_eco_base(city_name,cities_config_model):
    # 获取配置中城市的总人口 (万人)
    for city in cities_config_model:
        if city['name'] == city_name:
            return city['economy_base']
    return None


def simulate_seiQR(strategy, weeks=52):
    """
    每个策略下的 S/E/I/Q/R 演化，以及每日经济损失。
    这里一个“周”包含 7 天，每天都用 Excel 中的一行天气进行更新。
    若 Excel 天气行数不足，可循环取。
    """
    # 初始化城市实例
    cities = []
    for cfg in cities_config_model:
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

    # 从 Excel 读取各城市的天气数据
    weather_data = load_weather_data(cities_config_model)

    # 用于记录 SEIQR 过程（按周保存一次）
    history = {c.name: {'S': [], 'E': [], 'I': [], 'Q': [], 'R': []} for c in cities}
    # 用于记录每日经济损失
    economy_history = {c.name: [] for c in cities}
    param_history = {c.name: {'mask': [], 'home': [], 'med': []} for c in cities}

    # 开始模拟
    total_days = weeks * 7

    for w in range(weeks):
        # 每周之前都先获取一下当前策略参数（动态策略可能依赖本周初状态）
        params = get_strategy_params(strategy, cities)
        # 逐天更新
        for day in range(7):
            # 对每个城市做当日更新
            for city in cities:
                # 取出当前城市 day_index 对应的天气，若超出则循环取
                #print(city.name)

                city_weather_list = weather_data[city.name]
                daily_weather = city_weather_list[w]
                # 更新状态
                city.update(daily_weather, params[city.name])
                if day == 6:
                    # 计算当日经济损失
                    loss = calculate_economic_loss(
                        city,
                        mask=params[city.name]['mask'],
                        home=params[city.name]['home'],
                        med=params[city.name]['med']
                    )
                    economy_history[city.name].append(loss)
                    param_history[city.name]['mask'].append(params[city.name]['mask'])
                    param_history[city.name]['home'].append(params[city.name]['home'])
                    param_history[city.name]['med'].append(params[city.name]['med'])
        # 一周结束后做一次迁移
        do_migration(cities, migration_matrix_model)
        for city in cities:
            history[city.name]['S'].append(city.S)
            history[city.name]['E'].append(city.E)
            history[city.name]['I'].append(city.I)
            history[city.name]['Q'].append(city.Q)
            history[city.name]['R'].append(city.R)

    return history, economy_history, param_history


def plot_results(results, econ_results):
    """
    results[strategy][city_name] = {'S':[...], 'E':[...], 'I':[...], 'Q':[...], 'R':[...] (按周记录)
    econ_results[strategy][city_name] = [loss_day1, loss_day2, ...] (按天记录)
    """

    # 先画感染人数折线图（万人）
    plt.figure(figsize=(14, 10))

    # 不同城市可用不同颜色(可自行按需设置，也可统一不设颜色)
    color_map = {
        "平原城市_成都": 'red',
        "丘陵城市_宜宾": 'blue',
        "高原城市_阿坝": 'green'
    }
    # 不同策略不同线型
    styles = {
        1: '-',
        2: '-.',
    }

    # --- 子图1：感染人数走势 ---
    plt.subplot(2, 1, 1)
    for strat in sorted(results.keys()):
        city_dict = results[strat]  # { city_name: {'S': [...], 'E': [...], 'I': [...], ...} }
        for city_name, compartment_dict in city_dict.items():
            infected_list = compartment_dict['I']  # 每周的 I
            population = get_population(city_name,cities_config_model)  # 转换为实际人数
            # 转为万人
            infected_in_10k = [i_val / 10000 for i_val in infected_list]
            infection_rate = [i_val / 10000 /population for i_val in infected_list]

            plt.plot(
                infection_rate,
                linestyle=styles[strat],
                color=color_map.get(city_name, 'black'),
                label=f"{city_name}-策略{strat}"
            )

    plt.title('各策略感染人数对比（单位：万人）')
    plt.xlabel('周数')
    plt.ylabel('感染率')
    plt.grid(True)
    plt.legend()

    # --- 子图2：经济损失走势(累计) ---
    #"""
        
    plt.subplot(2, 1, 2)

    for strat in sorted(econ_results.keys()):
        city_econ = econ_results[strat]  # { city_name: [day_loss1, day_loss2, ...] }
        for city_name, loss_list in city_econ.items():
            eco_base = get_eco_base(city_name,cities_config_model)  # 转换为实际人数
            loss_rate = [i_val / eco_base for i_val in loss_list]

            plt.plot(
                loss_rate,
                linestyle=styles[strat],
                color=color_map.get(city_name, 'black'),
                label=f"{city_name}-策略{strat}"
            )

    plt.title('累计经济损失（与 economy_base 同单位，如：亿元）')
    plt.xlabel('周数')
    plt.ylabel('经济损失 (累计)')
    plt.grid(True)
    plt.legend()
    #"""
    plt.tight_layout()
    plt.show()


def plot_parameter_changes(param_history):
    """绘制三行一列的参数变化图"""
    city_names = [cfg['name'] for cfg in cities_config_model]
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

    # 定义可视化配置
    param_config = {
        'mask': {'label': '口罩强度', 'color': 'blue'},
        'home': {'label': '居家隔离', 'color': 'green'},
        'med': {'label': '医疗投入', 'color': 'red'}
    }

    for idx, city_name in enumerate(city_names):
        ax = axs[idx]
        weeks = range(1, len(param_history[city_name]['mask']) + 1)

        # 绘制三条曲线
        for param in ['mask', 'home', 'med']:
            values = param_history[city_name][param]
            ax.plot(
                weeks, values,
                label=param_config[param]['label'],
                color=param_config[param]['color'],
                linewidth=2,
                linestyle='--'
            )

        # 设置图表属性
        ax.set_title(f'{city_name} 干预参数动态变化', fontsize=12)
        ax.set_xlabel('周数', fontsize=10)
        ax.set_ylabel('参数强度', fontsize=10)
        ax.set_ylim(-0.05, 0.35)  # 统一纵轴范围
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    infection_results = {}
    economy_results = {}
    params_results = {}

    # 可根据需要添加更多策略
    for strategy in [1, 2]:
        sir, econ,para  = simulate_seiQR(strategy, weeks=52)
        infection_results[strategy] = sir
        economy_results[strategy] = econ
        params_results[strategy] = para

    plot_results(infection_results, economy_results)
    plot_parameter_changes(para)

    # 将结果写入Excel文件
    with pd.ExcelWriter('simulation_results.xlsx') as writer:
        for strategy in [1, 2]:
            # 处理感染数据
            infection_data = infection_results[strategy]
            infection_rows = []
            for city_name in infection_data:
                data = infection_data[city_name]
                for week in range(len(data['S'])):
                    infection_rows.append({
                        '策略': strategy,
                        '城市': city_name,
                        '周数': week + 1,
                        'S': data['S'][week],
                        'E': data['E'][week],
                        'I': data['I'][week],
                        'Q': data['Q'][week],
                        'R': data['R'][week],
                        'I_rate':data['I'][week]/get_population(city_name,cities_config_model)/10000,
                        'R_rate':data['R'][week]/get_population(city_name,cities_config_model)/10000,
                    })
            df_infection = pd.DataFrame(infection_rows)
            df_infection.to_excel(writer, sheet_name=f'策略{strategy}-感染', index=False)

            # 处理经济数据
            economy_data = economy_results[strategy]
            economy_rows = []
            for city_name in economy_data:
                losses = economy_data[city_name]
                for week, loss in enumerate(losses):
                    economy_rows.append({
                        '策略': strategy,
                        '城市': city_name,
                        '周数': week + 1,
                        '经济损失': loss
                    })
            df_economy = pd.DataFrame(economy_rows)
            df_economy.to_excel(writer, sheet_name=f'策略{strategy}-经济', index=False)

            # 处理参数数据
            param_data = params_results[strategy]
            param_rows = []
            for city_name in param_data:
                params = param_data[city_name]
                for week in range(len(params['mask'])):
                    param_rows.append({
                        '策略': strategy,
                        '城市': city_name,
                        '周数': week + 1,
                        '口罩强度': params['mask'][week],
                        '居家隔离': params['home'][week],
                        '医疗投入': params['med'][week]
                    })
            df_params = pd.DataFrame(param_rows)
            df_params.to_excel(writer, sheet_name=f'策略{strategy}-参数', index=False)
