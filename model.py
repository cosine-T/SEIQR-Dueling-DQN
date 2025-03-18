import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体


class City:
    # 保持原有City类实现不变
    def __init__(self, name, S0, E0, I0, R0, beta_day, gamma_day, sigma_day, city_type, economy_base):
        self.name = name
        self.type = city_type
        self.S = S0
        self.E = E0
        self.I = I0
        self.R = R0
        self.economy_base = economy_base

        self.beta = beta_day
        self.gamma = gamma_day
        self.sigma = sigma_day

        if self.type == 1:
            self.beta *= 1.2
            self.sigma *= 1.1
        elif self.type == 2:
            self.sigma *= 0.9
            self.gamma *= 0.9
        elif self.type == 3:
            self.beta *= 0.8
            self.sigma *= 0.8
            self.gamma *= 0.7

    def total_population(self):
        return self.S + self.E + self.I + self.R

    def update(self, params):
        beta_eff = self.beta * (1 - params['mask']) * (1 - params['home'])
        gamma_eff = self.gamma * (1 + params['med'])
        sigma_eff = self.sigma

        N = self.total_population() + 1e-9

        dS = beta_eff * self.S * self.I / N
        dE = dS - sigma_eff * self.E
        dI = sigma_eff * self.E - gamma_eff * self.I
        dR = gamma_eff * self.I

        self.S = max(self.S - dS, 1e-9)
        self.E = max(self.E + dE, 1e-9)
        self.I = max(self.I + dI, 1e-9)
        self.R = max(self.R + dR, 1e-9)


def calculate_economic_loss(city, mask, home, med):
    coefficients = {
        1: (0.5, 0.2, 0.3),
        2: (0.4, 0.3, 0.2),
        3: (0.3, 0.4, 0.5)
    }
    c1, c2, c3 = coefficients[city.type]
    return city.economy_base * (c1 * home + c2 * mask + c3 * med)


def do_migration(cities, migration_matrix, alpha, week):
    num_cities = len(cities)
    weekly_smoothing = 1 - np.exp(-0.3 * week)

    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            exchange_rate = min(migration_matrix[i, j], migration_matrix[j, i]) * (1 - alpha)
            exchange_rate *= weekly_smoothing
            base_i = cities[i].total_population() * exchange_rate / 7
            base_j = cities[j].total_population() * exchange_rate / 7
            exchange_base = min(base_i, base_j)

            for src, dest in [(i, j), (j, i)]:
                total = cities[src].total_population()
                if total == 0: continue

                ratios = [cities[src].S / total,
                          cities[src].E / total,
                          cities[src].I / total,
                          cities[src].R / total]

                mig_total = exchange_base
                S_mig = mig_total * ratios[0]
                E_mig = mig_total * ratios[1]
                I_mig = mig_total * ratios[2]
                R_mig = mig_total * ratios[3]

                cities[src].S -= S_mig
                cities[src].E -= E_mig
                cities[src].I -= I_mig
                cities[src].R -= R_mig

                cities[dest].S += S_mig
                cities[dest].E += E_mig
                cities[dest].I += I_mig
                cities[dest].R += R_mig


def get_strategy_params(strategy, cities):
    if strategy == 1:  # 无干预
        return {'mask': 0, 'home': 0, 'med': 0}, 0
    elif strategy == 2:  # 持续干预
        return {'mask': 0.6, 'home': 0.4, 'med': 0.3}, 0.7
    elif strategy == 3:  # 动态响应
        total_infected = sum(city.I for city in cities)
        if total_infected > 50000:
            return {'mask': 0.5, 'home': 0.3, 'med': 0.2}, 0.6
        else:
            return {'mask': 0, 'home': 0, 'med': 0}, 0
    else:
        return {'mask': 0, 'home': 0, 'med': 0}, 0


def simulate_seir(strategy, weeks=365):
    cities = [
        City("平原城市", 999800, 150, 50, 0, 0.20, 0.1, 0.5, 1, 1e5),
        City("丘陵城市", 799900, 80, 20, 0, 0.18, 0.12, 0.4, 2, 8e4),
        City("高原城市", 499900, 50, 50, 0, 0.25, 0.08, 0.3, 3, 5e4)
    ]

    migration_matrix = np.array([
        [0.00, 0.05, 0.02],
        [0.04, 0.00, 0.03],
        [0.01, 0.06, 0.00],
    ])

    sir_history = {c.name: {'S': [], 'E': [], 'I': [], 'R': []} for c in cities}
    economy_history = {c.name: [] for c in cities}

    for w in range(weeks):
        params, alpha = get_strategy_params(strategy, cities)

        for city in cities:
            city.update(params)
            loss = calculate_economic_loss(city,params['mask'],params['home'],params['med'])
            economy_history[city.name].append(loss)

        do_migration(cities, migration_matrix, alpha, w)

        for city in cities:
            sir_history[city.name]['S'].append(city.S)
            sir_history[city.name]['E'].append(city.E)
            sir_history[city.name]['I'].append(city.I)
            sir_history[city.name]['R'].append(city.R)

    return sir_history, economy_history


def plot_results(results, econ_results):
    plt.figure(figsize=(14, 10))

    # 绘制感染曲线
    plt.subplot(2, 1, 1)
    styles = {1: '--', 2: '-.', 3: '-'}
    colors = {'平原城市': 'red', '丘陵城市': 'blue', '高原城市': 'green'}

    for strat in results:
        for city in results[strat]:
            I = results[strat][city]['I']
            plt.plot(I, linestyle=styles[strat], color=colors[city],
                     label=f"{city} - 策略{strat}")

    plt.title('SEIR模型 - 感染人数变化')
    plt.xlabel('周数')
    plt.ylabel('感染人数')
    plt.grid(True)
    plt.legend()

    # 绘制经济影响
    plt.subplot(2, 1, 2)
    for strat in econ_results:
        for city in econ_results[strat]:
            cumulative = np.cumsum(econ_results[strat][city])
            plt.plot(cumulative, linestyle=styles[strat], color=colors[city],
                     label=f"{city} - 策略{strat}")

    plt.title('累计经济损失')
    plt.xlabel('周数')
    plt.ylabel('经济损失')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    infection_results = {}
    economy_results = {}

    for strategy in [1, 2, 3]:
        sir, econ = simulate_seir(strategy)
        infection_results[strategy] = sir
        economy_results[strategy] = econ

    plot_results(infection_results, economy_results)