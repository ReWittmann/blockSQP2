import re
import collections
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()

#vibe coded with Gemma 4 31B Instruct

def parse_benchmark_file(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 3:
        return {}

    categories = re.split(r'\s{2,}', lines[0])
    print(categories)
    
    sub_cat_groups = lines[1].split('|')
    sub_categories = []
    for group in sub_cat_groups:
        labels = group.split()
        sub_categories.append(labels)

    result = collections.defaultdict(lambda: collections.defaultdict(dict))
    for line in lines[2:]:
        parts = re.split(r'\s{2,}', line, maxsplit=1)
        row_name = parts[0]
        data_sections = parts[1].split('|')

        for i, section in enumerate(data_sections):
            if i >= len(categories): break
                        
            cat_name = categories[i]
            values = re.split(r'[,\s;]+', section.strip())
            values = [v for v in values if v]

            for j, val in enumerate(values):
                if j < len(sub_categories[i]):
                    sub_label = sub_categories[i][j]
                    result[cat_name][row_name][sub_label] = val
    return result

# benchmark_files = [
#                    # str(cD/Path('blockSQP2_it_2026-08-23_21_49_57_514575.txt')),
#                    ('ipopt', str(cD/Path('out_ipopt_experiments/ipopt_it_2026-08-23_15_19_11_373571.txt'))),
#                    ('UNO (filtersqp)', str(cD/Path('out_UNO_experiments/UNO_it_2026-08-24_13_05_58_262772.txt'))),
#                    ('fatrop', str(cD/Path('out_fatrop_experiments/fatrop_it_2026-08-23_16_42_13_594945.txt')))
#                    ]
# benchmark_data_pre = [(name, parse_benchmark_file(str(cD / Path(f'{benchmark_file}')))) for name, benchmark_file in benchmark_files]


blockSQP2_data = parse_benchmark_file(str(cD/Path('out_blockSQP2_experiments/blockSQP2_it_2026-08-23_21_49_57_514575.txt')))
blockSQP2_data = blockSQP2_data['rr, automatic scaling']

fatrop_data = parse_benchmark_file(str(cD/Path('out_fatrop_experiments/fatrop_it_2026-08-23_16_42_13_594945.txt')))
fatrop_data = fatrop_data['fatrop, exact Hessian']

ipopt_data = parse_benchmark_file(str(cD/Path('out_ipopt_experiments/ipopt_it_2026-08-23_15_19_11_373571.txt')))
ipopt_data = ipopt_data['ipopt, exact Hessian']

UNO_data = parse_benchmark_file(str(cD/Path('out_UNO_experiments/UNO_it_2026-08-24_13_05_58_262772.txt')))
UNO_data = UNO_data['UNO (filtersqp, exact Hessian)']

benchmark_data = {
    'blockSQP2': blockSQP2_data,
    'fatrop': fatrop_data,
    'ipopt': ipopt_data,
    'UNO (filtersqp)': UNO_data,
    # 'blockSQP2': blockSQP_data
    }

def calculate_performance_profile(benchmark_data, metric):
    """
    Calculates the performance profile data for a given metric (e.g., 'mu_N' or 'mu_t').
    """
    categories = list(benchmark_data.keys())
    problems = list(benchmark_data[categories[0]].keys())
    
    print(problems)
    print(categories)
    
    data_matrix = np.zeros((len(problems), len(categories)))
    
    for i, prob in enumerate(problems):
        for j, cat in enumerate(categories):
            val_str = benchmark_data[cat][prob].get(metric, "inf")
            try:
                val = float(val_str.replace('s', ''))
                data_matrix[i, j] = val
            except ValueError:
                data_matrix[i, j] = np.inf

    print(data_matrix)
    
    # 2. Calculate Performance Ratios
    # Find the minimum value for each problem (best solver)
    best_values = np.min(data_matrix, axis=1)
    
    # Avoid division by zero if best_value is 0
    best_values[best_values == 0] = 1e-10 
    
    # Ratio = solver_value / best_value
    ratios = data_matrix / best_values[:, np.newaxis]

    # 3. Calculate Probability P(tau)
    # We evaluate tau from 1 to 100 (or a reasonable range)
    tau = np.geomspace(1, 100, 100)
    profiles = []

    for j in range(len(categories)):
        solver_ratios = ratios[:, j]
        # For each tau, count how many solvers have ratio <= tau
        prob_success = [np.mean(solver_ratios <= t) for t in tau]
        profiles.append(prob_success)

    return tau, profiles, categories

def plot_dolan_more(benchmark_data, xlim = 10):
    metrics = {
        'mu_N': r'Step Count ($\mu_N$)', 
        'mu_t': r'Computation Time ($\mu_t$)'
    }

    tab_colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:orange', 'tab:purple']
    
    for metric in metrics.keys():
        tau, profiles, categories = calculate_performance_profile(benchmark_data, metric)
        
        fig, ax = plt.subplots(dpi = 200)
        for i, category in enumerate(categories):
            ax.plot(tau, profiles[i], label=category, color = tab_colors[i], linewidth=2)
        
        # ax.set_title(f'Performance Profile: {metric_label}')
        # ax.set_xlabel(r'Performance Ratio $\tau$')
        # ax.set_ylabel(r'$P(\tau)$')
        ax.set_xscale('log')
        ax.set_xlim(1, xlim)
        ax.set_ylim(0, 1.05)
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.legend(fontsize = 'x-large')

        plt.tight_layout()
        plt.show()


plot_dolan_more(benchmark_data, xlim = 100)


