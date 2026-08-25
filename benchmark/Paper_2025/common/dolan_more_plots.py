import re
import collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

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

def calculate_performance_profile(benchmark_data, metric):
    categories = list(benchmark_data.keys())
    problems = list(benchmark_data[categories[0]].keys())
    
    for catg in categories[1:]:
        assert(benchmark_data[catg].keys() == benchmark_data[categories[0]].keys())
    
    print(problems)
    print(categories)
    
    data_matrix = np.zeros((len(problems), len(categories)))
    
    for i, prob in enumerate(problems):
        for j, cat in enumerate(categories):
            val_str = benchmark_data[cat][prob][metric]
            try:
                val = float(val_str.replace('s', ''))
                data_matrix[i, j] = val
            except ValueError:
                data_matrix[i, j] = np.inf

    print(data_matrix)
    
    best_values = np.min(data_matrix, axis = 1)
    best_values[best_values == 0] = 1e-10 
    ratios = data_matrix / best_values[:, np.newaxis]

    tau = np.geomspace(1, 100, 100)
    profiles = []

    for j in range(len(categories)):
        solver_ratios = ratios[:, j]
        prob_success = [np.mean(solver_ratios <= t) for t in tau]
        profiles.append(prob_success)

    return tau, profiles, categories

def plot(benchmark_data, xlim = 10):
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
        ax.tick_params(axis='x', labelsize='large')
        ax.tick_params(axis='y', labelsize='large')
        ax.set_xscale('log')
        
        ax.set_xlim(1, xlim)
        ax.set_ylim(0, 1.05)
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.legend(fontsize = 'x-large')


        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
        
        plt.tight_layout()
        plt.show()