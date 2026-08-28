import re
import collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

def parse_benchmark_file(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 3:
        return {}

    categories = re.split(r'\s{2,}', lines[0])
    
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

def calculate_performance_profile(benchmark_data, metric, xlim = 10.):
    categories = list(benchmark_data.keys())
    problems = list(benchmark_data[categories[0]].keys())
    
    for catg in categories[1:]:
        assert(benchmark_data[catg].keys() == benchmark_data[categories[0]].keys())
    
    data_matrix = np.zeros((len(problems), len(categories)))
    
    for i, prob in enumerate(problems):
        for j, cat in enumerate(categories):
            val_str = benchmark_data[cat][prob][metric]
            val = float(val_str.replace('s', ''))
            data_matrix[i, j] = val
    
    best_values = np.min(data_matrix, axis = 1)
    best_values[best_values == 0] = 1e-10 
    ratios = data_matrix / best_values[:, np.newaxis]
    
    
    tau = np.geomspace(1, xlim, 100)
    profiles = []

    for j in range(len(categories)):
        solver_ratios = ratios[:, j]
        prob_success = [np.mean(solver_ratios <= t) for t in tau]
        profiles.append(prob_success)
    
    return tau, profiles, categories

def plot(benchmark_data, xlim = 10, 
         colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple'],
         linestyles = ['-', '-', '-', '-', '-'],
         linewidths = [2.0]*5,
         dpi = 200,
         xticks = None,
         xlabel = None
         ):
    metrics = {
        'mu_N': r'iteration count', 
        'mu_t': r'solution time'
    }

    # tab_colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple']
    # # linestyles = ['-', '-', '-', '-', '-']
    
    for j,metric in enumerate(metrics.keys()):
        tau, profiles, categories = calculate_performance_profile(benchmark_data, metric)
        
        fig, ax = plt.subplots(dpi = dpi)
        for i, category in enumerate(categories):
            ax.plot(tau, profiles[i], label=category, linestyle = linestyles[i], color = colors[i], linewidth=linewidths[i])
        
        if metric == 'mu_N':
            ax.set_xlabel('relative iterations w.r.t. best', fontsize = 'x-large')
        elif metric == 'mu_t':
            ax.set_xlabel('relative solution time w.r.t. best', fontsize = 'x-large')
        else:
            raise Exception('Unknown metric')
        
        # if j == 0:
        ax.set_ylabel(r'fraction of problems solved', fontsize = 'x-large')
        
        
        ax.tick_params(axis='x', labelsize='large')
        ax.tick_params(axis='y', labelsize='large')
        ax.set_xscale('log')
        if xticks is not None:
            ax.set_xticks(xticks)
        
        
        ax.set_xlim(1, xlim)
        ax.set_ylim(0, 1.05)
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.legend(fontsize = 'x-large')

        
        # formatter = ScalarFormatter()
        # formatter.set_scientific(True)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        plt.tight_layout()
        plt.show()