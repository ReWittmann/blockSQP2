from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
import sys
sys.path.append(str(cD.parent/Path("common")))
import dolan_more_plots


# if __name__ == '__main__':
benchmark_outfile = 'blockSQP2_it_2026-08-24_21_38_43_575833.txt'
benchmark_data = dolan_more_plots.parse_benchmark_file(str(cD / Path(f'out_scaling_comparison/{benchmark_outfile}')))
dolan_more_plots.plot(benchmark_data, 
                      xlim = 1.6, 
                      linestyles = ['-', (0,(1,0.5))], 
                      colors = ['tab:cyan', 'tab:red'], 
                      xticks = [1.0,1.1,1.2,1.3,1.4,1.5,1.6],
                      dpi = 250)