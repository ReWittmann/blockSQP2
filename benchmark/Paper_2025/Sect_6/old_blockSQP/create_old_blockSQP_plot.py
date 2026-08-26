from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
import sys
sys.path.append(str(cD.parents[1]/Path("common")))
import dolan_more_plots


if __name__ == '__main__':
    benchmark_data = dolan_more_plots.parse_benchmark_file(str(cD/Path('out_old_blockSQP_experiments/blockSQP_it_2026-08-26_00_42_43_226804.txt')))
    dolan_more_plots.plot(benchmark_data, 
                          colors = ['tab:red', 'tab:blue','tab:olive','tab:green','tab:cyan'],
                          linestyles = [
                                        (0, (1, 0.5)),
                                        (0, (3, 1, 1, 1, 1, 1)),
                                        (0, (3, 2, 1, 2)),
                                        (0, (1, 2)), 
                                        # '--',
                                        (0, (2,1)),
                                        '-'],
                          linewidths = [2.0, 2.0, 2.0, 2.0, 2.0],
                          dpi = 250
                          )


