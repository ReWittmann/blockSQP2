from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
import sys
sys.path.append(str(cD.parent/Path("common")))
import dolan_more_plots



stylenames = {
     'loosely dotted',        (0, (1, 10)),
     'dotted',                (0, (1, 5)),
     'densely dotted',        (0, (1, 1)),

     'long dash with offset', (5, (10, 3)),
     'loosely dashed',        (0, (5, 10)),
     'dashed',                (0, (5, 5)),
     'densely dashed',        (0, (5, 1)),

     'loosely dashdotted',    (0, (3, 10, 1, 10)),
     'dashdotted',            (0, (3, 5, 1, 5)),
     'densely dashdotted',    (0, (3, 1, 1, 1)),

     'dashdotdotted',         (0, (3, 5, 1, 5, 1, 5)),
     'loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10)),
     'densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))
     }


# if __name__ == '__main__':
benchmark_outfile = 'blockSQP2_it_2026-08-24_21_14_39_206282.txt'
benchmark_data = dolan_more_plots.parse_benchmark_file(str(cD / Path(f'out_conv_strategy_comparison/{benchmark_outfile}')))
# dolan_more_plots.plot(benchmark_data, 
#                         colors = ['tab:red', 'tab:blue','tab:green','tab:olive','tab:cyan'],
#                       # linestyles = [(0, (3, 1, 1, 1, 1, 1)),(0, (3, 2, 1, 2)),(0, (1, 0.9)), '--','-'],
#                       linewidths = [2.0]*3 + [2.0] + [2.0],
#                       dpi = 200
#                       )
dolan_more_plots.plot(benchmark_data, 
                      colors = ['tab:red', 'tab:blue','tab:olive','tab:green','tab:cyan'],
                      linestyles = [(0, (3, 1, 1, 1, 1, 1)),
                                    # (0, (3, 2, 1, 2)),
                                    (0, (1, 0.9)), 
                                    '--',
                                    (0, (2,1.15)),
                                    '-'],
                      linewidths = [2.0, 2.0, 2.0, 2.0, 2.0],
                      dpi = 250,
                      xticks = [1.0,2.0,3.0,4.0,6.0,10.0]
                      )
