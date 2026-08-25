from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
import sys
sys.path.append(str(cD.parent/Path("common")))
import dolan_more_plots


if __name__ == '__main__':
    benchmark_outfile = 'blockSQP2_it_2026-08-24_21_14_39_206282.txt'
    benchmark_data = dolan_more_plots.parse_benchmark_file(str(cD / Path(f'out_conv_strategy_comparison/{benchmark_outfile}')))
    dolan_more_plots.plot(benchmark_data)
