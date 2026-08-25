from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
import sys
sys.path.append(str(cD.parent/Path("common")))
import dolan_more_plots


if __name__ == '__main__':
    blockSQP2_data = dolan_more_plots.parse_benchmark_file(str(cD/Path('out_blockSQP2_experiments/blockSQP2_it_2026-08-24_21_38_43_575833.txt')))
    blockSQP2_data = blockSQP2_data['scaling heuristic']
    
    fatrop_data = dolan_more_plots.parse_benchmark_file(str(cD/Path('out_fatrop_experiments/fatrop_it_2026-08-24_21_51_54_893410.txt')))
    fatrop_data = fatrop_data['fatrop, exact Hessian']
    
    ipopt_data = dolan_more_plots.parse_benchmark_file(str(cD/Path('out_ipopt_experiments/ipopt_it_2026-08-24_22_24_46_794635.txt')))
    ipopt_data = ipopt_data['ipopt, exact Hessian']
    
    UNO_data = dolan_more_plots.parse_benchmark_file(str(cD/Path('out_UNO_experiments/UNO_it_2026-08-24_22_41_44_519543.txt')))
    UNO_data = UNO_data['UNO (filtersqp preset), exact Hessian']
    
    benchmark_data = {
        'blockSQP2': blockSQP2_data,
        'fatrop': fatrop_data,
        'ipopt': ipopt_data,
        'UNO (filtersqp)': UNO_data,
        # 'blockSQP': blockSQP_data
        }
    
    dolan_more_plots.plot(benchmark_data, xlim = 100)


