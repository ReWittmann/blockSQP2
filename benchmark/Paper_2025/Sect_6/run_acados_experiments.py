#See https://github.com/acados/acados on how to install acados and 
#the python package acados_template

import acados_models as acmo
import time


benchmarks = [
    # acmo.setup_batch_reactor_ocp,               #
    # acmo.setup_cart_pendulum_ocp,
    # acmo.setup_catalyst_mixing_ocp,
    # acmo.setup_catalyst_mixing_oed_ocp,         # Works, but takes long to just-in-time compile.
    # acmo.setup_dielectrophoretic_ocp,           #
    # acmo.setup_D_Onofrio_ocp,                   #
    # acmo.setup_D_Onofrio_ocp_2,                 #
    # acmo.setup_D_Onofrio_ocp_3,                 #
    # acmo.setup_D_Onofrio_ocp_4,                 #
    # acmo.setup_ducted_fan_ocp,                  #
    # acmo.setup_egerstedt_ocp,                   
    # acmo.setup_electric_car_ocp,                #
    # acmo.setup_hang_glider_ocp,                 #
    # acmo.setup_lotka_ocp,
    # acmo.setup_lotka_oed_ocp,
    # acmo.setup_lotka_competitive_ocp,
    # acmo.setup_lotka_competitive_ocp_2,
    # acmo.setup_lotka_shared_ocp,
    acmo.setup_satellite_ocp,
    # acmo.setup_three_tank_ocp,
    # acmo.setup_time_optimal_car_ocp,
    # acmo.setup_tubular_reactor_ocp
    ]

times = []
it = []
for ocp_creator in benchmarks:
    ocp_solver = ocp_creator()
    t0 = time.time()
    ocp_solver.solve()
    t1 = time.time()
    ocp_solver.print_statistics()
    
    times.append(t1 - t0)
    it.append(ocp_solver.get_stats("nlp_iter"))
    