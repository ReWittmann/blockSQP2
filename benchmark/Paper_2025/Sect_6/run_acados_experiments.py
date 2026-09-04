#See https://github.com/acados/acados on how to install acados and 
#the python package acados_template

import acados_models as acmo
import time

# f - works fully, 
# r - works or gets close with reduced number of stages, 
# p - gets close to solution, but terminates with error
# n - does not work
# nc - does not work with condensing

benchmarks = [
    # acmo.setup_apollo_ocp,                      # n
    # acmo.setup_batch_distillation_ocp,          # n
    acmo.setup_batch_reactor_ocp,               # f
    acmo.setup_batch_reactor_oed_ocp,           # p
    # acmo.setup_calcium_oscillation_ocp,         # n
    acmo.setup_cart_pendulum_ocp,               # r, f
    # acmo.setup_cart_pendulum_ocp_2,             # n
    acmo.setup_catalyst_mixing_ocp,             # r, f
    acmo.setup_catalyst_mixing_oed_ocp,         # r, f but takes long to just-in-time compile.
    # acmo.setup_cushioned_oscillation_ocp,       # r, p
    acmo.setup_dielectrophoretic_ocp,           # f
    acmo.setup_D_Onofrio_ocp,                   # f
    acmo.setup_D_Onofrio_ocp_2,                 # f
    acmo.setup_D_Onofrio_ocp_3,                 # f
    acmo.setup_D_Onofrio_ocp_4,                 # f
    acmo.setup_ducted_fan_ocp,                  # f
    acmo.setup_egerstedt_ocp,                   # p
    acmo.setup_electric_car_ocp,                # f, nc
    # acmo.setup_fermenter_ocp,                   # n
    acmo.setup_goddard_ocp,                     # p
    acmo.setup_hang_glider_ocp,                 # f
    # acmo.setup_hanging_chain_ocp,               # n
    acmo.setup_lotka_ocp,                       # f
    acmo.setup_lotka_oed_ocp,                   # r, p
    acmo.setup_lotka_competitive_ocp,           # f, nc
    acmo.setup_lotka_competitive_ocp_2,         # f, nc
    acmo.setup_lotka_shared_ocp,                # f, nc
    acmo.setup_lotka_shared_ocp_2,              # r, f, nc
    acmo.setup_lotka_shared_oed_ocp,            # r, p
    # acmo.setup_ocean_ocp,                       # n
    acmo.setup_particle_steering_ocp,           # r, p
    acmo.setup_quadrotor_ocp,                   # f, nc
    acmo.setup_satellite_ocp,                   # f, nc
    acmo.setup_three_tank_ocp,                  # r, f
    acmo.setup_time_optimal_car_ocp,            # r, p
    acmo.setup_tubular_reactor_ocp              # f
    ]

setuptimes = []
runtimes = []
it = []
for ocp_creator in benchmarks:
    tm1 = time.time()
    ocp_solver = ocp_creator()
    t0 = time.time()
    ocp_solver.solve()
    t1 = time.time()
    ocp_solver.print_statistics()
    
    setuptimes.append(t0 - tm1)
    runtimes.append(t1 - t0)
    it.append(ocp_solver.get_stats("nlp_iter"))
    