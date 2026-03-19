import ctypes
from typing import Union, Optional

class Options:
    def __init__(self,
                 maxiters: int = 100,
                 eps: float = 1.0e-16,
                 inf: float = float('inf'),
                 print_level: int = 2,
                 result_print_color: int = 2,
                 debug_level: int = 0,
                 opt_tol: float = 1.0e-6,
                 feas_tol: float = 1.0e-6,
                 sparse: bool = True,
                 enable_rest: bool = True,
                 lim_mem: bool = True,
                 mem_size: int = 20,
                 block_hess: int = 1,
                 hess_approx: str = "SR1",
                 fallback_approx: str = "BFGS",
                 initial_hess_scale: float = 1.0,
                 sizing: str = "OL",
                 fallback_sizing: str = "COL",
                 COL_eps: float = 0.1,
                 COL_tau_1: float = 0.5,
                 COL_tau_2: float = 1.0e4,
                 OL_eps: float = 1.0e-4,
                 BFGS_damping_factor: float = 1 / 3,
                 conv_strategy: int = 1,
                 max_conv_QPs: int = 4,
                 enable_linesearch: bool = True,
                 max_linesearch_steps: int = 10,
                 max_consec_reduced_steps: int = 8,
                 max_consec_skipped_updates: int = 100,
                 skip_first_linesearch: bool = False,
                 max_SOC: int = 3,
                 qpsol: str = "qpOASES",
                 qpsol_options: Optional['qpOASESoptions'] = None,
                 max_QP_it: int = 5000,
                 max_QP_secs: float = 3600.0,
                 max_extra_steps: int = 0,
                 max_filter_overrides: int = 2,
                 par_QPs: bool = False,
                 enable_QP_cancellation: bool = True,
                 automatic_scaling: bool = False,
                 enable_premature_termination: bool = False,
                 indef_delay: int = 3):
        self.maxiters = maxiters
        self.eps = eps
        self.inf = inf
        self.print_level = print_level
        self.result_print_color = result_print_color
        self.debug_level = debug_level
        self.opt_tol = opt_tol
        self.feas_tol = feas_tol
        self.sparse = sparse
        self.enable_rest = enable_rest
        self.lim_mem = lim_mem
        self.mem_size = mem_size
        self.block_hess = block_hess
        self.hess_approx = hess_approx
        self.fallback_approx = fallback_approx
        self.initial_hess_scale = initial_hess_scale
        self.sizing = sizing
        self.fallback_sizing = fallback_sizing
        self.COL_eps = COL_eps
        self.COL_tau_1 = COL_tau_1
        self.COL_tau_2 = COL_tau_2
        self.OL_eps = OL_eps
        self.BFGS_damping_factor = BFGS_damping_factor
        self.conv_strategy = conv_strategy
        self.max_conv_QPs = max_conv_QPs
        self.enable_linesearch = enable_linesearch
        self.max_linesearch_steps = max_linesearch_steps
        self.max_consec_reduced_steps = max_consec_reduced_steps
        self.max_consec_skipped_updates = max_consec_skipped_updates
        self.skip_first_linesearch = skip_first_linesearch
        self.max_SOC = max_SOC
        self.qpsol = qpsol
        self.qpsol_options = qpsol_options
        self.max_QP_it = max_QP_it
        self.max_QP_secs = max_QP_secs
        self.max_extra_steps = max_extra_steps
        self.max_filter_overrides = max_filter_overrides
        self.par_QPs = par_QPs
        self.enable_QP_cancellation = enable_QP_cancellation
        self.automatic_scaling = automatic_scaling
        self.enable_premature_termination = enable_premature_termination
        self.indef_delay = indef_delay

class qpOASESoptions:
    def __init__(self, sparsityLevel: int = 2, printLevel: int = 0, terminationTolerance: float = 5.0e6 * 2.221e-16):
        self.sparsityLevel = sparsityLevel
        self.printLevel = printLevel
        self.terminationTolerance = terminationTolerance
        

def create_cxx_options(opts: Options):
    BSQP = opts.BSQP
    
    # Initialize the SQPoptions_obj (C++ side object)
    SQPoptions_obj = BSQP.create_SQPoptions()

    # Set options
    BSQP.SQPoptions_set_eps(SQPoptions_obj, ctypes.c_double(opts.eps))
    BSQP.SQPoptions_set_inf(SQPoptions_obj, ctypes.c_double(opts.inf))
    BSQP.SQPoptions_set_print_level(SQPoptions_obj, ctypes.c_int(opts.print_level))
    BSQP.SQPoptions_set_result_print_color(SQPoptions_obj, ctypes.c_int(opts.result_print_color))
    BSQP.SQPoptions_set_debug_level(SQPoptions_obj, ctypes.c_int(opts.debug_level))

    # Termination criteria
    BSQP.SQPoptions_set_opt_tol(SQPoptions_obj, ctypes.c_double(opts.opt_tol))
    BSQP.SQPoptions_set_feas_tol(SQPoptions_obj, ctypes.c_double(opts.feas_tol))
    BSQP.SQPoptions_set_enable_premature_termination(SQPoptions_obj, ctypes.c_char(opts.enable_premature_termination))
    BSQP.SQPoptions_set_max_extra_steps(SQPoptions_obj, ctypes.c_int(opts.max_extra_steps))

    # Line search heuristics
    BSQP.SQPoptions_set_max_filter_overrides(SQPoptions_obj, ctypes.c_int(opts.max_filter_overrides))

    # Derivative evaluation
    BSQP.SQPoptions_set_sparse(SQPoptions_obj, ctypes.c_char(opts.sparse))

    # Restoration phase
    BSQP.SQPoptions_set_enable_rest(SQPoptions_obj, ctypes.c_char(opts.enable_rest))

    # Full/limited memory quasi newton
    BSQP.SQPoptions_set_lim_mem(SQPoptions_obj, ctypes.c_char(opts.lim_mem))
    BSQP.SQPoptions_set_mem_size(SQPoptions_obj, ctypes.c_int(opts.mem_size))

    # Hessian approximation
    BSQP.SQPoptions_set_block_hess(SQPoptions_obj, ctypes.c_int(opts.block_hess))
    
    # Convert hess_approx to C string and set
    hess_approx_str = str(opts.hess_approx)
    BSQP.SQPoptions_set_hess_approx(SQPoptions_obj, ctypes.c_char_p(hess_approx_str.encode('utf-8')))
    
    # Handle `qpsol_options` if provided
    if opts.qpsol == "qpOASES":
        QPsolver_options_obj = BSQP.create_qpOASES_options()
        if opts.qpsol_options is not None:
            BSQP.qpOASES_options_set_sparsityLevel(QPsolver_options_obj, ctypes.c_int(opts.qpsol_options.sparsityLevel))
            BSQP.qpOASES_options_set_printLevel(QPsolver_options_obj, ctypes.c_int(opts.qpsol_options.printLevel))
            BSQP.qpOASES_options_set_terminationTolerance(QPsolver_options_obj, ctypes.c_double(opts.qpsol_options.terminationTolerance))
        BSQP.SQPoptions_set_qpsol_options(SQPoptions_obj, QPsolver_options_obj)

    return SQPoptions_obj, QPsolver_options_obj