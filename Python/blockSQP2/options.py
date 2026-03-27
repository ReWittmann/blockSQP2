from ctypes import c_int, c_char_p, c_char, c_double
from typing import Optional
from .cxxwrappers import CXXobjCreator, CXXobjHolder

class qpOASESoptions(CXXobjCreator):
    def __init__(self, sparsityLevel: int = 2, printLevel: int = 0, terminationTolerance: float = 5.0e6 * 2.221e-16):
        self.sparsityLevel = sparsityLevel
        self.printLevel = printLevel
        self.terminationTolerance = terminationTolerance
    def create_cxx_obj(self):
        BSQP = self.BSQP
        
        cxx_obj = BSQP.create_qpOASES_options()
        BSQP.qpOASES_options_set_sparsityLevel(cxx_obj, c_int(self.sparsityLevel))
        BSQP.qpOASES_options_set_printLevel(cxx_obj, c_int(self.printLevel))
        BSQP.qpOASES_options_set_terminationTolerance(cxx_obj, c_double(self.terminationTolerance))
        
        deleter = lambda ptr: self.BSQP.delete_QPsolver_options(ptr)
        return CXXobjHolder(cxx_obj, deleter)

class Options(CXXobjCreator):
    def __init__(self,
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
                 BFGS_damping_factor: float = 1./3.,
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
    
    def create_cxx_obj(self):
        BSQP = self.BSQP        
        cxx_obj = BSQP.create_SQPoptions()        

        # Set options
        BSQP.SQPoptions_set_eps(cxx_obj, c_double(self.eps))
        BSQP.SQPoptions_set_inf(cxx_obj, c_double(self.inf))
        BSQP.SQPoptions_set_print_level(cxx_obj, c_int(self.print_level))
        BSQP.SQPoptions_set_result_print_color(cxx_obj, c_int(self.result_print_color))
        BSQP.SQPoptions_set_debug_level(cxx_obj, c_int(self.debug_level))

        # Termination criteria
        BSQP.SQPoptions_set_opt_tol(cxx_obj, c_double(self.opt_tol))
        BSQP.SQPoptions_set_feas_tol(cxx_obj, c_double(self.feas_tol))
        BSQP.SQPoptions_set_enable_premature_termination(cxx_obj, c_char(self.enable_premature_termination))
        BSQP.SQPoptions_set_max_extra_steps(cxx_obj, c_int(self.max_extra_steps))

        # Line search heuristics
        BSQP.SQPoptions_set_max_filter_overrides(cxx_obj, c_int(self.max_filter_overrides))

        # Derivative evaluation
        BSQP.SQPoptions_set_sparse(cxx_obj, c_char(self.sparse))

        # Restoration phase
        BSQP.SQPoptions_set_enable_rest(cxx_obj, c_char(self.enable_rest))

        # Full/limited memory quasi newton
        BSQP.SQPoptions_set_lim_mem(cxx_obj, c_char(self.lim_mem))
        BSQP.SQPoptions_set_mem_size(cxx_obj, c_int(self.mem_size))

        # Hessian approximation
        BSQP.SQPoptions_set_block_hess(cxx_obj, c_int(self.block_hess))
        ret = BSQP.SQPoptions_set_hess_approx(cxx_obj, c_char_p(self.hess_approx.encode('utf-8')))
        if ret > 0:
            error_message = BSQP.get_error_message()
            raise Exception(error_message.value.decode('utf-8'))
        
        ret = BSQP.SQPoptions_set_fallback_approx(cxx_obj, c_char_p(self.fallback_approx.encode('utf-8')))
        if ret > 0:
            error_message = BSQP.get_error_message()
            raise Exception(error_message.value.decode('utf-8'))
        
        BSQP.SQPoptions_set_indef_delay(cxx_obj, c_int(self.indef_delay))
        
        # Hessian sizing
        BSQP.SQPoptions_set_initial_hess_scale(cxx_obj, c_double(self.initial_hess_scale))
        ret = BSQP.SQPoptions_set_sizing(cxx_obj, c_char_p(self.sizing.encode('utf-8')))
        if ret > 0:
            error_message = BSQP.get_error_message()
            raise Exception(error_message.value.decode('utf-8'))
        
        ret = BSQP.SQPoptions_set_fallback_sizing(cxx_obj, c_char_p(str(self.fallback_sizing).encode('utf-8')))
        if ret > 0:
            error_message = BSQP.get_error_message()
            raise Exception(error_message.value.decode('utf-8'))
        
        BSQP.SQPoptions_set_COL_eps(cxx_obj, c_double(self.COL_eps))
        BSQP.SQPoptions_set_COL_tau_1(cxx_obj, c_double(self.COL_tau_1))
        BSQP.SQPoptions_set_COL_tau_2(cxx_obj, c_double(self.COL_tau_2))
        BSQP.SQPoptions_set_OL_eps(cxx_obj, c_double(self.OL_eps))
        
        # Quasi-Newton
        BSQP.SQPoptions_set_BFGS_damping_factor(cxx_obj, c_double(self.BFGS_damping_factor))
        
        # Convexification strategy
        BSQP.SQPoptions_set_conv_strategy(cxx_obj, c_int(self.conv_strategy))
        BSQP.SQPoptions_set_max_conv_QPs(cxx_obj, c_int(self.max_conv_QPs))
        BSQP.SQPoptions_set_par_QPs(cxx_obj, c_char(self.par_QPs))
        BSQP.SQPoptions_set_enable_QP_cancellation(cxx_obj, c_char(self.enable_QP_cancellation))
        
        # Scaling
        BSQP.SQPoptions_set_automatic_scaling(cxx_obj, c_char(self.automatic_scaling))
        
        # Filter line search
        BSQP.SQPoptions_set_enable_linesearch(cxx_obj, c_char(self.enable_linesearch))
        BSQP.SQPoptions_set_max_linesearch_steps(cxx_obj, c_int(self.max_linesearch_steps))
        BSQP.SQPoptions_set_max_consec_reduced_steps(cxx_obj, c_int(self.max_consec_reduced_steps))
        BSQP.SQPoptions_set_max_consec_skipped_updates(cxx_obj, c_int(self.max_consec_skipped_updates))
        BSQP.SQPoptions_set_skip_first_linesearch(cxx_obj, c_int(self.skip_first_linesearch))
        BSQP.SQPoptions_set_max_SOC(cxx_obj, c_int(self.max_SOC))
        
        # qpsol and qpsol_options below
        BSQP.SQPoptions_set_max_QP_it(cxx_obj, c_int(self.max_QP_it))
        BSQP.SQPoptions_set_max_QP_secs(cxx_obj, c_double(self.max_QP_secs))
        
        # Handle `qpsol_options` if provided
        if self.qpsol == "qpOASES":
            QPopts = self.qpsol_options if self.qpsol_options is not None else qpOASESoptions()
            QPsolver_options_hld = QPopts.create_cxx_obj()
            BSQP.SQPoptions_set_qpsol_options(cxx_obj, QPsolver_options_hld.ptr)
        
        deleter = lambda ptr: self.BSQP.delete_SQPoptions(ptr)
        return CXXobjHolder(cxx_obj, deleter, QPsolver_options_hld)






