# blockSQP2 -- A structure-exploiting nonlinear programming solver based
#              on blockSQP by Dennis Janka.
# Copyright (C) 2025 by Reinhold Wittmann <reinhold.wittmann@ovgu.de>

# Licensed under the zlib license. See LICENSE for more details.


import sys
from pathlib import Path
try:
    cD = Path(__file__).parent
except:
    cD = Path.cwd()
sys.path += [str(cD.parent)]

import blockSQP2
import numpy as np
import time
from ctypes import *


opts = blockSQP2.Options()
opts.opt_tol = 1.0e-12
opts.feas_tol = 1.0e-12
opts.enable_linesearch = 0
opts.hess_approx = 'BFGS'
opts.fallback_approx = 'BFGS'
opts.sizing = 'None'
opts.fallback_sizing = 'None'
opts.lim_mem = 1
opts.mem_size = 20
opts.block_hess = 1
opts.sparse = False
opts.print_level = 2
opts.debug_level = 0
opts.qpsol = "qpOASES"


stats = blockSQP2.Stats("./")

prob = blockSQP2.Problem(2, 1)
prob.set_blockIndex(np.array([0,1,2],dtype = np.int32))
prob.set_bounds([-np.inf, -np.inf], [np.inf, np.inf], [0.], [0.])
#######
prob.x_start = [10.,10.]
prob.lam_start = [0.,0.,0.]
#######
prob.f = lambda x: x[0]**2 - 0.5*x[1]**2
prob.g = lambda x: x[0] - x[1]
prob.grad_f = lambda x: [2*x[0], -x[1]]
prob.jac_g = lambda x: [[1,-1]]
#######

meth = blockSQP2.Solver(prob, opts, stats)
meth.init()
# time.sleep(0.01)
# print("starting run")

ret = meth.run(100)
# meth.finish()

# time.sleep(0.25)
# print("\nPrimal solution:\n")
# print(np.array(meth.vars.xi))
# print("\nDual solution:\n")
# print(np.array(meth.vars.lam))










# import numpy as np
# import ctypes

# # 1. Create a NumPy array of dtype np.float64 (which is equivalent to double in C)
# np_array = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

# # 2. Create a ctypes pointer (ctypes.POINTER(ctypes.c_double)) for a buffer
# # We'll allocate enough space for the NumPy array to be copied into
# buffer_ptr = (ctypes.c_double * np_array.size)()  # This creates an empty buffer of the same size

# # 3. Copy the data from the NumPy array to the ctypes buffer
# # Use numpy's `ctypes` API to get a pointer to the array
# np_array_ptr = np_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

# # Now copy the data from np_array_ptr to buffer_ptr
# # Since np_array_ptr is already a ctypes pointer, we can copy the data like so:
# ctypes.memmove(buffer_ptr, np_array_ptr, np_array.nbytes)

# # 4. Now buffer_ptr holds the same data as np_array
# # Print the result
# for i in range(np_array.size):
#     print(buffer_ptr[i])  # This prints each element of the buffer



# import numpy as np
# import ctypes

# # 1. Create a NumPy array of dtype np.float64
# np_array = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

# # 2. Create a ctypes buffer (ctypes array) with the same size as the NumPy array
# buffer_ptr = (ctypes.c_double * np_array.size).from_buffer(np_array)

# # 3. Access data from the ctypes buffer
# for i in range(np_array.size):
#     print(buffer_ptr[i])  # Prints: 1.0, 2.0, 3.0, 4.0



# import numpy as np
# import ctypes

# # 1. Create a ctypes buffer (ctypes array)
# buffer_ptr = (ctypes.c_double * 4)(1.0, 2.0, 3.0, 4.0)  # A ctypes array of 4 doubles

# # 2. Wrap the ctypes buffer in a NumPy array using np.frombuffer
# # We need to make sure the ctypes buffer is exposed as a buffer object:
# np_array = np.frombuffer(buffer_ptr, dtype=np.float64, count=4)

# # 3. Access the NumPy array
# print(np_array)  # Output: [1. 2. 3. 4.]

# # Modify the NumPy array and observe the changes in the ctypes buffer
# np_array[0] = 100.0
# print(np_array)  # Output: [100.  2.  3.  4.]
# print(buffer_ptr[0])  # Output: 100.0 (The change is reflected in the ctypes buffer)



# import numpy as np
# import ctypes

# # Let's assume the C function returns a void* pointing to a memory block
# # We'll mock this up with ctypes for demonstration.

# # Create a ctypes buffer manually (for example, from a C function returning a void*)
# buffer_ptr = (ctypes.c_double * 5)(1.0, 2.0, 3.0, 4.0, 5.0)

# # Wrap the ctypes buffer in a NumPy array using np.frombuffer
# np_array = np.frombuffer(buffer_ptr, dtype=np.float64, count=5)

# # Access and modify the NumPy array
# print(np_array)  # Output: [1. 2. 3. 4. 5.]

# np_array[0] = 100.0
# print(np_array)  # Output: [100.  2.  3.  4.  5.]

# # Since the array is a view, changes reflect in the ctypes buffer
# print(buffer_ptr[0])  # Output: 100.0