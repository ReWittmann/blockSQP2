**blockSQP2** -- A structure-exploiting nonlinear programming solver based  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; on blockSQP by Dennis Janka.  
Copyright (c) 2023-2025 Reinhold Wittmann <reinhold.wittmann@ovgu.de>  
  
  
### Licensing
blockSQP2 preserves the zlib license of blockSQP, see LICENSE.txt and blockSQP2/LICENSE.txt.  
  
This project includes a modified version of qpOASES <https://github.com/coin-or/qpOASES>, located in blockSQP2/dep and subject to a separate LGPL-2.1 license, see blockSQP2/dep/modified_qpOASES/LICENSE.txt  

When building this project with CMake, the following dependencies are downloaded.  
1. MUMPS <https://mumps-solver.org/index.php?page=home> (CeCILL-C license)  
2. MUMPS-CMake build system <https://github.com/scivision/mumps> (MIT license)  
3. pybind11 <https://github.com/pybind/pybind11> (custom license)  
  
Each license applies to the respective package, and any statement in it regarding compiled code applies to binary files produced by this projects build system that include that compiled code. In addition, BLAS and LAPACK libraries will be linked to or included, e.g. OpenBLAS (BSD-3-Clause license) <https://github.com/OpenMathLib/OpenBLAS>.


### Build requirements
1. Fortran, C and C++-20 compilers, e.g. gfortran, gcc, g++-14
2. The CMake build system <https://cmake.org/>
3. A build system backend (GNU make, Ninja build)
4. BLAS and LAPACK (LAPACKE), likely to be already installed on Linux systems,  
else run &nbsp; `sudo apt install libblas-dev liblapack-dev liblapacke-dev`

## Building
In the command line, navigate to this folder and invoke  
&nbsp;&nbsp; `cmake -B .build ${OPTIONS}`  
&nbsp;&nbsp; `cmake --build .build`

### General build options
These are general CMake options, it is usually not necessary to set them manually.

1. `-DCMAKE_Fortran_COMPILER=...` - choose the Fortran compiler
2. `-DCMAKE_C_COMPILER=...` - choose the C compiler
3. `-DCMAKE_CXX_COMPILER=...` - choose the C++ compiler
4. `-G` ("Unix Makefiles" / Ninja) - choose the build system backend

### blockSQP2 build options
1. `-DPYTHON_INTERFACE= (ON/OFF)` - build Python interface, default ON
2. `-DPYTHON_INTERPRETER= (/PATH/TO/PYTHON_EXECUTABLE)` - optional, choose python interpreter to build for
3. `-DJULIA_INTERFACE= (ON/OFF)` - build blocksqp.jl from the C interface, default ON

See README_WINDOWS.md for details on how to build for Windows with mingW.

### Binaries
The binaries are placed into /blockSQP2/lib or /blockSQP2/bin, /Python/blockSQP2 and /blockSQP2.jl/bin, as required for the Python and Julia interfaces.

## Examples and documentation

### Python interface requirements
blockSQP2 Python requires numpy, running the Python scripts additionally requires casadi and matplotlib. The project was tested for Python3.13, numpy 2.3.2, casadi 3.7.1 and matplotlib 3.10.5.  
In addition, some plot functions require LaTeX to be available on the system.

### Testing and benchmarking
Run benchmark/run_blockSQP.py to confirm the solver was built correctly.  
Edit the script to select various example problems and options.  

The script benchmark/experiments/run_blockSQP_experiments.py can be used to benchmark blockSQP on several problems for perturbed start point for different options.

### C++ and Julia examples
C++ examples are located in blockSQP2/examples, the example executables are placed into blockSQP2/examples/bin.  

If blockSQP2.jl was fetched and built, Julia examples are located at blockSQP2.jl/examples. The script install_deps can be used to setup the local project and install the required dependencies.  
Then examples can be run via `julia --project=PATH/TO/blockSQP2.jl/examples FILE`,  
see <https://docs.julialang.org/en/v1/manual/code-loading/#Project-environments>
