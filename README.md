**blockSQP 2** -- Condensing, convexification strategies, scaling heuristics and more  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; for blockSQP, the nonlinear programming solver by Dennis Janka.  
Copyright (c) 2023-2025 Reinhold Wittmann <reinhold.wittmann@ovgu.de>  

Part of blockSQP 2 is **py_blockSQP** -- A python interface to blockSQP 2,  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; the nonlinear programming solver based on blockSQP by Dennis Janka.

###Licensing
blockSQP 2 preserves the zlib license of blockSQP, see LICENSE.txt and blockSQP/LICENSE.txt.  
  
This project includes a modified version of qpOASES <https://github.com/coin-or/qpOASES>, located in blockSQP/dep and subject to a separate LGPL-2.1 license, see blockSQP/dep/modified_qpOASES/LICENSE.txt  

When building this project with CMake, the following dependencies are downloaded.  
1. MUMPS <https://mumps-solver.org/index.php?page=home> (CeCILL-C license)  
2. MUMPS-CMake build system <https://github.com/scivision/mumps> (MIT license)  
3. pybind11 <https://github.com/pybind/pybind11> (custom license)  
  
Each license applies to the respective package, and any statement in it regarding compiled code applies to binary files produced by this projects build system that include that compiled code.


#Build requirements
1. A Fortran compiler, e.g. gfortran
2. A C++-20 compatible C++ compiler, e.g. g++-14
3. The CMake build system <https://cmake.org/>
4. A build system backend (GNU make, Ninja build)

##Building
Build by calling:  
&nbsp;&nbsp; cmake -B .build ${OPTIONS}  
&nbsp;&nbsp; cmake --build .build

###General build options
1. -DCMAKE_Fortran_COMPILER=... - choose the Fortran compiler
2. -DCMAKE_CXX_COMPILER=... - choose the C++ compiler
3. -G ("Unix Makefiles" / Ninja) - choose the build system backend

###blockSQP build options
1. -DPYTHON_INTERFACE= (ON/OFF) - build py_blockSQP, default ON
2. -DPYTHON_INTERPRETER= (/PATH/TO/PYTHON_EXECUTABLE) - optional, choose python interpreter to build for
3. -DJULIA_INTERFACE= (ON/OFF) - build blocksqp.jl from the C interface, default ON

See README_WINDOWS.md on how to build for windows with MSVC and intel Fortran.

###Binaries
The binaries are placed into /blockSQP/lib or /blockSQP/bin, /py_blockSQP and /blocksqp.jl/bin. 

##Examples and documentation

###Python interface requirements
The py_blockSQP requires numpy, running the Python scripts additionally requires casadi and matplotlib. The project was tested for numpy 2.3.2, casadi 3.7.1 and matplotlib 3.10.5.  
In addition, some plot functions require LaTeX to be available on the system.

##Testing and benchmarking
Run benchmark/run_blockSQP.py to confirm the solver was built correctly.  
Edit the script to select various example problems and options.  

The script benchmark/experiments/run_blockSQP_experiments.py can be used to benchmark blockSQP on several problems for perturbed start point for different options.

###C++ and Julia examples
C++ examples are located in blockSQP/examples, the example executables are placed into blockSQP/examples/bin.  

If blockSQP.jl was built, julia examples are located at blocksqp.jl/scripts.  
Tests can be run after installing the dependencies specified in the project.toml

