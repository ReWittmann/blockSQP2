# build system and Python interface for the original blockSQP by Dennis Janka
This project contains CMake build specifications, patches, a Python interface and example problems for the nonlinear programming solver blockSQP (Copyright (c) 2012-2015 Dennis Janka <dennis.janka@iwr.uni-heidelberg.de>) archived at <https://github.com/ReWittmann/blockSQP_reference>. It includes various example problems and scripts to perform numerical experiments with them.

Copyright (c) 2025-2026 Reinhold Wittmann <reinhold.wittmann@ovgu.de>  
Licensed under the zlib license. See LICENSE for more details.

This project downloads MUMPS <https://mumps-solver.org/index.php?page=home> (CeCILL-C license), MUMPS-CMake build system <https://github.com/scivision/mumps> (MIT license), qpOASES <https://github.com/coin-or/qpOASES> (LGPL v2.1 license), <https://github.com/ReWittmann/blockSQP_reference> (zlib license) and pybind11 <https://github.com/pybind/pybind11> (custom license). Each license applies to the respective package, and any statement in it regarding compiled code applies to binary files produced by this build system that include that compiled code.

##Build requirements
1. Linux
2. CMake
3. Git
4. A fortran compiler, e.g. gfortran
5. A C++ 11 compiler, e.g. g++-14
6. Python, this project was tested for Python 3.13

##Building
Invoke CMake on the CMakeLists.txt, e.g. navigate to the folder and use the commands  
&nbsp;&nbsp; cmake -B .build  
&nbsp;&nbsp; cmake --build .build  



If you wish to select a specific Python installation to build for, use  
&nbsp;&nbsp; cmake -B .build -DPYTHON_INTERPRETER=/PATH/TO/PYTHON/EXECUTABLE  
e.g. ... =/home/SomeOne/.localpython/bin/python3.13

##Python package requirements
Running the scripts requires numpy, casadi and matplotlib. The project was tested for version 2.3.2, 3.7.1 and 3.10.5 respectively.

To test the solver, run  
&nbsp;&nbsp; python run_old_blockSQP.py  

Edit run_old_blockSQP.py to select different examples and options. 

The script run_old_blockSQP_experiments.py can be used for benchmarking over several problems for perturbed start points for different solver options. 

