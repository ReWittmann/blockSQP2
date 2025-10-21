# BlockSQP - C++, Python, Julia - build system using qpOASES and MUMPS

This is a modified version of the blockSQP nonlinear program solver that includes new cmake build specifications, a python interface and a julia interface.
Copyright (c) 2024-2025 Reinhold Wittmann <reinhold.wittmann@ovgu.de>



#Build requirements
1. A Fortran compiler, e.g. gfortran
2. A C++-20 compatible C++ compiler, e.g. g++-14
3. The CMake build system
4. A build system backend (GNU make, Ninja)

##Building
Build by calling:  
&nbsp;&nbsp; cmake -B .build ${OPTIONS}  
&nbsp;&nbsp; cmake --build .build

###General build options
1. -DCMAKE_Fortran_COMPILER=... - chose the Fortran compiler
2. -DCMAKE_CXX_COMPILER=... - chose the C++ compiler
3. -G ("Unix Makefiles" / Ninja) - chose the build system backend

###blockSQP build options
1. -DPYTHON_INTERFACE= (ON/OFF) - build py_blockSQP, default ON
2. -DPYTHON_INTERPRETER= (/PATH/TO/PYTHON_EXECUTABLE) - optional, choose python interpreter to build for
3. -DJULIA_INTERFACE= (ON/OFF) - build blocksqp.jl from the C interface, default ON

See README_WINDOWS.md on how to build for windows with MSVC and intel Fortran.

###Binaries
The binaries are placed into /blockSQP/lib or /blockSQP/bin, /py_blockSQP and /blocksqp.jl/bin. 

##Examples and documentation
Try running benchmark/run_blockSQP.py to confirm the solver was built correctly.  
Edit run_blockSQP.py to select various example problems and options. 

###C++ and julia examples
C++ examples are located in blockSQP/examples, the example executables are placed into blockSQP/examples/bin.  

If blockSQP.jl was built, julia examples are located at blocksqp.jl/scripts.  

