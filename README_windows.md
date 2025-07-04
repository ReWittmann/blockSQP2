# BlockSQP - C++, Python, Julia - build system using qpOASES and MUMPS

This is a modified version of the blockSQP nonlinear program solver that includes new cmake build specifications, a python interface and a julia interface.
Copyright (c) 2012-2015 Reinhold Wittmann <reinhold.wittmann@ovgu.de>


The following configuration is known to work for building blockSQP on windows. 
    The following need to be installed
        - CMake build system
        - Ninja build system backend
        - MSVC C++ compiler (included in visual studio community 2022)
        - Intel oneAPI Base Toolkit and oneAPI HPC Toolkit (Fortran compiler, oneMKL (for LAPACK and BLAS))

Activate the Intel oneAPI command prompt for Visual Studios 2022. To enable this command prompt,
the oneAPI setvars.bat, normally located in C:\Program Files (x86)\Intel\oneAPI must be run.
Navigate to the blockSQP folder and call CMake via
 CMake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release [options]
 CMake --build build 

See the normal readme.md for build options
