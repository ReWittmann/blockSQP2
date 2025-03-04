# BlockSQP - C++, Python, Julia - build system using qpOASES and MUMPS

Both a Fortran and a C++ are required. 
Create build files by calling
cmake -B build ${OPTIONS}

The following OPTIONS may be set
    PYTHON_INTERFACE (ON/OFF)-  build the python interface
    PYTHON_PATH (/...)-       path to your python installation folder containing bin, include, ...
    
    JULIA_INTERFACE (ON/OFF)-   build the julia interface, requires installing the CxxWrap.jl library via julia pkg
    CXXWRAP_PATH (/...)-      path to the libcxxwrap-julia artifact of CxxWrap.jl, e.g. 
                        /path/to/.../.julia/artifacts/65c14d6c8b06e52ca794200129a8f3dd8b7ce34e/lib/cmake/JlCxx
Compile via calling either
cmake --build build
    or
make -C build


See 
    blockSQP/examples/example1,
    py_blockSQP/example1.py
    blockSQP_jl/example1.jl
for how to interface BlockSQP. 
See
    blockSQP/include/options.hpp
for possible settings. 

