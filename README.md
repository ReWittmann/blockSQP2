# BlockSQP - C++, Python, Julia - build system using qpOASES and MUMPS

Both a Fortran and a C++ compiler are required, change from defaults via -DCMAKE_FC_COMPILER=..., -DCMAKE_CXX_COMPILER=

Create build files by calling
cmake -B build ${OPTIONS}

Options are set via the argument -D${OPTION}=...
The following OPTIONS may be set
    PYTHON_INTERFACE (ON/OFF)-  build the python interface
    PYTHON_INTERPRETER (/...)-  path to the python interpreter you wish to build for, set if different from default system python
    
    JULIA_INTERFACE (ON/OFF)-   build the julia interface, requires installing the CxxWrap.jl library via julia pkg
    CXXWRAP_PATH (/...)-        path to the libcxxwrap-julia artifact of CxxWrap.jl, e.g. 
                        /path/to/.../.julia/artifacts/65c14d6c8b06e52ca794200129a8f3dd8b7ce34e/lib/cmake/JlCxx
                        Search for cxxwrap in your artifacts folder, find libcxxwrap_julia_jll in your julia packages folder
                        and find the git-tree-sha1 for your julia version and system. 
                        
Compile via
cmake --build build
    OR
make -C build


See 
    blockSQP/examples/example1,
    py_blockSQP/example1.py
    blockSQP_jl/example1.jl
for how to interface BlockSQP. 
See
    blockSQP/include/options.hpp
for possible settings. 

