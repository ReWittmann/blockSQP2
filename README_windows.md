blockSQP 2 -- Condensing, convexification strategies, scaling heuristics and more  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; for blockSQP, the nonlinear programming solver by Dennis Janka.  
Copyright (c) 2023-2025 Reinhold Wittmann <reinhold.wittmann@ovgu.de>  

## Configuration for Building BlockSQP on Windows

The following configuration is known to work for building blockSQP 2 on Windows.

### Requirements:
- CMake build system (<https://cmake.org/download/>).
- Ninja build system backend (<https://github.com/ninja-build/ninja/releases>). Make sure the executable it is in the search path i.e. add it to "Path" in the environment variables or later add -DCMAKE_MAKE_PROGRAM=/PATH/TO/ninja.exe in the build step  
- The mingw- gcc, g++ and gfortran compilers from <https://www.mingw-w64.org/>. The x86_64-15.2.0-release-posix-seh-ucrt-rt_v13-rev0 variant from <https://github.com/niXman/mingw-builds-binaries/releases> is recommended. 
Make sure the folder "bin" inside the mingw64 folder is in the path by again modifying the "Path" environment variable.  
- OpenBLAS <https://github.com/OpenMathLib/OpenBLAS>. The 32 bit integer version is required. Either download OpenBLAS-*-x64.zip from the github releases or build from source via the commands  
`cmake -B build -G Ninja -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=PATH/TO/OpenBLAS/INSTALL/FOLDER`  
`cmake --build build --target install`  

Note: A build using MSVC, Intel ifx and MKL was possible, but suffered from strong memory leaks.  


#### Python and Julia interfaces
Download Python from (<https://www.python.org/downloads/>). It should include the required libpython.dll and enable it being found. Microsoft store Python installations may result in libpython.dll not being found.

blockSQP.jl does not require Julia to be installed. Julia 1.10 is recommended for testing and using it.


### Building:
In the command line, navigate to the blockSQP_2 folder and run the commands  
`cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DOPENBLAS_DIR=*`  
`cmake --build build`  
with `*` being the path to the OpenBLAS installation folder.  

See README.md for build options.
