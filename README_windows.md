blockSQP 2 -- Condensing, convexification strategies, scaling heuristics and more  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; for blockSQP, the nonlinear programming solver by Dennis Janka.  
Copyright (c) 2023-2025 Reinhold Wittmann <reinhold.wittmann@ovgu.de>  

## Configuration for Building BlockSQP on Windows

The following configuration is known to work for building blockSQP 2 on Windows.

### Requirements:
- CMake build system (<https://cmake.org/download/>).
- Ninja build system backend (<https://github.com/ninja-build/ninja/releases>). Make sure the executable it is in the search path, adapt "PATH" in the environment variables or later add -DCMAKE_MAKE_PROGRAM=/PATH/TO/ninja.exe in the build step
- MSVC C++ compiler (included in Visual Studio Community 2022)
- Intel Fortran compiler and intel oneMKL from the intel oneAPI HPC Toolkit (<https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit-download.html>)

#### Python and Julia interfaces
Download Python from (<https://www.python.org/downloads/>), guaranteed to include required libpython.dll

### Steps:
1. Activate the Intel oneAPI command prompt for Visual Studio 2022:  
    Run `setvars.bat` located in:
     ```
     C:\Program Files (x86)\Intel\oneAPI
     ```  
    Search for intel oneAPI command prompt and run it.
   
2. Navigate to this folder and run CMake with the following commands:

    ```  
    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release [options]  
    
    cmake --build build  
    ```

3. See README.md for build options.
