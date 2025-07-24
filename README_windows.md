# BlockSQP - C++, Python, Julia - Build System Using qpOASES and MUMPS

## Configuration for Building BlockSQP on Windows

The following configuration is known to work for building **blockSQP** on Windows.

### Requirements:
- CMake build system (https://cmake.org/download/).
- Ninja build system backend (https://github.com/ninja-build/ninja/releases). Make sure the executable it is in the search path, adapt "PATH" in the environment variables.
- MSVC C++ compiler (included in Visual Studio Community 2022)
- Intel Fortran compiler and intel oneMKL from the intel oneAPI HPC Toolkit (https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit-download.html)

#### Python and Julia interfaces
Download Python from (https://www.python.org/downloads/), guaranteed to include required libpython.dll

### Steps:
1. Activate the Intel oneAPI command prompt for Visual Studio 2022:  
    Run `setvars.bat` located in:
     ```
     C:\Program Files (x86)\Intel\oneAPI
     ```  
    Search for intel oneAPI command prompt and run it.
   
2. Navigate to the `blocksqp` folder and run CMake with the following commands:

    ```  
    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release [options]  
    
    cmake --build build  
    ```

3. See **readme.md** for build options.
