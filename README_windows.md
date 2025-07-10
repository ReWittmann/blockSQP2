# BlockSQP - C++, Python, Julia - Build System Using qpOASES and MUMPS

## Configuration for Building BlockSQP on Windows

The following configuration is known to work for building **blockSQP** on Windows.

### Requirements:
- **CMake build system (https://cmake.org/download/)**
- **Ninja build system backend (https://github.com/ninja-build/ninja/releases)**
- **MSVC C++ compiler** (included in Visual Studio Community 2022)
- **Intel Fortran compiler and intel oneMKL from the intel oneAPI HPC Toolkit (https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit-download.html)**

#### Python and Julia interfaces
- **Download Python from (https://www.python.org/downloads/), guaranteed to include required libpython.dll**
- 

### Steps:
1. Activate the Intel oneAPI command prompt for Visual Studio 2022.
   - To enable this command prompt, run `setvars.bat` located in:
     ```
     C:\Program Files (x86)\Intel\oneAPI
     ```
   
2. Navigate to the `blockSQP` folder and run CMake with the following commands:

    ```bash
    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release [options]
    cmake --build build
    ```

3. See **readme.md** for additional build options.
