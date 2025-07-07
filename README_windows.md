# BlockSQP - C++, Python, Julia - Build System Using qpOASES and MUMPS

This is a modified version of the **blockSQP** nonlinear program solver that includes new CMake build specifications, a Python interface, and a Julia interface.

**Copyright (c) 2012-2015 Reinhold Wittmann <reinhold.wittmann@ovgu.de>**

## Configuration for Building BlockSQP on Windows

The following configuration is known to work for building **blockSQP** on Windows.

### Requirements:
- **CMake build system**
- **Ninja build system backend**
- **MSVC C++ compiler** (included in Visual Studio Community 2022)
- **Intel oneAPI Base Toolkit and oneAPI HPC Toolkit**  
  (Includes Fortran compiler, oneMKL (for LAPACK and BLAS))

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
