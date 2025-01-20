cmake_minimum_required(VERSION 3.14)

project(blockSQP
    DESCRIPTION "Sequential quadratic programming for problems with block-diagonal Hessian matrix."
    HOMEPAGE_URL https://github.com/djanka2/blockSQP
    LANGUAGES CXX
    )

# Enable C++11 standard or higher
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories for the project (e.g., src and include directories)
include_directories(${CMAKE_SOURCE_DIR}/include)


# QPOASES
include(FetchContent)

FetchContent_Declare(
  qpOASES
  GIT_REPOSITORY https://github.com/coin-or/qpOASES.git
  GIT_TAG        51d3fbea30142d3acbf40cf7a1c519efc27ea67b # release-3.21
)

FetchContent_MakeAvailable(qpOASES)
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries" FORCE)

set(QPSOLVER_INCLUDE ${QPSOLVER_INCLUDE} ${qpOASES_SOURCE_DIR}/include)
set(QPSOLVER_LIB ${qpOASES_BINARY_DIR}/libs/libqpOASES.a)
set(QPSOLVER_FLAGS ${QPSOLVER_FLAGS} -DQPSOLVER_QPOASES)
if(UNIX)
    target_compile_options(qpOASES PUBLIC -fPIC)
endif()

# Platform-specific configuration
if(APPLE)
    # On macOS, use the Accelerate framework for BLAS and LAPACK
    find_library(Accelerate_FRAMEWORK Accelerate REQUIRED)
    
    message(STATUS "Found Accelerate framework: ${Accelerate_FRAMEWORK}")
    
    # Set the BLAS and LAPACK libraries to Accelerate framework
    set(BLAS_LIBRARIES ${Accelerate_FRAMEWORK})
    set(LAPACK_LIBRARIES ${Accelerate_FRAMEWORK})

elseif(UNIX AND NOT APPLE)
    # On Linux (or other UNIX-like systems), use OpenBLAS via pkg-config
    find_package(PkgConfig REQUIRED)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    
    message(STATUS "Found BLAS libraries: ${BLAS_INCLUDE_DIRS}")
    message(STATUS "Found LAPACK include dirs: ${LAPACK_INCLUDE_DIRS}")

    # Set the libraries and include directories
    set(BLAS_LIBRARIES ${BLAS_LIBRARIES})
    set(LAPACK_LIBRARIES ${LAPACK_LIBRARIES})

    # Include the OpenBLAS headers
    include_directories(${LAPACK_INCLUDE_DIRS})
    include_directories(${BLAS_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

# Specify the source files for your shared library
set(SOURCE_FILES
	src/blocksqp_condensing.cpp
    src/blocksqp_matrix.cpp
    src/blocksqp_problemspec.cpp
    src/blocksqp_general_purpose.cpp
    src/blocksqp_glob.cpp
    src/blocksqp_hess.cpp
    src/blocksqp_iter.cpp
    src/blocksqp_main.cpp
    src/blocksqp_options.cpp
    src/blocksqp_qp.cpp
    src/blocksqp_restoration.cpp
    src/blocksqp_stats.cpp
    src/blocksqp_qpsolver.cpp
)

# Define the shared library target
add_library(blockSQP SHARED ${SOURCE_FILES})

add_dependencies(blockSQP qpOASES)

# Link the BLAS and LAPACK libraries to the shared library
if(APPLE)
    target_link_libraries(blockSQP PRIVATE ${Accelerate_FRAMEWORK})
endif()
target_link_libraries(blockSQP
    PRIVATE
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# Link external libraries and set include directories
target_link_libraries(blockSQP
    PRIVATE ${LAPACK_LIBRARIES}
    PRIVATE ${BLAS_LIBRARIES}
    PRIVATE ${QPSOLVER_LIB}
)

target_include_directories(blockSQP
    PRIVATE ${LAPACK_INCLUDE_DIRS}
    PRIVATE ${BLAS_INCLUDE_DIRS}
    PUBLIC include
    PUBLIC ${QPSOLVER_INCLUDE}
)

# Compilation flags for debugging and optimization
target_compile_options(blockSQP
    PUBLIC -g -O0 -fPIC ${QPSOLVER_FLAGS}
)

# TODO Add the examples
add_subdirectory(examples)

# Install the library and public headers
install(TARGETS blockSQP
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include/blockSQP
)

# Optional: Install the library (uncomment if needed)
# install(TARGETS blockSQP DESTINATION lib)

# Optional: Install header files (uncomment if needed)
# install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include FILES_MATCHING PATTERN "*.h")
