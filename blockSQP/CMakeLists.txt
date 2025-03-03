cmake_minimum_required(VERSION 3.18)

# Project Information
project(blockSQP
    DESCRIPTION "Sequential quadratic programming for problems with block-diagonal Hessian matrix."
    HOMEPAGE_URL https://github.com/djanka2/blockSQP
    LANGUAGES CXX
    )
    

#Link possible QP solvers
#QPOASES
	set(QPSOLVER_INCLUDE ${QPSOLVER_INCLUDE} /home/reinhold/qpOASES/include)
	set(QPSOLVER_LIB ${QPSOLVER_LIB} /home/reinhold/qpOASES/libs/libqpOASES.so)
	set(QPSOLVER_FLAGS ${QPSOLVER_FLAGS} -DQPSOLVER_QPOASES)
#GUROBI
	set(QPSOLVER_INCLUDE ${QPSOLVER_INCLUDE} /home/reinhold/gurobi/gurobi1102/linux64/include)
	set(QPSOLVER_LIB ${QPSOLVER_LIB} /home/reinhold/gurobi/gurobi1102/linux64/lib/libgurobi_c++.a /home/reinhold/gurobi/gurobi1102/linux64/lib/libgurobi110.so)
	set(QPSOLVER_FLAGS ${QPSOLVER_FLAGS} -DQPSOLVER_GUROBI)
#QPALM
	find_package(Eigen3 REQUIRED)
	add_library(QPALMCXX SHARED IMPORTED)
	target_link_libraries(QPALMCXX INTERFACE /home/reinhold/QPALM/QPALM/libqpalm.a 
		/home/reinhold/LADEL/LADEL/libladel.a /home/reinhold/LADEL/thirdparty/SuiteSparse/libladel_amd.a)
	set_target_properties(QPALMCXX PROPERTIES IMPORTED_LOCATION /home/reinhold/QPALM/QPALM/interfaces/cxx/libqpalm_cxx.so)

	set(QPSOLVER_FLAGS ${QPSOLVER_FLAGS} -DQPSOLVER_QPALM)
	set(QPSOLVER_INCLUDE ${QPSOLVER_INCLUDE} /home/reinhold/LADEL/LADEL/include 
		/home/reinhold/QPALM/QPALM/include /home/reinhold/QPALM/QPALM/interfaces/cxx/include)
	set(QPSOLVER_LIB ${QPSOLVER_LIB} QPALMCXX Eigen3::Eigen)
#################



# Find dependencies: LAPACK
#find_package(LAPACK REQUIRED)
#set(BLA_STATIC ON)
find_package(BLAS REQUIRED)

# Add target
add_library(blockSQP SHARED
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
    src/blocksqp_defs.cpp
    )
set_property(TARGET blockSQP PROPERTY PUBLIC_HEADER
	include/blocksqp_condensing.hpp
    include/blocksqp_defs.hpp
    include/blocksqp_general_purpose.hpp
    include/blocksqp_iterate.hpp
    include/blocksqp_lapack.h
    include/blocksqp_matrix.hpp
    include/blocksqp_method.hpp
    include/blocksqp_options.hpp
    include/blocksqp_problemspec.hpp
    include/blocksqp_restoration.hpp
    include/blocksqp_stats.hpp
    include/blocksqp_qpsolver.hpp
    )
    

target_compile_options(blockSQP PUBLIC -g -Wall -Wno-unused-variable -Wno-unused-but-set-variable -Wno-overloaded-virtual -O0 ${QPSOLVER_FLAGS})
# target_compile_options(blockSQP PUBLIC -g -O0 ${QPSOLVER_FLAGS})
target_include_directories(blockSQP PUBLIC include ${QPSOLVER_INCLUDE})
target_link_libraries(blockSQP BLAS::BLAS ${QPSOLVER_LIB})

set_target_properties(blockSQP
	PROPERTIES
	LIBRARY_OUTPUT_DIRECTORY "./lib")

add_executable(example1 examples/example1.cpp)
target_compile_options(example1 PUBLIC -g -O0)
target_link_libraries(example1 blockSQP)
set_target_properties(example1
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "./examples"
	)

add_executable(TEST_condensing examples/TEST_condensing.cpp)
target_compile_options(TEST_condensing PUBLIC -g -O0)
target_link_libraries(TEST_condensing blockSQP)
set_target_properties(TEST_condensing
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "./examples"
)

add_executable(TEST_condensing_var examples/TEST_condensing_var.cpp)
target_compile_options(TEST_condensing_var PUBLIC -g -O0)
target_link_libraries(TEST_condensing_var blockSQP)
set_target_properties(TEST_condensing_var
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "./examples"
)

add_executable(TEST_condensing_multiple examples/TEST_condensing_multiple.cpp)
target_compile_options(TEST_condensing_multiple PUBLIC -g -O0)
target_link_libraries(TEST_condensing_multiple blockSQP)
set_target_properties(TEST_condensing_multiple
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "./examples"
)

add_executable(TEST_condensing_second_hessian examples/TEST_condensing_second_hessian.cpp)
target_compile_options(TEST_condensing_second_hessian PUBLIC -g -O0)
target_link_libraries(TEST_condensing_second_hessian blockSQP)
set_target_properties(TEST_condensing_second_hessian
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "./examples"
)

add_executable(TEST_condensing_SOC examples/TEST_condensing_SOC.cpp)
target_compile_options(TEST_condensing_SOC PUBLIC -g -O0)
target_link_libraries(TEST_condensing_SOC blockSQP)
set_target_properties(TEST_condensing_SOC
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "./examples"
)

add_executable(TEST_block examples/TEST_block.cpp)
target_compile_options(TEST_block PUBLIC -g -O0)
target_link_libraries(TEST_block blockSQP)
set_target_properties(TEST_block
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "./examples"
)

add_executable(TEST_qp examples/TEST_qp.cpp)
target_compile_options(TEST_qp PUBLIC -g -O0)
target_link_libraries(TEST_qp blockSQP)
set_target_properties(TEST_qp
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "./examples"
)

add_executable(TEST_condensing_QPS examples/TEST_condensing_QPS.cpp)
target_compile_options(TEST_condensing_QPS PUBLIC -g -O0)
target_link_libraries(TEST_condensing_QPS blockSQP)
set_target_properties(TEST_condensing_QPS
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "./examples"
)


install(TARGETS blockSQP
    LIBRARY
    PUBLIC_HEADER
        DESTINATION include/blockSQP
    )
