
cmake_minimum_required(VERSION 3.18)


# Project Information
project(blockSQP
    DESCRIPTION "Sequential quadratic programming for problems with block-diagonal Hessian matrix."
    HOMEPAGE_URL https://github.com/djanka2/blockSQP
    LANGUAGES CXX
    )

# Find dependencies: LAPACK
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)

# Find dependencies: qpOASES
find_path(QPOASES_PREFIX
    include/qpOASES.hpp
    DOC "qpOASES directory prefix."
    REQUIRED
    )
#find_library(QPOASES_LIBRARY
#    qpOASES
#    REQUIRED
#    )


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
    )
    
target_compile_options(blockSQP PUBLIC)# -g -O0)

#############################
#Path to qpOASES needs to be added here
#############################
target_include_directories(blockSQP PUBLIC include PATH/TO/qpOASES/include)
target_link_libraries(blockSQP LAPACK::LAPACK PATH/TO/qpOASES/libs/libqpOASES.so)
set_target_properties(blockSQP
	PROPERTIES
	LIBRARY_OUTPUT_DIRECTORY "~/blockSQP_modified/lib")

add_executable(example1 examples/example1.cpp)
target_link_libraries(example1 blockSQP)
set_target_properties(example1
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "~/blockSQP_modified/examples"
	)


add_executable(TEST_condensing examples/TEST_condensing.cpp)
target_compile_options(TEST_condensing PUBLIC -g -O0)
target_link_libraries(TEST_condensing blockSQP)
set_target_properties(TEST_condensing
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "~/blockSQP_modified/examples"
)

add_executable(TEST_condensing_multiple examples/TEST_condensing_multiple.cpp)
target_compile_options(TEST_condensing_multiple PUBLIC -g -O0)
target_link_libraries(TEST_condensing_multiple blockSQP)
set_target_properties(TEST_condensing_multiple
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "~/blockSQP_modified/examples"
)

add_executable(TEST_condensing_second_hessian examples/TEST_condensing_second_hessian.cpp)
target_compile_options(TEST_condensing_second_hessian PUBLIC -g -O0)
target_link_libraries(TEST_condensing_second_hessian blockSQP)
set_target_properties(TEST_condensing_second_hessian
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "~/blockSQP_modified/examples"
)

add_executable(TEST_condensing_SOC examples/TEST_condensing_SOC.cpp)
target_compile_options(TEST_condensing_SOC PUBLIC -g -O0)
target_link_libraries(TEST_condensing_SOC blockSQP)
set_target_properties(TEST_condensing_SOC
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "~/blockSQP_modified/examples"
)

#add_executable(TEST_block examples/TEST_block.cpp)
#target_compile_options(TEST_block PUBLIC -g -O0)
#target_link_libraries(TEST_block blockSQP)
#set_target_properties(TEST_block
#	PROPERTIES
#	RUNTIME_OUTPUT_DIRECTORY "~/blockSQP_modified/examples"
#)

install(TARGETS blockSQP
    LIBRARY
    PUBLIC_HEADER
        DESTINATION include/blockSQP
    )
