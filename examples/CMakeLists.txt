# Gather all .cpp files in the examples folder
file(GLOB EXAMPLE_FILES "${CMAKE_SOURCE_DIR}/examples/*.cpp")
list(REMOVE_ITEM EXAMPLE_FILES ${CMAKE_SOURCE_DIR}/examples/TEST_qp.cpp)

# Loop through the example files and add each as an executable
foreach(EXAMPLE ${EXAMPLE_FILES})
    # Get the name of the example file without path and extension
    get_filename_component(EXAMPLE_NAME ${EXAMPLE} NAME_WE)

    # Add the "BLOCKSQP_" prefix to the executable name
    set(EXECUTABLE_NAME "BLOCKSQP_${EXAMPLE_NAME}")

    # Create the executable for each example
    add_executable(${EXECUTABLE_NAME} ${EXAMPLE})
    
    # Link against your shared library (blockSQP)
    target_link_libraries(${EXECUTABLE_NAME} blockSQP)
    
    # Set the output directory for executables
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples)
    
    # Optionally set compile options (e.g., debugging flags)
    target_compile_options(${EXECUTABLE_NAME} PRIVATE -g -O0)
endforeach()
