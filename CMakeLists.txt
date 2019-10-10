cmake_minimum_required(VERSION 3.15)
project(Machine_Learning_course)

set(CMAKE_CXX_STANDARD 14)

# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
        ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
        EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include")
endif()

include_directories(hw_lib)

add_executable(Machine_Learning_course
        hw_lib/Dataset.cpp
        hw_lib/Dataset.h
        hw_lib/matrix.cpp
        hw_lib/matrix.h
        src/HW1/LSE.cpp
        src/HW1/Newton.cpp
        test/matrix.cpp src/HW2/MNISTNaiveBayesClassifier.cpp src/HW2/MNISTNaiveBayesClassifier.h hw_lib/IdxFileReader.cpp hw_lib/IdxFileReader.h test/IDXFileReader.cpp src/HW2/main.cpp)

add_executable(hw2
        hw_lib/Dataset.cpp
        hw_lib/Dataset.h
        hw_lib/matrix.cpp
        hw_lib/matrix.h
        src/HW2/MNISTNaiveBayesClassifier.cpp
        src/HW2/MNISTNaiveBayesClassifier.h
        hw_lib/IdxFileReader.cpp
        hw_lib/IdxFileReader.h
        src/HW2/main.cpp)

add_executable(matrix_test test/matrix.cpp hw_lib/matrix.cpp)
target_link_libraries(matrix_test gtest gtest_main)

add_executable(idx_file_reader test/IDXFileReader.cpp hw_lib/IdxFileReader.cpp hw_lib/matrix.cpp)
target_link_libraries(idx_file_reader gtest gtest_main)