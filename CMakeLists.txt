cmake_minimum_required(VERSION 3.14)
project(Homework)

set(CMAKE_CXX_STANDARD 14)

# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
if (result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif ()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
if (result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif ()

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
endif ()

include_directories(hw_lib)
add_subdirectory(hw_lib)
#==============================Homework 1==============================
add_executable(HW1_LSE src/HW1/LSE.cpp)
add_executable(HW1_Newton src/HW1/Newton.cpp)

target_link_libraries(HW1_LSE Matrix_lib)
target_link_libraries(HW1_Newton Matrix_lib)

#==============================Homework 2==============================
add_executable(hw2 src/HW2/main.cpp src/HW2/OnlineLearning.cpp src/HW2/MultinomialNaiveBayes.cpp src/HW2/GaussianNaiveBayes.cpp
        src/HW2/NaiveBayes.cpp)

target_link_libraries(hw2 Matrix_lib)

#==============================Homework 3==============================
add_executable(rdg src/HW3/main.cpp)
add_executable(seq src/HW3/SequentialEstimator.cpp)
add_executable(blr src/HW3/BayesianLinearRegression.cpp)

target_link_libraries(rdg Matrix_lib)
target_link_libraries(seq Matrix_lib)
target_link_libraries(blr Matrix_lib)

target_link_libraries(rdg Random_lib)
target_link_libraries(seq Random_lib)
target_link_libraries(blr Random_lib)

#==============================Homework 4==============================
add_executable(logistic_regression src/HW4/LogisticRegression.cpp)

target_link_libraries(logistic_regression Matrix_lib)
target_link_libraries(logistic_regression Random_lib)

add_executable(HW1_test test/matrix.cpp)
target_link_libraries(HW1_test Matrix_lib)
target_link_libraries(HW1_test gtest gtest_main)
add_test(HW1_test COMMAND HW1)