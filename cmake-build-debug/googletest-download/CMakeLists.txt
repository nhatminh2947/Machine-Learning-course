cmake_minimum_required(VERSION 3.14)

project(googletest-download NONE)

include(ExternalProject)
ExternalProject_Add(googletest
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           master
  SOURCE_DIR        "/home/nhatminh2947/working/semester/ML/Homework/cmake-build-debug/googletest-src"
  BINARY_DIR        "/home/nhatminh2947/working/semester/ML/Homework/cmake-build-debug/googletest-build"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)
