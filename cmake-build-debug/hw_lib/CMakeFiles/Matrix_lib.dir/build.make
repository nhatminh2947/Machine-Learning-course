# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/nhatminh2947/working/tools/clion-2019.1.3/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/nhatminh2947/working/tools/clion-2019.1.3/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nhatminh2947/working/semester/ML/Homework

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nhatminh2947/working/semester/ML/Homework/cmake-build-debug

# Include any dependencies generated for this target.
include hw_lib/CMakeFiles/Matrix_lib.dir/depend.make

# Include the progress variables for this target.
include hw_lib/CMakeFiles/Matrix_lib.dir/progress.make

# Include the compile flags for this target's objects.
include hw_lib/CMakeFiles/Matrix_lib.dir/flags.make

hw_lib/CMakeFiles/Matrix_lib.dir/matrix.cpp.o: hw_lib/CMakeFiles/Matrix_lib.dir/flags.make
hw_lib/CMakeFiles/Matrix_lib.dir/matrix.cpp.o: ../hw_lib/matrix.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nhatminh2947/working/semester/ML/Homework/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object hw_lib/CMakeFiles/Matrix_lib.dir/matrix.cpp.o"
	cd /home/nhatminh2947/working/semester/ML/Homework/cmake-build-debug/hw_lib && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Matrix_lib.dir/matrix.cpp.o -c /home/nhatminh2947/working/semester/ML/Homework/hw_lib/matrix.cpp

hw_lib/CMakeFiles/Matrix_lib.dir/matrix.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Matrix_lib.dir/matrix.cpp.i"
	cd /home/nhatminh2947/working/semester/ML/Homework/cmake-build-debug/hw_lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nhatminh2947/working/semester/ML/Homework/hw_lib/matrix.cpp > CMakeFiles/Matrix_lib.dir/matrix.cpp.i

hw_lib/CMakeFiles/Matrix_lib.dir/matrix.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Matrix_lib.dir/matrix.cpp.s"
	cd /home/nhatminh2947/working/semester/ML/Homework/cmake-build-debug/hw_lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nhatminh2947/working/semester/ML/Homework/hw_lib/matrix.cpp -o CMakeFiles/Matrix_lib.dir/matrix.cpp.s

hw_lib/CMakeFiles/Matrix_lib.dir/Dataset.cpp.o: hw_lib/CMakeFiles/Matrix_lib.dir/flags.make
hw_lib/CMakeFiles/Matrix_lib.dir/Dataset.cpp.o: ../hw_lib/Dataset.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nhatminh2947/working/semester/ML/Homework/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object hw_lib/CMakeFiles/Matrix_lib.dir/Dataset.cpp.o"
	cd /home/nhatminh2947/working/semester/ML/Homework/cmake-build-debug/hw_lib && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Matrix_lib.dir/Dataset.cpp.o -c /home/nhatminh2947/working/semester/ML/Homework/hw_lib/Dataset.cpp

hw_lib/CMakeFiles/Matrix_lib.dir/Dataset.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Matrix_lib.dir/Dataset.cpp.i"
	cd /home/nhatminh2947/working/semester/ML/Homework/cmake-build-debug/hw_lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nhatminh2947/working/semester/ML/Homework/hw_lib/Dataset.cpp > CMakeFiles/Matrix_lib.dir/Dataset.cpp.i

hw_lib/CMakeFiles/Matrix_lib.dir/Dataset.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Matrix_lib.dir/Dataset.cpp.s"
	cd /home/nhatminh2947/working/semester/ML/Homework/cmake-build-debug/hw_lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nhatminh2947/working/semester/ML/Homework/hw_lib/Dataset.cpp -o CMakeFiles/Matrix_lib.dir/Dataset.cpp.s

# Object files for target Matrix_lib
Matrix_lib_OBJECTS = \
"CMakeFiles/Matrix_lib.dir/matrix.cpp.o" \
"CMakeFiles/Matrix_lib.dir/Dataset.cpp.o"

# External object files for target Matrix_lib
Matrix_lib_EXTERNAL_OBJECTS =

hw_lib/libMatrix_lib.a: hw_lib/CMakeFiles/Matrix_lib.dir/matrix.cpp.o
hw_lib/libMatrix_lib.a: hw_lib/CMakeFiles/Matrix_lib.dir/Dataset.cpp.o
hw_lib/libMatrix_lib.a: hw_lib/CMakeFiles/Matrix_lib.dir/build.make
hw_lib/libMatrix_lib.a: hw_lib/CMakeFiles/Matrix_lib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nhatminh2947/working/semester/ML/Homework/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libMatrix_lib.a"
	cd /home/nhatminh2947/working/semester/ML/Homework/cmake-build-debug/hw_lib && $(CMAKE_COMMAND) -P CMakeFiles/Matrix_lib.dir/cmake_clean_target.cmake
	cd /home/nhatminh2947/working/semester/ML/Homework/cmake-build-debug/hw_lib && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Matrix_lib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
hw_lib/CMakeFiles/Matrix_lib.dir/build: hw_lib/libMatrix_lib.a

.PHONY : hw_lib/CMakeFiles/Matrix_lib.dir/build

hw_lib/CMakeFiles/Matrix_lib.dir/clean:
	cd /home/nhatminh2947/working/semester/ML/Homework/cmake-build-debug/hw_lib && $(CMAKE_COMMAND) -P CMakeFiles/Matrix_lib.dir/cmake_clean.cmake
.PHONY : hw_lib/CMakeFiles/Matrix_lib.dir/clean

hw_lib/CMakeFiles/Matrix_lib.dir/depend:
	cd /home/nhatminh2947/working/semester/ML/Homework/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nhatminh2947/working/semester/ML/Homework /home/nhatminh2947/working/semester/ML/Homework/hw_lib /home/nhatminh2947/working/semester/ML/Homework/cmake-build-debug /home/nhatminh2947/working/semester/ML/Homework/cmake-build-debug/hw_lib /home/nhatminh2947/working/semester/ML/Homework/cmake-build-debug/hw_lib/CMakeFiles/Matrix_lib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : hw_lib/CMakeFiles/Matrix_lib.dir/depend

