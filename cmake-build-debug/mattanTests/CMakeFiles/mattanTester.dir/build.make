# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/c/Users/nirfi/CLionProjects/OS-NEW/EX3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/Users/nirfi/CLionProjects/OS-NEW/EX3/cmake-build-debug

# Include any dependencies generated for this target.
include mattanTests/CMakeFiles/mattanTester.dir/depend.make

# Include the progress variables for this target.
include mattanTests/CMakeFiles/mattanTester.dir/progress.make

# Include the compile flags for this target's objects.
include mattanTests/CMakeFiles/mattanTester.dir/flags.make

mattanTests/CMakeFiles/mattanTester.dir/SampleClient.cpp.o: mattanTests/CMakeFiles/mattanTester.dir/flags.make
mattanTests/CMakeFiles/mattanTester.dir/SampleClient.cpp.o: ../mattanTests/SampleClient.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/nirfi/CLionProjects/OS-NEW/EX3/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object mattanTests/CMakeFiles/mattanTester.dir/SampleClient.cpp.o"
	cd /mnt/c/Users/nirfi/CLionProjects/OS-NEW/EX3/cmake-build-debug/mattanTests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mattanTester.dir/SampleClient.cpp.o -c /mnt/c/Users/nirfi/CLionProjects/OS-NEW/EX3/mattanTests/SampleClient.cpp

mattanTests/CMakeFiles/mattanTester.dir/SampleClient.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mattanTester.dir/SampleClient.cpp.i"
	cd /mnt/c/Users/nirfi/CLionProjects/OS-NEW/EX3/cmake-build-debug/mattanTests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/nirfi/CLionProjects/OS-NEW/EX3/mattanTests/SampleClient.cpp > CMakeFiles/mattanTester.dir/SampleClient.cpp.i

mattanTests/CMakeFiles/mattanTester.dir/SampleClient.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mattanTester.dir/SampleClient.cpp.s"
	cd /mnt/c/Users/nirfi/CLionProjects/OS-NEW/EX3/cmake-build-debug/mattanTests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/nirfi/CLionProjects/OS-NEW/EX3/mattanTests/SampleClient.cpp -o CMakeFiles/mattanTester.dir/SampleClient.cpp.s

mattanTests/CMakeFiles/mattanTester.dir/SampleClient.cpp.o.requires:

.PHONY : mattanTests/CMakeFiles/mattanTester.dir/SampleClient.cpp.o.requires

mattanTests/CMakeFiles/mattanTester.dir/SampleClient.cpp.o.provides: mattanTests/CMakeFiles/mattanTester.dir/SampleClient.cpp.o.requires
	$(MAKE) -f mattanTests/CMakeFiles/mattanTester.dir/build.make mattanTests/CMakeFiles/mattanTester.dir/SampleClient.cpp.o.provides.build
.PHONY : mattanTests/CMakeFiles/mattanTester.dir/SampleClient.cpp.o.provides

mattanTests/CMakeFiles/mattanTester.dir/SampleClient.cpp.o.provides.build: mattanTests/CMakeFiles/mattanTester.dir/SampleClient.cpp.o


# Object files for target mattanTester
mattanTester_OBJECTS = \
"CMakeFiles/mattanTester.dir/SampleClient.cpp.o"

# External object files for target mattanTester
mattanTester_EXTERNAL_OBJECTS =

mattanTests/mattanTester: mattanTests/CMakeFiles/mattanTester.dir/SampleClient.cpp.o
mattanTests/mattanTester: mattanTests/CMakeFiles/mattanTester.dir/build.make
mattanTests/mattanTester: libMapReduceFramework.a
mattanTests/mattanTester: lib/libgtest_main.a
mattanTests/mattanTester: lib/libgtest.a
mattanTests/mattanTester: mattanTests/CMakeFiles/mattanTester.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/nirfi/CLionProjects/OS-NEW/EX3/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable mattanTester"
	cd /mnt/c/Users/nirfi/CLionProjects/OS-NEW/EX3/cmake-build-debug/mattanTests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mattanTester.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
mattanTests/CMakeFiles/mattanTester.dir/build: mattanTests/mattanTester

.PHONY : mattanTests/CMakeFiles/mattanTester.dir/build

mattanTests/CMakeFiles/mattanTester.dir/requires: mattanTests/CMakeFiles/mattanTester.dir/SampleClient.cpp.o.requires

.PHONY : mattanTests/CMakeFiles/mattanTester.dir/requires

mattanTests/CMakeFiles/mattanTester.dir/clean:
	cd /mnt/c/Users/nirfi/CLionProjects/OS-NEW/EX3/cmake-build-debug/mattanTests && $(CMAKE_COMMAND) -P CMakeFiles/mattanTester.dir/cmake_clean.cmake
.PHONY : mattanTests/CMakeFiles/mattanTester.dir/clean

mattanTests/CMakeFiles/mattanTester.dir/depend:
	cd /mnt/c/Users/nirfi/CLionProjects/OS-NEW/EX3/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/nirfi/CLionProjects/OS-NEW/EX3 /mnt/c/Users/nirfi/CLionProjects/OS-NEW/EX3/mattanTests /mnt/c/Users/nirfi/CLionProjects/OS-NEW/EX3/cmake-build-debug /mnt/c/Users/nirfi/CLionProjects/OS-NEW/EX3/cmake-build-debug/mattanTests /mnt/c/Users/nirfi/CLionProjects/OS-NEW/EX3/cmake-build-debug/mattanTests/CMakeFiles/mattanTester.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : mattanTests/CMakeFiles/mattanTester.dir/depend
