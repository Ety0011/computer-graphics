# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/leonardo/dev/Uni/computer-graphics

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/leonardo/dev/Uni/computer-graphics

# Include any dependencies generated for this target.
include CMakeFiles/moller_only.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/moller_only.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/moller_only.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/moller_only.dir/flags.make

CMakeFiles/moller_only.dir/Bonus1/code/main.cpp.o: CMakeFiles/moller_only.dir/flags.make
CMakeFiles/moller_only.dir/Bonus1/code/main.cpp.o: Bonus1/code/main.cpp
CMakeFiles/moller_only.dir/Bonus1/code/main.cpp.o: CMakeFiles/moller_only.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/leonardo/dev/Uni/computer-graphics/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/moller_only.dir/Bonus1/code/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/moller_only.dir/Bonus1/code/main.cpp.o -MF CMakeFiles/moller_only.dir/Bonus1/code/main.cpp.o.d -o CMakeFiles/moller_only.dir/Bonus1/code/main.cpp.o -c /home/leonardo/dev/Uni/computer-graphics/Bonus1/code/main.cpp

CMakeFiles/moller_only.dir/Bonus1/code/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/moller_only.dir/Bonus1/code/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/leonardo/dev/Uni/computer-graphics/Bonus1/code/main.cpp > CMakeFiles/moller_only.dir/Bonus1/code/main.cpp.i

CMakeFiles/moller_only.dir/Bonus1/code/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/moller_only.dir/Bonus1/code/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/leonardo/dev/Uni/computer-graphics/Bonus1/code/main.cpp -o CMakeFiles/moller_only.dir/Bonus1/code/main.cpp.s

# Object files for target moller_only
moller_only_OBJECTS = \
"CMakeFiles/moller_only.dir/Bonus1/code/main.cpp.o"

# External object files for target moller_only
moller_only_EXTERNAL_OBJECTS =

moller_only: CMakeFiles/moller_only.dir/Bonus1/code/main.cpp.o
moller_only: CMakeFiles/moller_only.dir/build.make
moller_only: CMakeFiles/moller_only.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/leonardo/dev/Uni/computer-graphics/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable moller_only"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/moller_only.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/moller_only.dir/build: moller_only
.PHONY : CMakeFiles/moller_only.dir/build

CMakeFiles/moller_only.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/moller_only.dir/cmake_clean.cmake
.PHONY : CMakeFiles/moller_only.dir/clean

CMakeFiles/moller_only.dir/depend:
	cd /home/leonardo/dev/Uni/computer-graphics && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/leonardo/dev/Uni/computer-graphics /home/leonardo/dev/Uni/computer-graphics /home/leonardo/dev/Uni/computer-graphics /home/leonardo/dev/Uni/computer-graphics /home/leonardo/dev/Uni/computer-graphics/CMakeFiles/moller_only.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/moller_only.dir/depend
