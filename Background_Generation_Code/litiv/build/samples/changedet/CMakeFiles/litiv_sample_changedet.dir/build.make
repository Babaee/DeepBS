# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.6.3/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.6.3/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/Capricorn/Desktop/litiv-a

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/Capricorn/Desktop/litiv-a/build

# Include any dependencies generated for this target.
include samples/changedet/CMakeFiles/litiv_sample_changedet.dir/depend.make

# Include the progress variables for this target.
include samples/changedet/CMakeFiles/litiv_sample_changedet.dir/progress.make

# Include the compile flags for this target's objects.
include samples/changedet/CMakeFiles/litiv_sample_changedet.dir/flags.make

samples/changedet/CMakeFiles/litiv_sample_changedet.dir/src/main.cpp.o: samples/changedet/CMakeFiles/litiv_sample_changedet.dir/flags.make
samples/changedet/CMakeFiles/litiv_sample_changedet.dir/src/main.cpp.o: ../samples/changedet/src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Capricorn/Desktop/litiv-a/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object samples/changedet/CMakeFiles/litiv_sample_changedet.dir/src/main.cpp.o"
	cd /Users/Capricorn/Desktop/litiv-a/build/samples/changedet && /usr/local/bin/g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/litiv_sample_changedet.dir/src/main.cpp.o -c /Users/Capricorn/Desktop/litiv-a/samples/changedet/src/main.cpp

samples/changedet/CMakeFiles/litiv_sample_changedet.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/litiv_sample_changedet.dir/src/main.cpp.i"
	cd /Users/Capricorn/Desktop/litiv-a/build/samples/changedet && /usr/local/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Capricorn/Desktop/litiv-a/samples/changedet/src/main.cpp > CMakeFiles/litiv_sample_changedet.dir/src/main.cpp.i

samples/changedet/CMakeFiles/litiv_sample_changedet.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/litiv_sample_changedet.dir/src/main.cpp.s"
	cd /Users/Capricorn/Desktop/litiv-a/build/samples/changedet && /usr/local/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Capricorn/Desktop/litiv-a/samples/changedet/src/main.cpp -o CMakeFiles/litiv_sample_changedet.dir/src/main.cpp.s

samples/changedet/CMakeFiles/litiv_sample_changedet.dir/src/main.cpp.o.requires:

.PHONY : samples/changedet/CMakeFiles/litiv_sample_changedet.dir/src/main.cpp.o.requires

samples/changedet/CMakeFiles/litiv_sample_changedet.dir/src/main.cpp.o.provides: samples/changedet/CMakeFiles/litiv_sample_changedet.dir/src/main.cpp.o.requires
	$(MAKE) -f samples/changedet/CMakeFiles/litiv_sample_changedet.dir/build.make samples/changedet/CMakeFiles/litiv_sample_changedet.dir/src/main.cpp.o.provides.build
.PHONY : samples/changedet/CMakeFiles/litiv_sample_changedet.dir/src/main.cpp.o.provides

samples/changedet/CMakeFiles/litiv_sample_changedet.dir/src/main.cpp.o.provides.build: samples/changedet/CMakeFiles/litiv_sample_changedet.dir/src/main.cpp.o


# Object files for target litiv_sample_changedet
litiv_sample_changedet_OBJECTS = \
"CMakeFiles/litiv_sample_changedet.dir/src/main.cpp.o"

# External object files for target litiv_sample_changedet
litiv_sample_changedet_EXTERNAL_OBJECTS =

bin/litiv_sample_changedet: samples/changedet/CMakeFiles/litiv_sample_changedet.dir/src/main.cpp.o
bin/litiv_sample_changedet: samples/changedet/CMakeFiles/litiv_sample_changedet.dir/build.make
bin/litiv_sample_changedet: lib/liblitiv_world.dylib
bin/litiv_sample_changedet: lib/liblitiv_datasets.a
bin/litiv_sample_changedet: lib/liblitiv_bsds500.a
bin/litiv_sample_changedet: lib/liblitiv_imgproc.a
bin/litiv_sample_changedet: lib/liblitiv_video.a
bin/litiv_sample_changedet: lib/liblitiv_features2d.a
bin/litiv_sample_changedet: lib/liblitiv_utils.a
bin/litiv_sample_changedet: /usr/local/lib/libopencv_videostab.3.1.0.dylib
bin/litiv_sample_changedet: /usr/local/lib/libopencv_superres.3.1.0.dylib
bin/litiv_sample_changedet: /usr/local/lib/libopencv_stitching.3.1.0.dylib
bin/litiv_sample_changedet: /usr/local/lib/libopencv_shape.3.1.0.dylib
bin/litiv_sample_changedet: /usr/local/lib/libopencv_video.3.1.0.dylib
bin/litiv_sample_changedet: /usr/local/lib/libopencv_photo.3.1.0.dylib
bin/litiv_sample_changedet: /usr/local/lib/libopencv_objdetect.3.1.0.dylib
bin/litiv_sample_changedet: /usr/local/lib/libopencv_calib3d.3.1.0.dylib
bin/litiv_sample_changedet: /usr/local/lib/libopencv_features2d.3.1.0.dylib
bin/litiv_sample_changedet: /usr/local/lib/libopencv_ml.3.1.0.dylib
bin/litiv_sample_changedet: /usr/local/lib/libopencv_highgui.3.1.0.dylib
bin/litiv_sample_changedet: /usr/local/lib/libopencv_videoio.3.1.0.dylib
bin/litiv_sample_changedet: /usr/local/lib/libopencv_imgcodecs.3.1.0.dylib
bin/litiv_sample_changedet: /usr/local/lib/libopencv_imgproc.3.1.0.dylib
bin/litiv_sample_changedet: /usr/local/lib/libopencv_flann.3.1.0.dylib
bin/litiv_sample_changedet: /usr/local/lib/libopencv_core.3.1.0.dylib
bin/litiv_sample_changedet: samples/changedet/CMakeFiles/litiv_sample_changedet.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/Capricorn/Desktop/litiv-a/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/litiv_sample_changedet"
	cd /Users/Capricorn/Desktop/litiv-a/build/samples/changedet && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/litiv_sample_changedet.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
samples/changedet/CMakeFiles/litiv_sample_changedet.dir/build: bin/litiv_sample_changedet

.PHONY : samples/changedet/CMakeFiles/litiv_sample_changedet.dir/build

samples/changedet/CMakeFiles/litiv_sample_changedet.dir/requires: samples/changedet/CMakeFiles/litiv_sample_changedet.dir/src/main.cpp.o.requires

.PHONY : samples/changedet/CMakeFiles/litiv_sample_changedet.dir/requires

samples/changedet/CMakeFiles/litiv_sample_changedet.dir/clean:
	cd /Users/Capricorn/Desktop/litiv-a/build/samples/changedet && $(CMAKE_COMMAND) -P CMakeFiles/litiv_sample_changedet.dir/cmake_clean.cmake
.PHONY : samples/changedet/CMakeFiles/litiv_sample_changedet.dir/clean

samples/changedet/CMakeFiles/litiv_sample_changedet.dir/depend:
	cd /Users/Capricorn/Desktop/litiv-a/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Capricorn/Desktop/litiv-a /Users/Capricorn/Desktop/litiv-a/samples/changedet /Users/Capricorn/Desktop/litiv-a/build /Users/Capricorn/Desktop/litiv-a/build/samples/changedet /Users/Capricorn/Desktop/litiv-a/build/samples/changedet/CMakeFiles/litiv_sample_changedet.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : samples/changedet/CMakeFiles/litiv_sample_changedet.dir/depend

