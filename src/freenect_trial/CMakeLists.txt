######################################################################################
# License
######################################################################################

# This file is part of the OpenKinect Project. http://www.openkinect.org
#
# Copyright (c) 2010 individual OpenKinect contributors. See the CONTRIB file
# for details.
#
# This code is licensed to you under the terms of the Apache License, version
# 2.0, or, at your option, the terms of the GNU General Public License,
# version 2.0. See the APACHE20 and GPL2 files for the text of the licenses,
# or the following URLs:
# http://www.apache.org/licenses/LICENSE-2.0
# http://www.gnu.org/licenses/gpl-2.0.txt
#
# If you redistribute this file in source form, modified or unmodified, you
# may:
#   1) Leave this header intact and distribute it under the same terms,
#      accompanying it with the APACHE20 and GPL20 files, or
#   2) Delete the Apache 2.0 clause and accompany it with the GPL2 file, or
#   3) Delete the GPL v2 clause and accompany it with the APACHE20 file
# In all cases you must keep the copyright notice intact and include a copy
# of the CONTRIB file.
#
# Binary distributions must follow the binary distribution requirements of
# either License.

######################################################################################
# CMake directives
######################################################################################

cmake_minimum_required(VERSION 2.8.12)

######################################################################################
# Project declaration and options
######################################################################################


message(asssss="${CMAKE_MODULE_PATH} ")
# Find the host operating system and architecture
include (FindOS)
# Set up installation directories
include (SetupDirectories)

set (PROJECT_VER_MAJOR 0)
set (PROJECT_VER_MINOR 7)
set (PROJECT_VER_PATCH 0)
set (PROJECT_VER
  "${PROJECT_VER_MAJOR}.${PROJECT_VER_MINOR}.${PROJECT_VER_PATCH}")
set (PROJECT_APIVER
  "${PROJECT_VER_MAJOR}")

OPTION(BUILD_REDIST_PACKAGE "Build libfreenect in a legally-redistributable manner (only affects audio)" ON)
OPTION(BUILD_EXAMPLES "Build example programs" ON)
OPTION(BUILD_FAKENECT "Build fakenect mock library" ON)
OPTION(BUILD_C_SYNC "Build c synchronous library" ON)
OPTION(BUILD_CPP "Build C++ Library (currently header only)" ON)
OPTION(BUILD_CV "Build OpenCV wrapper" OFF)
OPTION(BUILD_AS3_SERVER "Build the Actionscript 3 Server Example" OFF)
OPTION(BUILD_PYTHON "Build Python extensions" OFF)
OPTION(BUILD_PYTHON2 "Build Python 2 extension" OFF)
OPTION(BUILD_PYTHON3 "Build Python 3 extension" OFF)
OPTION(BUILD_OPENNI2_DRIVER "Build libfreenect driver for OpenNI2" OFF)
IF(PROJECT_OS_LINUX)
	OPTION(BUILD_CPACK_DEB "Build an DEB using CPack" OFF)
	OPTION(BUILD_CPACK_RPM "Build an RPM using CPack" OFF)
	OPTION(BUILD_CPACK_TGZ "Build an TGZ using CPack" OFF)
ENDIF(PROJECT_OS_LINUX)

######################################################################################
# Dependencies and Definitions
######################################################################################

# Find packages needed to build library
find_package(libusb-1.0 REQUIRED)

# Check the endianness of the system
include (TestBigEndian)
test_big_endian(BIG_ENDIAN)
if(BIG_ENDIAN)
  add_definitions(-DFN_BIGENDIAN)
endif()

if (WIN32)
  set(MATH_LIB "")
else(WIN32)
  set(MATH_LIB "m")
endif()

######################################################################################
# CMake
######################################################################################

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
SET(DOC_OUTPUT_PATH ${CMAKE_BINARY_DIR}/doc)

if    (MSVC)
  set(C_FLAGS_WARNING "-W4")
else  ()
  set(C_FLAGS_WARNING "-Wall")
endif (MSVC)

set(C_CXX_FLAGS_DEFAULT "${C_FLAGS_WARNING} -O2")

# These defaults can be overriden by -DCMAKE_C_FLAGS=""
set(CMAKE_C_FLAGS "${C_CXX_FLAGS_DEFAULT} ${CMAKE_C_FLAGS}")
# C Configurations
SET(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS} -O0 -g -DDEBUG=1")
SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS}")
SET(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELEASE} -g")

# These defaults can be overriden by -DCMAKE_CXX_FLAGS=""
set(CMAKE_CXX_FLAGS "${C_CXX_FLAGS_DEFAULT} ${CMAKE_CXX_FLAGS}")
# C++ Configurations
SET(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS} -O0 -g -DDEBUG=1")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g")

# Pretty much everyone is going to need the main includes
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/include)

# libfreenect.h includes libusb.h, so everyone needs this too
include_directories(${LIBUSB_1_INCLUDE_DIRS})

if(WIN32)
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/platform/windows")
endif()

# Add library project
install(FILES libfreenect.hpp
        DESTINATION ${PROJECT_INCLUDE_INSTALL_DIR})
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(Threads)
find_package(OpenGL)
find_package(GLUT)
if (Threads_FOUND AND OPENGL_FOUND AND GLUT_FOUND)
    add_library (freenect_cv SHARED libfreenect_cv.cpp)
    set_target_properties (freenect_cv PROPERTIES
        VERSION ${PROJECT_VER}
        SOVERSION ${PROJECT_APIVER})
  

    include_directories(. ${THREADS_PTHREADS_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR})
    target_link_libraries (freenect_cv freenect_sync ${OpenCV_LIBS})


    install (TARGETS freenect_cv
        DESTINATION "${PROJECT_LIBRARY_INSTALL_DIR}")
    install (FILES "libfreenect_cv.h"
        DESTINATION ${PROJECT_INCLUDE_INSTALL_DIR})

    add_executable(freenect-cvtrial opencv_freenect.cpp)
    target_link_libraries(freenect-cvtrial freenect freenect_sync freenect_cv ${OpenCV_LIBS})
    install (TARGETS freenect-cvtrial
        DESTINATION bin)  

    add_executable(freenect-cppview cppview.cpp)
    add_executable(freenect-cpp_pcview cpp_pc_view.cpp)
    add_executable(freenect-opencv_freenect opencv_freenect.cpp)


    target_compile_features(freenect-cppview PUBLIC cxx_std_11)
    target_compile_features(freenect-cpp_pcview PUBLIC cxx_std_11)
    target_compile_features(freenect-opencv_freenect PUBLIC cxx_std_11)


    target_link_libraries(freenect-cppview freenect OpenGL::GL OpenGL::GLU GLUT::GLUT ${CMAKE_THREAD_LIBS_INIT} ${MATH_LIB})
    target_link_libraries(freenect-cpp_pcview freenect OpenGL::GL OpenGL::GLU GLUT::GLUT ${CMAKE_THREAD_LIBS_INIT} ${MATH_LIB})
    target_link_libraries(freenect-opencv_freenect freenect OpenGL::GL OpenGL::GLU GLUT::GLUT ${CMAKE_THREAD_LIBS_INIT} ${MATH_LIB} ${OpenCV})

endif()

IF(BUILD_OPENNI2_DRIVER)
  add_subdirectory(OpenNI2-FreenectDriver)
ENDIF()

