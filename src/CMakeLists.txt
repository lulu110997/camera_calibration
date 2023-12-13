cmake_minimum_required(VERSION 3.0.0)
project(openposetest VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

# Caffe not automatically found even though it was built with OpenPose
link_directories(/home/louis/Git/openpose/build/caffe/lib) 
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

# Add related libraries
find_package(GFlags)
find_package(Glog)
find_package(OpenCV)
find_package(OpenPose REQUIRED)

include_directories(${OpenPose_INCLUDE_DIRS} ${GFLAGS_INCLUDE_DIR} ${glog_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_executable(openposetest_exec hello_world.cpp)
# add_executable(openposetest_exec open_pose_trial_kinect.cpp)

# Link libraries to executables
target_link_libraries( openposetest_exec ${OpenPose_LIBS} ${GFLAGS_LIBRARY} ${glog_LIBRARIES} ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# Kinect stuff
add_subdirectory("freenect_trial")