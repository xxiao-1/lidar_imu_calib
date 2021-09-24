cmake_minimum_required(VERSION 2.8.3)
project(lidar_imu_calib)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
add_compile_options(-o3)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  tf
  rosbag
  std_msgs
  message_generation
  tf_conversions
  trajectory_msgs
)

find_package(cmake_modules REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(PCL 1.7 REQUIRED)


find_package(Threads QUIET)
find_package(Ceres QUIET REQUIRED)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
	message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    if(NOT OpenCV_VERSION VERSION_LESS "3.0.0")
        add_definitions(-DHAVE_OPENCV3)
		message(STATUS "defined HAVE_OPENCV3")

        set(CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES} ${OPENCV_INCLUDE_DIRS})
        include(CheckIncludeFileCXX)

        check_include_file_cxx(opencv2/face/facerec.hpp HAVE_OPENCV_CONTRIB)
        if(HAVE_OPENCV_CONTRIB)
            add_definitions(-DHAVE_OPENCV_CONTRIB)
        elseif()
            message(STATUS "OPENCV_CONTRIB NOT PRESENT, DISABLING LOTS OF FUNCTIONALITY, SEE https://github.com/opencv/opencv_contrib")
        endif()

        check_include_file_cxx(opencv2/xfeatures2d/nonfree.hpp HAVE_OPENCV_XFEATURES2D_NONFREE)
        if(HAVE_OPENCV_XFEATURES2D_NONFREE)
            add_definitions(-DHAVE_OPENCV_XFEATURES2D_NONFREE)
        elseif()
            message(STATUS "OPENCV_XFEATURES2D_NONFREE NOT PRESENT, DISABLING LOTS OF FUNCTIONALITY, SEE https://github.com/opencv/opencv_contrib")
        endif()

        check_include_file_cxx(opencv2/cudafeatures2d.hpp HAVE_OPENCV_CUDAFEATURES2D)
        if(HAVE_OPENCV_CUDAFEATURES2D)
            add_definitions(-DHAVE_OPENCV_CUDAFEATURES2D)
        elseif()
            message(STATUS "OPENCV_CUDAFEATURES2D NOT PRESENT, DISABLING LOTS OF FUNCTIONALITY")
        endif()

    else()
        add_definitions(-DHAVE_OPENCV2)
    endif()
endif()

add_message_files(
  FILES
  chassis_data.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include

  CATKIN_DEPENDS
  std_msgs
  message_generation
)
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES lidar_imu_calib
#  CATKIN_DEPENDS roscpp rospy sensor_msgs
#  DEPENDS system_lib
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

include_directories(${EIGEN3_INCLUDE_DIR})
add_definitions(${EIGEN3_DEFINITIONS})

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_library(handeye src/lib/HandEyeCalibration.cc)
target_link_libraries(handeye
  ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${CERES_LIBRARIES}
)

add_library(calib src/lib/calibExRLidar2Imu.cpp src/lib/HandEyeCalibration.cc)
target_link_libraries(calib ${catkin_LIBRARIES} ${PCL_LIBRARIES} ndt_omp handeye)
add_executable(calib_exR_lidar2imu_node src/calib_exR_lidar2imu_node.cpp)
target_link_libraries(calib_exR_lidar2imu_node calib)

# add_executable(calib
#   src/HandEyeCalibration.cc)


