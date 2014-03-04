set(PROJECT_NAME OVV)
project(${PROJECT_NAME})

cmake_minimum_required(VERSION 2.8.0)

################################################################################
# Release by default
# Turn on Debug with "-DCMAKE_BUILD_TYPE=Debug"
IF( NOT CMAKE_BUILD_TYPE )
       SET( CMAKE_BUILD_TYPE Release )
ENDIF()

################################################################################
# Add local path for finding packages, set the local version first
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules ${CMAKE_MODULE_PATH})

################################################################################
# Create variables used for exporting (from build tree) in SophusConfig.cmake
set (OVV_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set (OVV_LIBRARY_DIR ${CMAKE_SOURCE_DIR}/lib)

################################################################################
FIND_PACKAGE( Eigen3 REQUIRED )
LIST( APPEND OVV_INCLUDE_DIR ${EIGEN3_INCLUDE_DIR} )
message("=== Eigen include directory: "${EIGEN3_INCLUDE_DIR})
find_package(LAPACK REQUIRED)
find_package(Boost REQUIRED signals serialization)
LIST( APPEND OVV_INCLUDE_DIR ${Boost_INCLUDE_DIR} )
message("=== boost include directory: "${Boost_INCLUDE_DIR})

find_package(Flann 1.8.4 REQUIRED)
if(FLANN_FOUND)
    message("======= Found ${FLANN_VERSION} ========")
    message("======= ${FLANN_INCLUDE_DIRS} ========")
    message("======= ${FLANN_LIBRARIES} ========")
endif(FLANN_FOUND)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(THIRDPARTY_SRCS "/Users/liz/codescratch/thirdparty" CACHE STRING "Location of Thirdparty Source Directory")
message("tree library is in "${THIRDPARTY_SRCS})
include_directories(${EIGEN3_INCLUDE_DIR} ${FLANN_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS} "${THIRDPARTY_SRCS}/tree-2.81/src" ${OVV_INCLUDE_DIR} "${THIRDPARTY_SRCS}/matfile") 

#set the default path for built libraries to the "lib" directory 
add_library(ovv src/VisualWord.cpp src/Vocab.cpp src/Cluster.cpp)
set(Boost_USE_STATIC_LIBS ON)
link_directories("/opt/local/lib")
include_directories("/opt/local/include")
add_executable(test_basics src/test_basics.cpp)
target_link_libraries(test_basics ${BOOST_LIBRARIES} ${FLANN_LIBRARIES} ovv lapack boost_signals boost_serialization z)
