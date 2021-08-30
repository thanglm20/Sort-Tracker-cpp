
#/*
#Module: Object Tracking
#Author: Le Manh Thang
#Created: August 26th, 2021
#*/
cmake_minimum_required(VERSION 3.0)

enable_language(C)
enable_language(CXX)

project (tracker)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#add lib ncnn
set( NCNN_LIB libncnn.a)
link_directories(${CMAKE_SOURCE_DIR}/libncnn-linux_x86_64/lib)
include_directories( ${CMAKE_SOURCE_DIR}/libncnn-linux_x86_64/include/ncnn)

#opencv
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})


file( GLOB SOURCES 	src/*.cpp 
                    src/detector/*.cpp
                    src/sort-tracker/*.cpp)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/src/detector)
include_directories(${CMAKE_SOURCE_DIR}/src/sort-tracker)
#add_library(bkav_Ai SHARED ${SOURCES})
add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${NCNN_LIB}
-Wall -lm -lpthread -lstdc++ -std=c++11 -lgomp
) 

