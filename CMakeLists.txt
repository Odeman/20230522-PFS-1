project(fora)
# Specify the minimum version for CMake
cmake_minimum_required(VERSION 2.8)

#SET(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall")
SET(CMAKE_CXX_FLAGS "-std=c++11 -O3 -DNDEBUG -w -pthread -lboost_serialization -lboost_filesystem -lboost_system")
#SET(CMAKE_CXX_FLAGS "-std=c++11 -O3 -w -pthread")

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(Boost REQUIRED  unit_test_framework filesystem system date_time program_options)
#find_package(Threads)

#set(HEADER algo.h build_oracle.h config.h graph.h head.h heap.h mylib.h query.h)
#set(PPRLIB  mylib.cpp config.cpp)

# Set the output folder where your program will be created
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# The following folder will be included
include_directories("${PROJECT_SOURCE_DIR}")


#add_library(mylib ${PROJECT_SOURCE_DIR}/mylib.cpp ${PROJECT_SOURCE_DIR}/mylib.h)
#add_library(config ${PROJECT_SOURCE_DIR}/config.cpp ${PROJECT_SOURCE_DIR}/config.h)

add_executable(agenda ${PROJECT_SOURCE_DIR}/agenda.cpp ${PROJECT_SOURCE_DIR}/mylib.cpp ${PROJECT_SOURCE_DIR}/config.cpp)
#add_executable(HubPPR hubppr.cpp ${PPRLIB} ${HEADER})

target_link_libraries(agenda pthread boost_serialization boost_filesystem boost_system)
