cmake_minimum_required(VERSION 2.8)

project(ThresholdScaling CXX)

set(NUM_TRIALS 5
  CACHE STRING "A number representing how many trials to run per configuration."
  )

find_file(DATA_FILE normal_1349.dat
  PATHS
    /Users/kmorel/data/supernova
  )
set(GRID_SIZE 432)

find_package(Dax REQUIRED)

find_package(Boost REQUIRED)

find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

configure_file(
  ${CMAKE_SOURCE_DIR}/MCScalingConfig.h.in
  ${CMAKE_BINARY_DIR}/MCScalingConfig.h
  )

include_directories(${CMAKE_BINARY_DIR})

DaxConfigureTBB(REQUIRED)
add_executable(MCScalingTbb
  MCScalingTbb.cxx
  )

DaxConfigureSerial(REQUIRED)
add_executable(MCScalingMPI
  MCScalingMPI.cxx
  )
set_target_properties(MCScalingMPI
  PROPERTIES
  COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}"
  LINK_FLAGS "${MPI_CXX_LINK_FLAGS}"
  )
target_link_libraries(MCScalingMPI
  ${MPI_CXX_LIBRARIES}
  )

