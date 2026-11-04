cmake_minimum_required(VERSION 3.24...4.4)

project(sdnn 
  VERSION 0.1.0 
  LANGUAGES CXX)

#------------------------------------------------------------------------------#
#                                   Options
#------------------------------------------------------------------------------#

option(SDNN_ENABLE_CUDA "Build the CUDA backend" OFF)

if (SDNN_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolKit REQUIRED)
endif()

#------------------------------------------------------------------------------#
#                                 Dependencies
#------------------------------------------------------------------------------#

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

#------------------------------------------------------------------------------#
#                                   Targets
#------------------------------------------------------------------------------#

add_subdirectory(src)
add_subdirectory(apps)

#------------------------------------------------------------------------------#
#                                   Testing
#------------------------------------------------------------------------------#

if(PROJECT_IS_TOP_LEVEL)
  include(CTest)
endif()

if(PROJECT_IS_TOP_LEVEL AND BUILD_TESTING)
  add_subdirectory(tests)
endif()
