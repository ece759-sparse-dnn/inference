add_executable(sdnn_run sdnn_run.cpp)

target_link_libraries(sdnn_run PRIVATE sdnn OpenMP::OpenMP_CXX)

target_compile_definitions(sdnn_run PRIVATE
  SDNN_DEFAULT_DATA_DIR="${PROJECT_SOURCE_DIR}/data")
