add_library(sdnn)

target_sources(sdnn
  PRIVATE
    version.cpp
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/include
    FILES
      ${PROJECT_SOURCE_DIR}/include/sdnn/version.hpp
)

target_compile_features(sdnn PUBLIC cxx_std_20)

target_compile_definitions(sdnn PRIVATE SDNN_VERSION="${PROJECT_VERSION}")

target_link_libraries(sdnn
  PRIVATE
    OpenMP::OpenMP_CXX
    Threads::Threads
)

if(SDNN_ENABLE_CUDA)
  target_sources(sdnn
    PRIVATE
      #TODO
  )

  target_compile_features(sdnn PUBLIC cuda_std_20)

  target_link_libraries(sdnn PRIVATE CUDA::cudart)

  target_compile_definitions(sdnn PUBLIC CUDA_ENABLED)
endif()
