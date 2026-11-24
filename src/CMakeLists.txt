add_library(sdnn)

target_sources(sdnn
  PRIVATE
    backends.cpp
    csr.cpp
    dataset.cpp
    dense.cpp
    npy.cpp
    openmp.cpp
    serial.cpp
    version.cpp
    ell.cpp
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/include
    FILES
      ${PROJECT_SOURCE_DIR}/include/sdnn/backend.hpp
      ${PROJECT_SOURCE_DIR}/include/sdnn/csr.hpp
      ${PROJECT_SOURCE_DIR}/include/sdnn/dataset.hpp
      ${PROJECT_SOURCE_DIR}/include/sdnn/dense.hpp
      ${PROJECT_SOURCE_DIR}/include/sdnn/npy.hpp
      ${PROJECT_SOURCE_DIR}/include/sdnn/version.hpp
      ${PROJECT_SOURCE_DIR}/include/sdnn/ell.hpp
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
      cuda.cu
      cuda/kernels.cu
      cuda/check.cuh
      cuda/device_buffer.cuh
      cuda/device_ell.cuh
      cuda/device_views.cuh
      cuda/kernels.cuh
  )

  target_compile_features(sdnn PUBLIC cuda_std_20)

  target_link_libraries(sdnn PRIVATE CUDA::cudart)

  target_compile_definitions(sdnn PUBLIC CUDA_ENABLED)
endif()
