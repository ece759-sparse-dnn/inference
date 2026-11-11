include (FetchContent)

FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.18.0.tar.gz
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(sdnn_tests
  test_dataset.cpp
  test_version.cpp
)
target_link_libraries(sdnn_tests PRIVATE sdnn GTest::gtest_main)
target_compile_definitions(sdnn_tests PRIVATE
  SDNN_DEFAULT_DATA_DIR="${PROJECT_SOURCE_DIR}/data")

include(GoogleTest)
gtest_discover_tests(sdnn_tests)
