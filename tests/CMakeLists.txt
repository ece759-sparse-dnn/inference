include (FetchContent)

FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.18.0.tar.gz
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(sdnn_tests
  test_version.cpp
)
target_link_libraries(sdnn_tests PRIVATE sdnn GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(sdnn_tests)
