include (FetchContent)

FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.18.0.tar.gz
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# TODO: Add test executable
# add_executable(EXEC_NAME ...)
# target_link_libraries(EXEC_NAME ...)

include(GoogleTest)
# TODO: 
# gtest_discover_tests(EXEC_NAME)
