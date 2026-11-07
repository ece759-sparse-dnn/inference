#include <gtest/gtest.h>

#include "sdnn/version.hpp"

TEST(Version, Nonempty) { EXPECT_FALSE(sdnn::version().empty()); }
