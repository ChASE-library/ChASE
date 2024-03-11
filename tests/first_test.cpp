// Must include the gtest header to use the testing library
#include <gtest/gtest.h>
#include <mpi.h>

namespace {
  // test dummy function
  int GetMeaningOfLife() {  return 42; }
}

TEST(TestTopic, TrivialEquality) {
  EXPECT_EQ(GetMeaningOfLife(), 42);
}

TEST(TestTopic, MoreEqualityTests) {
  ASSERT_EQ(GetMeaningOfLife(), 42) << "Oh no, a mistake!";
  EXPECT_FLOAT_EQ(23.23F, 23.23F);
}

TEST(TestMPIranks, Ranks) {
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  ASSERT_EQ(size, 4) << "Not same number of ranks";
}