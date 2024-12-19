// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <gtest/gtest.h>
#include <mpi.h>

int main(int argc, char* argv[]) {

    ::testing::InitGoogleTest(&argc, argv);
   
   MPI_Init(&argc, &argv);

   const int result = RUN_ALL_TESTS();

   MPI_Finalize();
   return result;
}
