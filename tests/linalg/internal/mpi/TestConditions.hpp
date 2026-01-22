// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "algorithm/types.hpp"
#include "external/blaspp/blaspp.hpp"
#include "grid/mpiTypes.hpp"
#include "mpi.h"

using namespace chase::linalg;

template <typename T>
chase::Base<T> orthogonality(std::size_t m, std::size_t n, T* V,
                             std::size_t ldv, MPI_Comm comm)
{
    T one = T(1.0);
    T negone = T(-1.0);
    T zero = T(0.0);

    std::vector<T> A(n * n);

    blaspp::t_gemm<T>(1, CblasConjTrans, CblasNoTrans, n, n, m, &one, V, ldv, V,
                      ldv, &zero, A.data(), n);

    MPI_Allreduce(MPI_IN_PLACE, A.data(), n * n, chase::mpi::getMPI_Type<T>(),
                  MPI_SUM, comm);

    std::size_t incx = 1, incy = n + 1;
    std::vector<T> hI(n, T(1.0));

    blaspp::t_axpy(n, &negone, hI.data(), incx, A.data(), incy);

    chase::Base<T> nrmf = blaspp::t_nrm2(n * n, A.data(), 1);

    return (nrmf / std::sqrt(n));
}