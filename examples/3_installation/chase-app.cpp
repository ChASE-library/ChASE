/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2023, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <complex>
#include <memory>
#include <random>
#include <vector>

#include "ChASE-MPI/chase_mpi.hpp"
#include "algorithm/performance.hpp"

#include "ChASE-MPI/impl/chase_mpidla_blaslapack.hpp"

#ifdef DRIVER_BUILD_MGPU
#include "ChASE-MPI/impl/chase_mpidla_cuda_seq.hpp"
#include "ChASE-MPI/impl/chase_mpidla_mgpu.hpp"
#endif

using T = std::complex<double>;
using namespace chase;
using namespace chase::mpi;

#ifdef DRIVER_BUILD_MGPU
typedef ChaseMpi<ChaseMpiDLAMultiGPU, T> CHASE;
#else
typedef ChaseMpi<ChaseMpiDLABlaslapack, T> CHASE;
#endif

int main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::size_t N = 1001;
    std::size_t nev = 100;
    std::size_t nex = 20;
    std::size_t idx_max = 3;
    Base<T> perturb = 1e-4;

    if (rank == 0)
        std::cout << "ChASE example driver\n"
                  << "Usage: ./driver \n";

    auto props = new ChaseMpiProperties<T>(N, nev, nex, MPI_COMM_WORLD);
    auto m = props->get_m();
    auto V = std::vector<T>(m * (nev + nex));
    auto Lambda = std::vector<Base<T>>(nev + nex);

    CHASE single(props, V.data(), Lambda.data());

    auto& config = single.GetConfig();
    config.SetTol(1e-10);
    config.SetDeg(20);
    config.SetOpt(true);
    config.SetApprox(false);

    if (rank == 0)
        std::cout << "Solving " << idx_max << " symmetrized Clement matrices ("
                  << N << "x" << N
                  << ") with element-wise random perturbation of " << perturb
                  << '\n'
                  << config;

    std::size_t xoff, yoff, xlen, ylen;

    props->get_off(&xoff, &yoff, &xlen, &ylen);

    std::vector<T> H(N * N, T(0.0));
    for (auto i = 0; i < N; ++i)
    {
        H[i + N * i] = 0;
        if (i != N - 1)
            H[i + 1 + N * i] = std::sqrt(i * (N + 1 - i));
        if (i != N - 1)
            H[i + N * (i + 1)] = std::sqrt(i * (N + 1 - i));
    }

    if (rank == 0)
    {
        std::cout << "Starting Problem"
                  << "\n";
        if (config.UseApprox())
        {
            std::cout << "Using approximate solution\n";
        }
    }

    for (std::size_t x = 0; x < xlen; x++)
    {
        for (std::size_t y = 0; y < ylen; y++)
        {
            single.GetMatrixPtr()[x + xlen * y] =
                H.at((xoff + x) * N + (yoff + y));
        }
    }

    PerformanceDecoratorChase<T> performanceDecorator(&single);
    chase::Solve(&performanceDecorator);

    if (rank == 0)
    {
        performanceDecorator.GetPerfData().print();
        Base<T>* resid = single.GetResid();
        std::cout << "Finished Problem"
                  << "\n";
        std::cout << "Printing first 5 eigenvalues and residuals\n";
        std::cout
            << "| Index |       Eigenvalue      |         Residual      |\n"
            << "|-------|-----------------------|-----------------------|"
               "\n";
        std::size_t width = 20;
        std::cout << std::setprecision(12);
        std::cout << std::setfill(' ');
        std::cout << std::scientific;
        std::cout << std::right;
        for (auto i = 0; i < std::min(std::size_t(5), nev); ++i)
            std::cout << "|  " << std::setw(4) << i + 1 << " | "
                      << std::setw(width) << Lambda[i] << "  | "
                      << std::setw(width) << resid[i] << "  |\n";
        std::cout << "\n\n\n";
    }

    MPI_Finalize();
}
