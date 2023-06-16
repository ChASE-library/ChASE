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
    MPI_Init(&argc, &argv);
    int rank = 0, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::size_t N = 1001; // problem size
    std::size_t nev = 100; // number of eigenpairs to be computed
    std::size_t nex = 40; // extra searching space


    auto props = new ChaseMpiProperties<T>(N, nev, nex, MPI_COMM_WORLD);

    auto m_ = props->get_m();
    auto n_ = props->get_n();
    auto ldh_ = props->get_ldh();

    auto V = std::vector<T>(m_ * (nev + nex));     // eigevectors
    auto Lambda = std::vector<Base<T>>(nev + nex); // eigenvalues
    auto H = std::vector<T>(ldh_ * n_);

    CHASE single(props, H.data(), ldh_, V.data(), Lambda.data());

    std::vector<T> Clement(N * N, T(0.0));

    /*Generate Clement matrix*/
    for (auto i = 0; i < N; ++i)
    {
        H[i + N * i] = 0;
        if (i != N - 1)
            Clement[i + 1 + N * i] = std::sqrt(i * (N + 1 - i));
        if (i != N - 1)
            Clement[i + N * (i + 1)] = std::sqrt(i * (N + 1 - i));
    }

    if (rank == 0)
    {
        std::cout << "Starting Problem #1"
                  << "\n";
    }

    std::cout << std::setprecision(16);

    std::size_t xoff, yoff, xlen, ylen;

    /*Get Offset and length of block of H on each node*/
    props->get_off(&xoff, &yoff, &xlen, &ylen);

    /*Load different blocks of H to each node*/
    for (std::size_t x = 0; x < xlen; x++)
    {
        for (std::size_t y = 0; y < ylen; y++)
        {
            H[x + xlen * y] =
                Clement[(xoff + x) * N + (yoff + y)];
        }
    }

    /*Setup configure for ChASE*/
    auto& config = single.GetConfig();
    /*Tolerance for Eigenpair convergence*/
    config.SetTol(1e-10);
    /*Initial filtering degree*/
    config.SetDeg(20);
    /*Optimi(S)e degree*/
    config.SetOpt(true);
    config.SetMaxIter(25);
    if (rank == 0)
        std::cout << "Solving a symmetrized Clement matrices (" << N << "x" << N
                  << ")"
                  << '\n'
                  << config;

    /*Performance Decorator to meaure the performance of kernels of ChASE*/
    PerformanceDecoratorChase<T> performanceDecorator(&single);
    /*Solve the eigenproblem*/
    chase::Solve(&performanceDecorator);

    /*Output*/
    if (rank == 0)
    {
        performanceDecorator.GetPerfData().print();
        Base<T>* resid = single.GetResid();
        std::cout << "Finished Problem #1"
                  << "\n";
        std::cout << "Printing first 5 eigenvalues and residuals\n";
        std::cout
            << "| Index |       Eigenvalue      |         Residual      |\n"
            << "|-------|-----------------------|-----------------------|\n";
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
