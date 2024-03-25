/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2023, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <complex>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

#include "ChASE-MPI/chase_mpi.hpp"
#include "algorithm/performance.hpp"

#include "ChASE-MPI/impl/chase_mpidla_blaslapack_seq.hpp"
#include "ChASE-MPI/impl/chase_mpidla_blaslapack_seq_inplace.hpp"
#if defined(USE_GPU)
#include "ChASE-MPI/impl/chase_mpidla_cuda_seq.hpp"
#endif

using T = std::complex<double>;
// using T = double;
using namespace chase;
using namespace chase::mpi;

#if defined(USE_GPU)
typedef ChaseMpi<ChaseMpiDLACudaSeq, T> CHASE;
#else
typedef ChaseMpi<ChaseMpiDLABlaslapackSeq, T> CHASE;
//typedef ChaseMpi<ChaseMpiDLABlaslapackSeqInplace, T> CHASE;
#endif

int main()
{
    MPI_Init(NULL, NULL);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::size_t N = 1001;
    std::size_t LDH = 1001;
    std::size_t nev = 80;
    std::size_t nex = 60;
    std::size_t idx_max = 5;
    Base<T> perturb = 1e-4;

    std::mt19937 gen(1337.0);
    std::normal_distribution<> d;

    if (rank == 0)
        std::cout << "ChASE example driver\n"
                  << "Usage: ./driver \n";

    auto V = std::vector<T>(N * (nev + nex));
    auto Lambda = std::vector<Base<T>>(nev + nex);
    std::vector<T> H(N * LDH, T(0.0));

    CHASE single(N, nev, nex, H.data(), LDH, V.data(), Lambda.data());

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

    // Generate Clement matrix
    for (auto i = 0; i < N; ++i)
    {
        H[i + N * i] = 0;
        if (i != N - 1)
            H[i + 1 + LDH * i] = std::sqrt(i * (N + 1 - i));
        if (i != N - 1)
            H[i + LDH * (i + 1)] = std::sqrt(i * (N + 1 - i));
    }

    for (auto idx = 0; idx < idx_max; ++idx)
    {
        if (rank == 0)
        {
            std::cout << "Starting Problem #" << idx << "\n";
            if (config.UseApprox())
            {
                std::cout << "Using approximate solution\n";
            }
        }

        PerformanceDecoratorChase<T> performanceDecorator(&single);
        chase::Solve(&performanceDecorator);

        if (rank == 0)
        {
            performanceDecorator.GetPerfData().print();
            Base<T>* resid = single.GetResid();
            std::cout << "Finished Problem #" << idx << "\n";
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

        config.SetApprox(true);
        // Perturb Full Clement matrix
        for (std::size_t i = 1; i < N; ++i)
        {
            for (std::size_t j = 1; j < i; ++j)
            {
                T element_perturbation = T(d(gen), d(gen)) * perturb;
                H[j + LDH * i] += element_perturbation;
                H[i + LDH * j] += std::conj(element_perturbation);
            }
        }
        std::vector<T> H_2(N * N);
        std::memcpy(H_2.data(), H.data(), N * N * sizeof(T));
        std::vector<Base<T>> ritzv(N);
        t_heevd(LAPACK_COL_MAJOR, 'N', 'U', N, H_2.data(), N, ritzv.data());
        std::cout << "real max rtizv = " << *std::max_element(ritzv.begin(), ritzv.end()) << std::endl;;
    }
}
