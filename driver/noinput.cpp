/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2018, Simulation Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany
// and
// Copyright (c) 2016-2018, Aachen Institute for Advanced Study in Computational
//   Engineering Science, RWTH Aachen University, Germany All rights reserved.
// License is 3-clause BSD:
// https://github.com/SimLabQuantumMaterials/ChASE/

#include <complex>
#include <memory>
#include <random>
#include <vector>

#include "ChASE-MPI/chase_mpi.hpp"

#include "ChASE-MPI/impl/chase_mpihemm_blas.hpp"
#include "ChASE-MPI/impl/chase_mpihemm_blas_seq.hpp"
#include "ChASE-MPI/impl/chase_mpihemm_blas_seq_inplace.hpp"

#ifdef BUILD_CUDA
#include "ChASE-MPI/impl/chase_mpihemm_cuda.hpp"
#include "ChASE-MPI/impl/chase_mpihemm_cuda_seq.hpp"
#endif

using T = std::complex<double>;
using namespace chase;
using namespace chase::mpi;

#ifdef USE_MPI
typedef ChaseMpi<ChaseMpiHemmBlas, T> CHASE;
#else
typedef ChaseMpi<ChaseMpiHemmBlasSeq, T> CHASE;
#endif

int main() {
  MPI_Init(NULL, NULL);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::size_t N = 1001;
  std::size_t nev = 100;
  std::size_t nex = 20;
  std::size_t idx_max = 3;
  Base<T> perturb = 1e-4;

  std::mt19937 gen(1337.0);
  std::normal_distribution<> d;

  if (rank == 0)
    std::cout << "ChASE example driver\n"
              << "Usage: ./driver \n";

  auto V = std::vector<T>(N * (nev + nex));
  auto Lambda = std::vector<Base<T>>(nev + nex);

#ifdef USE_MPI
  CHASE single(new ChaseMpiProperties<T>(N, nev, nex, MPI_COMM_SELF), V.data(),
               Lambda.data());
#else
  CHASE single(N, nev, nex, V.data(), Lambda.data());
#endif

  auto& config = single.GetConfig();
  config.SetTol(1e-10);
  config.SetDeg(20);
  config.SetOpt(true);
  config.SetApprox(false);

  if (rank == 0)
    std::cout << "Solving " << idx_max << " symmetrized Clement matrices (" << N
              << "x" << N << ") with element-wise random perturbation of "
              << perturb << '\n'
              << config;

  // randomize V
  for (std::size_t i = 0; i < N * (nev + nex); ++i) {
    V[i] = T(d(gen), d(gen));
  }

  std::size_t xoff, yoff, xlen, ylen;
#ifdef USE_MPI
  single.GetOff(&xoff, &yoff, &xlen, &ylen);
#else
  xoff = 0;
  yoff = 0;
  xlen = N;
  ylen = N;
#endif

  // Generate Clement matrix
  std::vector<T> H(N * N, T(0.0));
  for (auto i = 0; i < N; ++i) {
    H[i + N * i] = 0;
    if (i != N - 1) H[i + 1 + N * i] = std::sqrt(i * (N + 1 - i));
    if (i != N - 1) H[i + N * (i + 1)] = std::sqrt(i * (N + 1 - i));
  }

  for (auto idx = 0; idx < idx_max; ++idx) {
    if (rank == 0) {
      std::cout << "Starting Problem #" << idx << "\n";
      if (config.UseApprox()) {
        std::cout << "Using approximate solution\n";
      }
    }

    // Load matrix into distributed buffer
    for (std::size_t x = 0; x < xlen; x++) {
      for (std::size_t y = 0; y < ylen; y++) {
        single.GetMatrixPtr()[x + xlen * y] = H.at((xoff + x) * N + (yoff + y));
      }
    }

    PerformanceDecoratorChase<T> performanceDecorator(&single);
    chase::Solve(&performanceDecorator);

    if (rank == 0) {
      performanceDecorator.GetPerfData().print();
      Base<T>* resid = single.GetResid();
      std::cout << "Finished Problem #" << idx << "\n";
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
        std::cout << "|  " << std::setw(4) << i + 1 << " | " << std::setw(width)
                  << Lambda[i] << "  | " << std::setw(width) << resid[i]
                  << "  |\n";
      std::cout << "\n\n\n";
    }

    config.SetApprox(true);
    // Perturb Full Clement matrix
    for (std::size_t i = 1; i < N; ++i) {
      for (std::size_t j = 1; j < i; ++j) {
        T element_perturbation = T(d(gen), d(gen)) * perturb;
        H[j + N * i] += element_perturbation;
        H[i + N * j] += std::conj(element_perturbation);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
}
