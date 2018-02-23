/* -*- Mode: C++; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set ts=8 sts=2 et sw=2 tw=80: */
/* TODO License */

#include <complex>
#include <memory>
#include <random>
#include <vector>

#include "algorithm/chase.h"
#include "genera/matrixfree/blas_templates.h"
#include "genera/matrixfree/chase.h"

#include "genera/matrixfree/impl/blas/matrixFreeBlas.hpp"
#include "genera/matrixfree/impl/blasInplace/matrixFreeBlasInplace.hpp"
#include "genera/matrixfree/impl/blasSkewed/matrixFreeBlas.hpp"
#include "genera/matrixfree/impl/cuda/matrixFreeCuda.hpp"

using T = std::complex<double>;
using namespace chase;
using namespace chase::matrixfree;

#ifdef USE_MPI
typedef MatrixFreeChase<MatrixFreeBlasSkewed, T> MFC;
#else
typedef MatrixFreeChase<MatrixFreeBlasInplace, T> MFC;
#endif

int main() {
  MPI_Init(NULL, NULL);

  std::size_t N = 501;
  std::size_t nev = 100;
  std::size_t nex = 20;
  std::size_t idx_max = 3;
  Base<T> perturb = 1e-5;

  std::mt19937 gen(1337.0);
  std::normal_distribution<> d;

  std::cout << "ChASE example driver\n"
            << "Usage: ./driver \n";

  auto V = std::vector<T>(N * (nev + nex));
  auto Lambda = std::vector<Base<T>>(nev + nex);

#if USE_MPI
  auto single =
      MFC(config, new SkewedMatrixProperties<T>(N, nev + nex, MPI_COMM_SELF),
          V.data(), Lambda.data());
#else
  auto single = MFC(N, nev, nex, V.data(), Lambda.data());
#endif

  T* H = single.GetMatrixPtr();
  auto& config = single.GetConfig();
  config.setTol(1e-10);
  config.setDeg(20);
  config.setOpt(true);
  config.setApprox(false);

  std::cout << "Solving " << idx_max << " symmetrized Clement matrices (" << N
            << "x" << N << ") with element-wise random perturbation of "
            << perturb << '\n';
  std::cout << config;

  // randomize V
  for (std::size_t i = 0; i < N * (nev + nex); ++i) {
    V[i] = T(d(gen), d(gen));
  }

  // fill matrix
  for (auto i = 0; i < N; ++i) {
    H[i + N * i] = 0;
    if (i != N - 1) H[i + 1 + N * i] = std::sqrt(i * (N + 1 - i));
    if (i != 0) H[i + N * (i + 1)] = std::sqrt(i * (N + 1 - i));
  }

  for (auto idx = 0; idx < idx_max; ++idx) {
    std::cout << "Staring Problem #" << idx << "\n";
    if (config.use_approx()) {
      std::cout << "Using approximate solution\n";
    }

    // perturb matrix
    T element_perturbation;
    for (std::size_t i = 1; i < N; ++i)
      for (std::size_t j = 1; j < i; ++j) {
        element_perturbation = T(d(gen), d(gen)) * perturb;
        H[j + N * i] += element_perturbation;
        H[i + N * j] += std::conj(element_perturbation);
      }

    // single.Solve();
    chase::Solve(&single);
    Base<T>* resid = single.GetResid();

    std::cout << "Finished Problem #" << idx << "\n";
    std::cout << "Printing first 5 eigenvalues and residuals\n";
    std::cout << "| Index |       Eigenvalue      |         Residual      |\n"
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

    config.setApprox(true);
  }
}
