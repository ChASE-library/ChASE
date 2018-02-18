/* -*- Mode: C++; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set ts=8 sts=2 et sw=2 tw=80: */
/* TODO License */

#include <complex>
#include <random>
#include <vector>

#include "algorithm/chase.h"
#include "genera/matrixfree/blas_templates.h"
#include "genera/matrixfree/factory.h"

using T = std::complex<double>;
using namespace chase;

int main() {
  std::size_t N = 3000;
  std::size_t nev = 20;
  std::size_t nex = 20;
  std::size_t idx_max = 3;
  Base<T> perturb = 3e-9;

  std::cout << "ChASE example driver\n"
            << "Usage: ./driver \n";

  ChaseConfig<T> config(N, nev, nex);
  config.setTol(1e-8);
  config.setDeg(20);
  config.setOpt(true);
  config.setApprox(false);

  auto V__ = std::unique_ptr<T[]>(new T[N * (nev + nex)]);
  auto Lambda__ = std::unique_ptr<Base<T>[]>(new Base<T>[(nev + nex)]);

  T* V = V__.get();
  Base<T>* Lambda = Lambda__.get();

  std::unique_ptr<MatrixFreeChase<T>> single_ = constructChASE(
      config, static_cast<T*>(nullptr), V, Lambda, MPI_COMM_WORLD);

  T* H = single_->GetMatrixPtr();

  // std::unique_ptr<MatrixFreeChase<T>> single_ =
  //   constructChASE<MatrixFreeBlas, T>(config, H, V, Lambda);

  // randomize V
  std::mt19937 gen(1337.0);
  std::normal_distribution<> d;
  for (std::size_t i = 0; i < N * (nev + nex); ++i) {
    V[i] = getRandomT<T>([&]() { return d(gen); });
  }

  std::cout << "Solving " << idx_max << " Wilkinson matrices (" << N << "x" << N
            << ") with element-wise random perturbation of " << perturb
            << '\n';
  std::cout << config;

  for (auto idx = 0; idx < idx_max; ++idx) {
    std::cout << "Staring Problem #" << idx << "\n";
    if (config.use_approx()) {
      std::cout << "Using approximate solution\n";
    }

    // fill Wilkinson matrix
    // singed integers make the indexing straight-forward
    for (signed long i = 0; i < N; ++i) {
      H[i + N * i] = std::abs(static_cast<signed long>(N / 2) - i);
      if (i != N - 1) H[i + 1 + N * i] = 1;
      if (i != 0) H[i - 1 + N * i] = 1;
    }

    // perturb matrix
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = 0; j < N; ++j) {
        H[j + N * i] += getRandomT<T>([&]() { return d(gen); }) * perturb;
        H[i + N * j] = std::conj(H[j + N * i]);
      }

    single_.get()->Solve();
    Base<T>* resid = single_.get()->getResid();

    std::cout << "Finished Problem #" << idx << "\n";
    std::cout << "| Index |       Eigenvalue      |         Residual      |\n"
              << "|-------|-----------------------|-----------------------|\n";
    std::size_t width = 20;
    std::cout << std::setprecision(12);
    std::cout << std::setfill(' ');
    std::cout << std::scientific;
    std::cout << std::right;
    for (auto i = 0; i < nev; ++i)
      std::cout << "|  " << std::setw(4) << i + 1 << " | " << std::setw(width)
                << Lambda[i] << "  | " << std::setw(width) << resid[i]
                << "  |\n";
    std::cout << "\n\n\n";

    config.setApprox(true);
  }
}
