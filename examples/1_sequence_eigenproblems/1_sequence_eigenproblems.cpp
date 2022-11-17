/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials, 
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <complex>
#include <memory>
#include <random>
#include <vector>

/*include ChASE headers*/
#include "algorithm/performance.hpp"
#include "ChASE-MPI/chase_mpi.hpp"

#include "ChASE-MPI/impl/chase_mpidla_blaslapack.hpp"
#include "ChASE-MPI/impl/chase_mpidla_blaslapack_seq.hpp"
#include "ChASE-MPI/impl/chase_mpidla_blaslapack_seq_inplace.hpp"

using T = std::complex<double>;
using namespace chase;
using namespace chase::mpi;

/*use ChASE-MPI without GPU support*/
typedef ChaseMpi<ChaseMpiDLABlaslapack, T> CHASE;

int main(int argc, char** argv)
{
  MPI_Init(NULL, NULL);
  int rank = 0, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::size_t N = 4001; //problem size
  std::size_t nev = 500; //number of eigenpairs to be computed
  std::size_t nex = 100; //extra searching space
  std::size_t idx_max = 5; //number of eigenproblems to be solved in sequence
  Base<T> perturb = 1e-4; //perturbation of elements for the matrices in sequence

#ifdef USE_BLOCK_CYCLIC
  /*parameters of block-cyclic data layout*/
  std::size_t NB = 50; //block size for block-cyclic data layout
  int dims[2];
  dims[0] = dims[1] = 0;
  //MPI proc grid = dims[0] x dims[1]
  MPI_Dims_create(size, 2, dims);
  int irsrc = 0;
  int icsrc = 0;
#endif

  std::mt19937 gen(1337.0);
  std::normal_distribution<> d;

  if (rank == 0)
    std::cout << "ChASE example driver\n"
              << "Usage: ./driver \n";


  /*construct eigenproblem to be solved*/
#ifdef USE_BLOCK_CYCLIC
  auto props = new ChaseMpiProperties<T>(N, NB, NB, nev, nex, dims[0], dims[1], (char *)"C", irsrc, icsrc, MPI_COMM_WORLD);
#else
  auto props = new ChaseMpiProperties<T>(N, nev, nex, MPI_COMM_WORLD);
#endif

  auto m_ = props->get_m();
  auto V = std::vector<T>(m_ * (nev + nex)); //eigevectors
  auto Lambda = std::vector<Base<T>>(nev + nex); //eigenvalues

  CHASE single(props, V.data(), Lambda.data());

  /*Setup configure for ChASE*/
  auto& config = single.GetConfig();
  /*Tolerance for Eigenpair convergence*/
  config.SetTol(1e-10);
  /*Initial filtering degree*/
  config.SetDeg(20);
  /*Optimi(S)e degree*/
  config.SetOpt(true);
  /*If solving the problem with approximated eigenpairs*/
  config.SetApprox(false);

  if (rank == 0)
    std::cout << "Solving " << idx_max << " symmetrized Clement matrices (" << N
              << "x" << N << ") with element-wise random perturbation of "
	      << perturb
#ifdef USE_BLOCK_CYCLIC
              << " with block-cyclic data layout: " << NB << "x" << NB
#endif
	      << '\n'
              << config;

  std::vector<T> H(N * N, T(0.0));

  /*Generate Clement matrix*/
  for (auto i = 0; i < N; ++i) {
    H[i + N * i] = 0;
    if (i != N - 1) H[i + 1 + N * i] = std::sqrt(i * (N + 1 - i));
    if (i != N - 1) H[i + N * (i + 1)] = std::sqrt(i * (N + 1 - i));
  }

#ifdef USE_BLOCK_CYCLIC
  /*local block number = mblocks x nblocks*/
  std::size_t mblocks = single.get_mblocks();
  std::size_t nblocks = single.get_nblocks();

  /*local matrix size = m x n*/
  std::size_t m = single.get_m();
  std::size_t n = single.get_n();

  /*global and local offset/length of each block of block-cyclic data*/
  std::size_t *r_offs, *c_offs, *r_lens, *c_lens, *r_offs_l, *c_offs_l;
  single.get_offs_lens(r_offs, r_lens, r_offs_l, c_offs, c_lens, c_offs_l);
#else  
  std::size_t xoff, yoff, xlen, ylen;
  /*Get Offset and length of block of H on each node*/
  single.GetOff(&xoff, &yoff, &xlen, &ylen);
#endif

  /*randomize V*/
  single.initRndVecs(true);

  for (auto idx = 0; idx < idx_max; ++idx) {
    if (rank == 0) {
      std::cout << "Starting Problem #" << idx << "\n";
      if (config.UseApprox()) {
        std::cout << "Using approximate solution\n";
      }
    }

#ifdef USE_BLOCK_CYCLIC
    /*distribute Clement matrix into block cyclic data layout */
    for(std::size_t j = 0; j < nblocks; j++){
        for(std::size_t i = 0; i < mblocks; i++){
            for(std::size_t q = 0; q < c_lens[j]; q++){
                for(std::size_t p = 0; p < r_lens[i]; p++){
                    single.GetMatrixPtr()[(q + c_offs_l[j]) * m + p + r_offs_l[i]] = H[(q + c_offs[j]) * N + p + r_offs[i]];
                }
            }
        }
    }

#else    
    /*Load different blocks of H to each node*/
    for (std::size_t x = 0; x < xlen; x++) {
      for (std::size_t y = 0; y < ylen; y++) {
        single.GetMatrixPtr()[x + xlen * y] = H.at((xoff + x) * N + (yoff + y));
      }
    }
#endif

    /*Performance Decorator to meaure the performance of kernels of ChASE*/
    PerformanceDecoratorChase<T> performanceDecorator(&single);
    /*Solve the eigenproblem*/
    chase::Solve(&performanceDecorator);

    /*Output*/
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

    /*Generate next Clement matrix by the perturbation of elements*/
    for (std::size_t i = 1; i < N; ++i) {
      for (std::size_t j = 1; j < i; ++j) {
        T element_perturbation = T(d(gen), d(gen)) * perturb;
        H[j + N * i] += element_perturbation;
        H[i + N * j] += std::conj(element_perturbation);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Finalize();

}


