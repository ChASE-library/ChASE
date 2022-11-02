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
  MPI_Init(&argc,&argv);
  int rank = 0, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::size_t N = 1001; //problem size
  std::size_t nev = 100; //number of eigenpairs to be computed
  std::size_t nex = 10; //extra searching space
  
  int dims[2];
  dims[0] = dims[1] = 0;
  //MPI proc grid = dims[0] x dims[1]                                                                                                                                                                                         
  MPI_Dims_create(size, 2, dims);

#ifdef USE_BLOCK_CYCLIC
  /*parameters of block-cyclic data layout*/
  std::size_t NB = 32; //block size for block-cyclic data layout
  int irsrc = 0; 
  int icsrc = 0;
#endif

#ifdef USE_GIVEN_DIST
  //column major
  /*int tmp;
  tmp = dims[0];
  dims[0] = dims[1];
  dims[1] = tmp;
  */  
  std::size_t m, n;
  std::size_t len;
  int myrow = rank % dims[0];
  int mycol = rank / dims[0];
  if(N % dims[0] == 0){
      m = N / dims[0];
  }else{
      m =  std::min(N, N / dims[0] + 1);
  } 
  if(N % dims[1] == 0){
      n = N / dims[1];
  }else{
      n =  std::min(N, N / dims[1] + 1);
  }
  
#endif

  std::mt19937 gen(1337.0);
  std::normal_distribution<> d;

  /*construct eigenproblem to be solved*/
#ifdef USE_BLOCK_CYCLIC
  auto props = new ChaseMpiProperties<T>(N, NB, NB, nev, nex, dims[0], dims[1], (char *)"C", irsrc, icsrc, MPI_COMM_WORLD);
#elif defined(USE_GIVEN_DIST)
  auto props = new ChaseMpiProperties<T>(N, nev, nex, m, n, dims[0], dims[1], (char *)"C", MPI_COMM_WORLD);  
#elif defined(NO_COPY_H)
  auto props = new ChaseMpiProperties<T>(N, nev, nex, MPI_COMM_WORLD, false);
#else  
  auto props = new ChaseMpiProperties<T>(N, nev, nex, MPI_COMM_WORLD);
#endif

  auto m_ = props->get_m();

  auto V = std::vector<T>(m_ * (nev + nex)); //eigevectors
  auto Lambda = std::vector<Base<T>>(nev + nex); //eigenvalues

  std::vector<T> H(N * N, T(0.0));

  /*Generate Clement matrix*/
  for (auto i = 0; i < N; ++i) {
    H[i + N * i] = 0;
    if (i != N - 1) H[i + 1 + N * i] = std::sqrt(i * (N + 1 - i));
    if (i != N - 1) H[i + N * (i + 1)] = std::sqrt(i * (N + 1 - i));
  }

  if (rank == 0) {
      std::cout << "Starting Problem #1" << "\n";
  }

  std::cout << std::setprecision(16);

#ifdef USE_BLOCK_CYCLIC
  CHASE single(props, V.data(), Lambda.data());

  /*local block number = mblocks x nblocks*/
  std::size_t mblocks = single.get_mblocks();
  std::size_t nblocks = single.get_nblocks();

  /*local matrix size = m x n*/
  std::size_t m = single.get_m();
  std::size_t n = single.get_n();

  /*global and local offset/length of each block of block-cyclic data*/
  std::size_t *r_offs, *c_offs, *r_lens, *c_lens, *r_offs_l, *c_offs_l;
  single.get_offs_lens(r_offs, r_lens, r_offs_l, c_offs, c_lens, c_offs_l);


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

#elif defined(NO_COPY_H)
  std::size_t xoff, yoff, xlen, ylen, ldh;
  props->get_off(&xoff, &yoff, &xlen, &ylen);
  ldh = N / dims[0] + 1;
  T *h_loc = new T[ldh * ylen];
  for (std::size_t x = 0; x < xlen; x++) {
    for (std::size_t y = 0; y < ylen; y++) {
      h_loc[x + ldh * y] = H[(xoff + x) * N + (yoff + y)];
    }
  }

  CHASE single(props, h_loc, ldh, V.data(), Lambda.data());  

#else  
  CHASE single(props, V.data(), Lambda.data());

  std::size_t xoff, yoff, xlen, ylen;

  /*Get Offset and length of block of H on each node*/
  single.GetOff(&xoff, &yoff, &xlen, &ylen);

  /*Load different blocks of H to each node*/
  for (std::size_t x = 0; x < xlen; x++) {
    for (std::size_t y = 0; y < ylen; y++) {
      single.GetMatrixPtr()[x + xlen * y] = H.at((xoff + x) * N + (yoff + y));
    }
  }

#endif

  /*randomize V*/
  single.initRndVecs();

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
    std::cout << "Solving a symmetrized Clement matrices (" << N
              << "x" << N << ")"
#ifdef USE_BLOCK_CYCLIC       
              << " with block-cyclic data layout: " << NB << "x" << NB 
#endif
        << '\n'       
              << config;

  /*Performance Decorator to meaure the performance of kernels of ChASE*/
  PerformanceDecoratorChase<T> performanceDecorator(&single);
  /*Solve the eigenproblem*/
  chase::Solve(&performanceDecorator);
  /*
  //ortho
  T one(1.0);
  T zero(0.0);
  // Check eigenvector orthogonality
  auto unity = std::unique_ptr<T[]>(new T[nev * nev]);
  T neg_one(-1.0);
  for (int ttz = 0; ttz < nev; ttz++) {
    for (int tty = 0; tty < nev; tty++) {
      if (ttz == tty)
        unity[nev * ttz + tty] = 1.0;
      else
        unity[nev * ttz + tty] = 0.0;
    }
  }

  t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nev, nev, N, &one,
           V.data(), N, V.data(), N, &neg_one, unity.get(), nev);
  Base<T> norm = t_lange('M', nev, nev, unity.get(), nev);

  std::cout << "Eigenvectors Orthogonality: " << norm << std::endl;

  //residual
  auto prod = std::unique_ptr<T[]>(new T[N * nev]);

  t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, nev, N, &one, H.data(), N, V.data(), N, &neg_one, prod.get(), N);

  auto normv = std::unique_ptr<Base<T>[]> {
        new Base<T>[ nev ]
  }; 

  for(auto i =0; i < nev; i++){
    T alpha = -Lambda[i];
    t_axpy(N, &alpha, 
               V.data() + i * N, 1, 
               prod.get() + i * N, 1);

    normv[i] = t_nrm2(N, prod.get() + i * N, 1);

    std::cout << "norm " << i << " : " << normv[i] << "\n";
  }
*/
   /*Output*/
  if (rank == 0) {
    performanceDecorator.GetPerfData().print();
    Base<T>* resid = single.GetResid();
    std::cout << "Finished Problem #1" << "\n";
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

    MPI_Finalize();
}


