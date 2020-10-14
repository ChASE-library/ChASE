/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2018, Simulation Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany
// and
// Copyright (c) 2016-2018, Aachen Institute for Advanced Study in Computational
//   Engineering Science, RWTH Aachen University, Germany All rights reserved.
// License is 3-clause BSD:
// https://github.com/SimLabQuantumMaterials/ChASE/

#include <boost/program_options.hpp>
#include <limits>
#include <complex>
#include <memory>
#include <random>
#include <vector>
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream

#include "algorithm/performance.hpp"
#include "ChASE-MPI/chase_mpi.hpp"

#include "ChASE-MPI/impl/chase_mpihemm_blas_seq.hpp"
#include "ChASE-MPI/impl/chase_mpihemm_blas_seq_inplace.hpp"

#ifdef USE_MPI
#include "ChASE-MPI/impl/chase_mpihemm_blas.hpp"
  #ifdef DRIVER_BUILD_MGPU
  #include "ChASE-MPI/impl/chase_mpihemm_mgpu.hpp"
  #endif
#endif

using T = std::complex<double>;
using namespace chase;
using namespace chase::mpi;

#ifdef USE_MPI
    #ifdef DRIVER_BUILD_MGPU
        typedef ChaseMpi<ChaseMpiHemmMultiGPU, T> CHASE;
    #else
        typedef ChaseMpi<ChaseMpiHemmBlas, T> CHASE;
    #endif //CUDA or not
#else
    typedef ChaseMpi<ChaseMpiHemmBlasSeq, T> CHASE;
#endif //seq ChASE

namespace po = boost::program_options;

struct ChASE_DriverProblemConfig {
  std::size_t N;    // Size of the Matrix
  std::size_t nev;  // Number of sought after eigenvalues
  std::size_t nex;  // Extra size of subspace
  std::size_t deg;  // initial degree
  std::size_t bgn;  // beginning of sequence
  std::size_t end;  // end of sequence

  double tol;     // desired tolerance
  bool sequence;  // handle this as a sequence?

  std::string path_in;    // path to the matrix input files
  std::string input;    // path to the matrix input files
  std::string output;
  std::string mode;       // Approx or Random mode
  std::string opt;        // enable optimisation of degree
  std::string arch;       // ??
  std::string path_eigp;  // TODO
  std::string path_out;
  std::string path_name;
  std::string test_name;
  
  Base<T> perturb;

  std::size_t kpoint;
  bool legacy;
  std::string spin;

  bool complex;
  bool isdouble;
};

template <typename T>
void generateClement(std::vector<T> &H, std::size_t N) {
  std::cout << "size = " << N << std::endl;
  // Generate Clement matrix
  for (auto i = 0; i < N; ++i) {
    H[i + N * i] = 0;
    if (i != N - 1) H[i + 1 + N * i] = 1.0 + std::sqrt(i * (N + 1 - i));
    if (i != N - 1) H[i + N * (i + 1)] = std::sqrt(i * (N + 1 - i));
  }
}

template <typename T>
void readMatrix(T* H, std::string path_in, std::size_t size) {
  std::ostringstream problem(std::ostringstream::ate);
  problem << path_in;

  std::cout << "start reading matrix\n";
  std::cout << problem.str() << std::endl;
  std::cout << "size = " << size << std::endl;
  std::ifstream input(problem.str().c_str(), std::ios::binary);
  if (input.is_open()) {
    input.read((char*)H, sizeof(T) * size);
  } else {
    throw std::string("error reading file: ") + problem.str();
  }
  std::cout << "done reading matrix\n";
}

template <typename T>
void readMatrix(T* H, std::string path_in, std::size_t size,
                std::size_t xoff, std::size_t yoff,
                std::size_t xlen, std::size_t ylen) {
  std::size_t N = std::sqrt(size);
  std::ostringstream problem(std::ostringstream::ate);
  problem << path_in;

  std::cout << problem.str() << std::endl;
  std::ifstream input(problem.str().c_str(), std::ios::binary);
  if (!input.is_open()) {
    throw new std::logic_error(std::string("error reading file: ") +
                               problem.str());
  }

  for (std::size_t y = 0; y < ylen; y++) {
    input.seekg(((xoff) + N * (yoff + y)) * sizeof(T));
    input.read(reinterpret_cast<char*>(H + xlen * y), xlen * sizeof(T));
  }
}

int main(int argc, char* argv[]) {
  int rank = 0;

#ifdef USE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  ChASE_DriverProblemConfig conf;

  po::options_description desc("ChASE Simple Driver Options");
  desc.add_options()(                                                     //
      "help,h",                                                           //
      "show this message"                                                 //
      )(                                                                  //
      "n", po::value<std::size_t>(&conf.N)->default_value(1001),          //
      "Size of the Input Matrix"                                          //
      )(                                                                  //
      "double", po::value<bool>(&conf.isdouble)->default_value(true),     //
      "Is matrix complex double valued, false indicates the single type"  //
      )(                                                                  //
      "complex", po::value<bool>(&conf.complex)->default_value(true),     //
      "Matrix is complex valued"                                          //
      )(                                                                  //
      "nev", po::value<std::size_t>(&conf.nev)->default_value(100),       //
      "Wanted Number of Eigenpairs"                                       //
      )(                                                                  //
      "nex", po::value<std::size_t>(&conf.nex)->default_value(25),        //
      "Extra Search Dimensions"                                           //
      )(                                                                  //
      "deg", po::value<std::size_t>(&conf.deg)->default_value(20),        //
      "Initial filtering degree"                                          //
      )(                                                                  //
      "tol", po::value<double>(&conf.tol)->default_value(1e-10),          //
      "Tolerance for Eigenpair convergence"                               //
      )(                                                                  //
      "input", po::value<std::string>(&conf.input)->default_value(""),    //
      "Path to the input matrix/matrices"                                 //
      )(                                                                  //
      "output", po::value<std::string>(&conf.output)->default_value("eig.txt"),    //
      "Path to the write the eigenvalues"                                 //
      )(                                                                  //
      "mode", po::value<std::string>(&conf.mode)->default_value("R"),     //
      "valid values are R(andom) or A(pproximate)"                        //
      )(                                                                  //
      "opt", po::value<std::string>(&conf.opt)->default_value("N"),       //
      "Optimi(S)e degree, or do (N)ot optimise"                           //
      )(                                                                  //
      "perturb", po::value<Base<T>>(&conf.perturb)->default_value(0),     //
      "Perturbation of eigenvalues used for second run"                   //
      );                                                                  //

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);

  // print help
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  po::notify(vm);

  // some normalization
  conf.mode = toupper(conf.mode.at(0));
  conf.opt = toupper(conf.opt.at(0));

  // Additional Error checks
  // TODO this should be a member of struct

  if (conf.mode != "R" && conf.mode != "A") {
    std::cout << "Illegal value for mode: \"" << conf.mode << "\"" << std::endl
              << "Legal values are R or A" << std::endl;
    return -1;
  }

  if (conf.opt != "N" && conf.opt != "S") {
    std::cout << "Illegal value for opt: " << conf.opt << std::endl
              << "Legal values are N, S" << std::endl;
    return -1;
  }

  if (conf.input != "" && conf.N == 1001) {
    std::cout << "Ensure to manully set option --n with the correct matrix size. Currently n= " << conf.N << std::endl;
  }

  std::size_t N = conf.N;
  std::size_t nev = conf.nev;
  std::size_t nex = conf.nex;
  std::size_t idx_max=2;
  if( conf.perturb > 1e-20 ) {
    idx_max= 2;}
  else {
    idx_max = 1;}
//   Base<T> perturb = 1e-4;

  std::mt19937 gen(1337.0);
  std::normal_distribution<> d;

  if (rank == 0)
    std::cout << "ChASE example driver\n"
              << "Usage: ./driver \n";

  auto V = std::vector<T>(N * (nev + nex));
  auto Lambda = std::vector<Base<T>>(nev + nex);

#ifdef USE_MPI
  CHASE single(new ChaseMpiProperties<T>(N, nev, nex, MPI_COMM_WORLD), V.data(),
               Lambda.data());
#else
  CHASE single(N, nev, nex, V.data(), Lambda.data());
#endif

  auto& config = single.GetConfig();
  config.SetTol(conf.tol);
  config.SetDeg(conf.deg);
//   config.SetOpt(true);
  config.SetOpt(conf.opt == "S");
  config.SetApprox(conf.mode == "A");

  if (rank == 0) {
    if(conf.input == "") {
      std::cout << "Solving " << idx_max << " symmetrized Clement matrices (" << N
                << "x" << N << ") with element-wise random perturbation of "
                << conf.perturb << '\n'
                << config;
    } else {
      std::cout << "Solving input matrix " << conf.input << " (" << N
                << "x" << N << ") with element-wise random perturbation of "
                << conf.perturb << '\n'
                << config;
    }
  }

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
  /*std::vector<T> H(N * N, T(0.0));
  for (auto i = 0; i < N; ++i) {
    H[i + N * i] = 0;
    if (i != N - 1) H[i + 1 + N * i] = std::sqrt(i * (N + 1 - i));
    if (i != N - 1) H[i + N * (i + 1)] = std::sqrt(i * (N + 1 - i));
  }*/

  std::vector<T> H(N * N, T(0.0));

  if(conf.input == "") {
    generateClement(H, N);

    // Load matrix into distributed buffer
    for (std::size_t x = 0; x < xlen; x++) {
      for (std::size_t y = 0; y < ylen; y++) {
        single.GetMatrixPtr()[x + xlen * y] = H.at((xoff + x) * N + (yoff + y));
      }
    }

    // print first off-diag elements of H
    if (rank == 0) {
      for (auto i = 1; i < std::size_t(6); ++i)
        std::cout << "|  " << std::setw(4) << i  << " | " << std::setw(20)
                  << H[i+N*i-1]
                  << "  |\n";
      std::cout << "\n\n\n";
    }
  }
  else {
    T* M = single.GetMatrixPtr();
//     readMatrix(M, conf.input, N * N);
    readMatrix(M, conf.input, N * N, xoff, yoff, xlen, ylen);
    // print first off-diag elements of H
    // if (rank == 0) {
    //   for (auto i = 1; i < std::size_t(6); ++i)
    //     std::cout << "|  " << std::setw(4) << i  << " | " << std::setw(20)
    //               << single.GetMatrixPtr()[i+N*i-1]
    //               << "  |\n";
    //   std::cout << "\n\n\n";
    // }
    // std::cout << " Not jet supported  \n";
  }

  for (auto idx = 0; idx < idx_max; ++idx) {
    if (rank == 0) {
      if( conf.perturb > 1e-20 )
        std::cout << "Starting Problem #" << idx << "\n";
      else
        std::cout << "Starting Problem " << conf.input << "\n";
      if (config.UseApprox()) {
        std::cout << "Using approximate solution\n";
      }
    }

    PerformanceDecoratorChase<T> performanceDecorator(&single);
    chase::Solve(&performanceDecorator);

    if (rank == 0) {
      std::cout << " ChASE timings: " << "\n";
//#ifdef USE_MPI
      performanceDecorator.GetPerfData().print();
//#endif
      Base<T>* resid = single.GetResid();
      if( conf.perturb > 1e-20 )
        std::cout << "Finished Problem #" << idx << "\n";
      else
        std::cout << "Finished Problem \n";
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
    
    
    
      if(idx == 0){
        std::ofstream outputFile(conf.output);
        
        outputFile
          << "| Index |       Eigenvalue      |         Residual      |\n"
          << "|-------|-----------------------|-----------------------|\n";
        std::size_t width = 20;
        outputFile << std::setprecision(12);
        outputFile << std::setfill(' ');
        outputFile << std::scientific;
        outputFile << std::right;
        for (auto i = 0; i < nev; ++i)
          outputFile << "|  " << std::setw(4) << i + 1 << " | " << std::setw(width)
                    << Lambda[i] << "  | " << std::setw(width) << resid[i]
                    << "  |\n";
        outputFile << "\n\n\n";
        
        outputFile.close();
        
        std::cout << "Eigenvalues written to " << conf.output << "\n";
        
      }
    }
    
    
    if( conf.perturb > 1e-20 ) {
      config.SetApprox(true);
      // Perturb Full Clement matrix
//       for (std::size_t i = 1; i < N; ++i) {
//         for (std::size_t j = 1; j < i; ++j) {
//           T element_perturbation = T(d(gen), d(gen)) * conf.perturb;
//           H[j + N * i] += element_perturbation;
//           H[i + N * j] += std::conj(element_perturbation);
//         }
//       }
      // Pertrub Eigenvector
      Base<T> Vnorm = 0;
      for (std::size_t i = 1; i < N * (nev + nex); ++i) {
        T element_perturbation = T(d(gen), d(gen)) * conf.perturb;
        V[ i ] += element_perturbation;
        Vnorm += real(V[i] * V[i]);
      }
      Vnorm = std::sqrt(Vnorm);
      for (std::size_t i = 1; i < N * (nev + nex); ++i) {
        single.GetVectorsPtr()[i] = V[ i ] / Vnorm;
      }
    }

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

  }
}
