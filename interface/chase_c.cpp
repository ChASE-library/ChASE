/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2018, Simulation Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany
// and
// Copyright (c) 2016-2018, Aachen Institute for Advanced Study in Computational
//   Engineering Science, RWTH Aachen University, Germany All rights reserved.
// License is 3-clause BSD:
// https://github.com/SimLabQuantumMaterials/ChASE/

#include <complex.h>
#include <mpi.h>
#include <random>
#include <chrono>

#include "algorithm/performance.hpp"

#include "ChASE-MPI/chase_mpi.hpp"
#include "ChASE-MPI/chase_mpi_properties.hpp"
#include "ChASE-MPI/impl/chase_mpidla_blaslapack.hpp"
#include "ChASE-MPI/impl/chase_mpidla_blaslapack_seq.hpp"
#include "ChASE-MPI/impl/chase_mpidla_blaslapack_seq_inplace.hpp"

#ifdef HAS_GPU
  #include "ChASE-MPI/impl/chase_mpidla_mgpu.hpp"
#endif

using namespace chase;
using namespace chase::mpi;

class ChASE_State {
 public:
  /* N: dimension of matrix
   * mbsize: block size in first dimension
   * nbsize: block size in second dimension
   * nev: number of eigenpairs to be computed
   * nex: dimension of extra space to compute eigenpairs
   * dim0: dimension of row communicator
   * dim1: dimension of column communicator 
   * grid_major: grid major of MPI grid: R = row major, C = column major
   * irsrc: The process row over which the first row of matrix is distributed.
   * icsrc: The process column over which the first column of matrix is distributed.
   * comm: working MPI communicator
  */ 
  template <typename T>
  static ChaseMpiProperties<T>* constructProperties(std::size_t N,
		  				    std::size_t mbsize,
						    std::size_t nbsize,
                                                    std::size_t nev,
                                                    std::size_t nex,
						    int dim0,
						    int dim1,
						    char *grid_major,
						    int irsrc,
						    int icsrc,
						    MPI_Comm comm);
  /* N: dimension of matrix
   * nev: number of eigenpairs to be computed
   * nex: dimension of extra space to compute eigenpairs
   * m: row number of local matrix on each MPI rank
   * n: column number of local matrix on each MPI rank
   * dim0: dimension of row communicator
   * dim1: dimension of column communicator
   * comm: working MPI communicator
  */
  template <typename T>
  static ChaseMpiProperties<T>* constructProperties(std::size_t N,
                                                    std::size_t nev,
                                                    std::size_t nex,
						    std::size_t m,
						    std::size_t n,
                                                    int dim0,
                                                    int dim1,
						    char *grid_major,
						    MPI_Comm comm);
  /* N: dimension of matrix
   * nev: number of eigenpairs to be computed
   * nex: dimension of extra space to compute eigenpairs
   * comm: working MPI communicator
  */
  template <typename T>
  static ChaseMpiProperties<T>* constructProperties(std::size_t N,
                                                    std::size_t nev,
                                                    std::size_t nex,
                                                    MPI_Comm comm);


  template <typename T>
  static ChaseMpiProperties<T>* getProperties();

  static ChaseMpiProperties<double>* double_prec;
  static ChaseMpiProperties<std::complex<double>>* complex_double_prec;
  static ChaseMpiProperties<float>* single_prec;
  static ChaseMpiProperties<std::complex<float>>* complex_single_prec;  
};

ChaseMpiProperties<double>* ChASE_State::double_prec = nullptr;
ChaseMpiProperties<std::complex<double>>* ChASE_State::complex_double_prec = nullptr;
ChaseMpiProperties<float>* ChASE_State::single_prec = nullptr;
ChaseMpiProperties<std::complex<float>>* ChASE_State::complex_single_prec = nullptr;

template <>
ChaseMpiProperties<double>* ChASE_State::constructProperties(std::size_t N,
 	                                                     std::size_t mbsize,
                                                       	     std::size_t nbsize,
                                                             std::size_t nev,
                                                             std::size_t nex,
                                                       	     int dim0,
                                                             int dim1,
                                                             char *grid_major,
                                                             int irsrc,
                                                             int icsrc,    
                                                       	     MPI_Comm comm) {
  double_prec = new ChaseMpiProperties<double>(N, mbsize, nbsize, nev, nex, dim0, 
		  				dim1, grid_major, irsrc, icsrc,comm);
  return double_prec;
}

template <>
ChaseMpiProperties<std::complex<double>>* ChASE_State::constructProperties(std::size_t N,
                                                             std::size_t mbsize,
                                                             std::size_t nbsize,
                                                             std::size_t nev,
                                                             std::size_t nex,
                                                             int dim0,
                                                             int dim1,
                                                             char *grid_major,
                                                             int irsrc,
                                                             int icsrc,
                                                             MPI_Comm comm) {
  complex_double_prec = new ChaseMpiProperties<std::complex<double>>(N, mbsize, nbsize, nev, nex, dim0,
                                                dim1, grid_major, irsrc, icsrc,comm);
  return complex_double_prec;
}


template <>
ChaseMpiProperties<float>* ChASE_State::constructProperties(std::size_t N,
                                                             std::size_t mbsize,
                                                             std::size_t nbsize,
                                                             std::size_t nev,
                                                             std::size_t nex,
                                                             int dim0,
                                                             int dim1,
                                                             char *grid_major,
                                                             int irsrc,
                                                             int icsrc,
                                                             MPI_Comm comm) {
  single_prec = new ChaseMpiProperties<float>(N, mbsize, nbsize, nev, nex, dim0,
                                                dim1, grid_major, irsrc, icsrc,comm);
  return single_prec;
}

template <>
ChaseMpiProperties<std::complex<float>>* ChASE_State::constructProperties(std::size_t N,
                                                             std::size_t mbsize,
                                                             std::size_t nbsize,
                                                             std::size_t nev,
                                                             std::size_t nex,
                                                             int dim0,
                                                             int dim1,
                                                             char *grid_major,
                                                             int irsrc,
                                                             int icsrc,
                                                             MPI_Comm comm) {
  complex_single_prec = new ChaseMpiProperties<std::complex<float>>(N, mbsize, nbsize, nev, nex, dim0,
                                                dim1, grid_major, irsrc, icsrc,comm);
  return complex_single_prec;
}

template <>
ChaseMpiProperties<double>* ChASE_State::constructProperties(std::size_t N,
                                                    std::size_t nev,
                                                    std::size_t nex,
                                                    std::size_t m,
                                                    std::size_t n,
                                                    int dim0,
                                                    int dim1,
						    char *grid_major,
                                                    MPI_Comm comm){

  double_prec = new ChaseMpiProperties<double>(N, nev, nex, m, n, dim0, dim1, grid_major, comm);
  return double_prec;      	
}

template <>
ChaseMpiProperties<float>* ChASE_State::constructProperties(std::size_t N,
                                                    std::size_t nev,
                                                    std::size_t nex,
                                                    std::size_t m,
                                                    std::size_t n,
                                                    int dim0,
                                                    int dim1,
						    char *grid_major,
                                                    MPI_Comm comm){

  single_prec = new ChaseMpiProperties<float>(N, nev, nex, m, n, dim0, dim1, grid_major, comm);
  return single_prec;
}

template <>
ChaseMpiProperties<std::complex<double>>* ChASE_State::constructProperties(std::size_t N,
                                                    std::size_t nev,
                                                    std::size_t nex,
                                                    std::size_t m,
                                                    std::size_t n,
                                                    int dim0,
                                                    int dim1,
						    char *grid_major,
                                                    MPI_Comm comm){

  complex_double_prec = new ChaseMpiProperties<std::complex<double>>(N, nev, nex, m, n, dim0, dim1, grid_major, comm);
  return complex_double_prec;
}

template <>
ChaseMpiProperties<std::complex<float>>* ChASE_State::constructProperties(std::size_t N,
                                                    std::size_t nev,
                                                    std::size_t nex,
                                                    std::size_t m,
                                                    std::size_t n,
                                                    int dim0,
                                                    int dim1,
						    char *grid_major,
                                                    MPI_Comm comm){

  complex_single_prec = new ChaseMpiProperties<std::complex<float>>(N, nev, nex, m, n, dim0, dim1, grid_major, comm);
  return complex_single_prec;
}

template <>
ChaseMpiProperties<double>* ChASE_State::constructProperties(std::size_t N,
                                                    std::size_t nev,
                                                    std::size_t nex,
                                                    MPI_Comm comm){

  double_prec = new ChaseMpiProperties<double>(N, nev, nex, comm);
  return double_prec;
}

template <>
ChaseMpiProperties<float>* ChASE_State::constructProperties(std::size_t N,
                                                    std::size_t nev,
                                                    std::size_t nex,
                                                    MPI_Comm comm){

  single_prec = new ChaseMpiProperties<float>(N, nev, nex, comm);
  return single_prec;
}

template <>
ChaseMpiProperties<std::complex<double>>* ChASE_State::constructProperties(std::size_t N,
                                                    std::size_t nev,
                                                    std::size_t nex,
                                                    MPI_Comm comm){

  complex_double_prec = new ChaseMpiProperties<std::complex<double>>(N, nev, nex, comm);
  return complex_double_prec;
}

template <>
ChaseMpiProperties<std::complex<float>>* ChASE_State::constructProperties(std::size_t N,
                                                    std::size_t nev,
                                                    std::size_t nex,
                                                    MPI_Comm comm){

  complex_single_prec = new ChaseMpiProperties<std::complex<float>>(N, nev, nex, comm);
  return complex_single_prec;
}


template <>
ChaseMpiProperties<double>* ChASE_State::getProperties() {
  return double_prec;
}
template <>
ChaseMpiProperties<std::complex<double>>* ChASE_State::getProperties() {
  return complex_double_prec;
}

template <>
ChaseMpiProperties<float>* ChASE_State::getProperties() {
  return single_prec;
}
template <>
ChaseMpiProperties<std::complex<float>>* ChASE_State::getProperties() {
  return complex_single_prec;
}

template <typename T>
void chase_seq(T* H, int* N, T* V, Base<T>* ritzv, int* nev, int* nex,
                int* deg, double* tol, char* mode, char* opt) {
  typedef ChaseMpi<ChaseMpiDLABlaslapackSeq, T> SEQ_CHASE;

  std::vector<std::chrono::duration<double>> timings(3);
  std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> start_times(3);

  start_times[1] = std::chrono::high_resolution_clock::now();

  std::mt19937 gen(2342.0);
  std::normal_distribution<> d;

  SEQ_CHASE single(*N, *nev, *nex, V, ritzv, H);

  ChaseConfig<T>& config = single.GetConfig();
  config.SetTol(*tol);
  config.SetDeg(*deg);
  config.SetOpt(*opt == 'S');
  config.SetApprox(*mode == 'A');

  if (!config.UseApprox())
    for (std::size_t k = 0; k < *N * (*nev + *nex); ++k)
      V[k] = getRandomT<T>([&]() { return d(gen); });

  PerformanceDecoratorChase<T> performanceDecorator(&single);
  start_times[2] = std::chrono::high_resolution_clock::now();
  chase::Solve(&performanceDecorator);
  timings[2] = std::chrono::high_resolution_clock::now() - start_times[2];
  timings[1] = std::chrono::high_resolution_clock::now() - start_times[1];
  std::cout << "ChASE]> Seq-ChASE Solve done in: " << timings[2].count() << "\n";
  performanceDecorator.GetPerfData().print();  
  std::cout << "ChASE]> total time in ChASE: " << timings[1].count() << "\n";
}

template <typename T>
void chase_setup(MPI_Fint* fcomm, int* N, int *mbsize, int *nbsize, int* nev, int* nex, 
		 int *dim0, int *dim1, char *grid_major, int *irsrc, int *icsrc){
    MPI_Comm comm = MPI_Comm_f2c(*fcomm);
    auto props = ChASE_State::constructProperties<T>(*N, *mbsize, *nbsize, *nev, *nex, *dim0, 
		    				     *dim1, grid_major, *irsrc, *icsrc, comm);
}

template <typename T>
void chase_setup(MPI_Fint* fcomm, int* N, int *nev, int *nex, int* m, int* n,
                 int *dim0, int *dim1, char *grid_major){
    MPI_Comm comm = MPI_Comm_f2c(*fcomm);
    auto props = ChASE_State::constructProperties<T>(*N, *nev, *nex, *m, *n, *dim0,
                                                     *dim1, grid_major, comm);
}

template <typename T>
void chase_setup(MPI_Fint* fcomm, int* N, int *nev, int *nex ){
    MPI_Comm comm = MPI_Comm_f2c(*fcomm);
    auto props = ChASE_State::constructProperties<T>(*N, *nev, *nex, comm);
}

template <typename T>
void chase_solve(T* H, int *LDH, T* V, Base<T>* ritzv, int* deg, double* tol, char* mode,
                 char* opt) {
  typedef ChaseMpi<ChaseMpiDLABlaslapack, T> CHASE;

  std::vector<std::chrono::duration<double>> timings(3);
  std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> start_times(3);

  start_times[1] = std::chrono::high_resolution_clock::now();

  std::mt19937 gen(2342.0);
  std::normal_distribution<> d;
  ChaseMpiProperties<T>* props = ChASE_State::getProperties<T>();

  int myRank = props->get_my_rank();
  int ldh = *LDH;
  CHASE single(props, V, ritzv);

  T* H_ = single.GetMatrixPtr();
  std::size_t m, n;
  m = props->get_m();
  n = props->get_n();

  ChaseConfig<T>& config = single.GetConfig();
  auto N = config.GetN();
  auto nev = config.GetNev();
  auto nex = config.GetNex();

  if (!config.UseApprox())
    for (std::size_t k = 0; k < N * (nev + nex); ++k)
      V[k] = getRandomT<T>([&]() { return d(gen); });

  for(auto j = 0; j < n; j++ ){
      for(auto i = 0; i < m; i++){
          H_[m * j + i] = H[j * ldh + i];
      }
  }
  

  //std::cout << myRank << ": m = " << m << ", n = " << n << ", ldh = " << ldh << std::endl;  
  
  config.SetTol(*tol);
  config.SetDeg(*deg);
  config.SetOpt(*opt == 'S');
  config.SetApprox(*mode == 'A');

  PerformanceDecoratorChase<T> performanceDecorator(&single);
  start_times[2] = std::chrono::high_resolution_clock::now();
  chase::Solve(&performanceDecorator);

  timings[2] = std::chrono::high_resolution_clock::now() - start_times[2];
  timings[1] = std::chrono::high_resolution_clock::now() - start_times[1];
  if(myRank == 0){
      std::cout << "ChASE-MPI]> ChASE Solve done in: " << timings[2].count() << "\n";
      performanceDecorator.GetPerfData().print();
      std::cout << "ChASE-MPI]> total time in ChASE: " << timings[1].count() << "\n";      
  }
}

#ifdef HAS_GPU
template <typename T>
void chase_solve_mgpu(T* H, int *LDH, T* V, Base<T>* ritzv, int* deg, double* tol, char* mode,
                 char* opt) {
  
  typedef ChaseMpi<ChaseMpiDLAMultiGPU, T> CHASE;	
  
  int ldh = *LDH;
  std::vector<std::chrono::duration<double>> timings(3);
  std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> start_times(3);

  std::mt19937 gen(2342.0);
  std::normal_distribution<> d;
  ChaseMpiProperties<T>* props = ChASE_State::getProperties<T>();

  int myRank = props->get_my_rank();

  CHASE single(props, V, ritzv);

  T* H_ = single.GetMatrixPtr();
  std::size_t m, n;
  m = props->get_m();
  n = props->get_n();

  ChaseConfig<T>& config = single.GetConfig();
  auto N = config.GetN();
  auto nev = config.GetNev();
  auto nex = config.GetNex();

  if (!config.UseApprox())
    for (std::size_t k = 0; k < N * (nev + nex); ++k)
      V[k] = getRandomT<T>([&]() { return d(gen); });

  for(auto j = 0; j < n; j++ ){
      for(auto i = 0; i < m; i++){
          H_[m * j + i] = H[j * ldh + i];
      }
  }

  config.SetTol(*tol);
  config.SetDeg(*deg);
  config.SetOpt(*opt == 'S');
  config.SetApprox(*mode == 'A');

  PerformanceDecoratorChase<T> performanceDecorator(&single);
  start_times[2] = std::chrono::high_resolution_clock::now();
  chase::Solve(&performanceDecorator);

  timings[2] = std::chrono::high_resolution_clock::now() - start_times[2];
  timings[1] = std::chrono::high_resolution_clock::now() - start_times[1];
#ifdef INFO_PRINT
  if(myRank == 0){
      std::cout << "ChASE-MGPU]> ChASE Solve done in: " << timings[2].count() << "\n";
      performanceDecorator.GetPerfData().print();
      std::cout << "ChASE-MGPU]> total time in ChASE: " << timings[1].count() << "\n";
  }
#endif

}
#endif

extern "C" {

void zchase_(std::complex<double>* H, int* N, std::complex<double>* V,
             double* ritzv, int* nev, int* nex, int* deg, double* tol,
             char* mode, char* opt) {
  chase_seq<std::complex<double>>(H, N, V, ritzv, nev, nex, deg, tol, mode,
                                   opt);
}

void dchase_(double* H, int* N, double* V, double* ritzv, int* nev, int* nex,
             int* deg, double* tol, char* mode, char* opt) {
  chase_seq<double>(H, N, V, ritzv, nev, nex, deg, tol, mode, opt);
}

void cchase_(std::complex<float>* H, int* N, std::complex<float>* V,
             float* ritzv, int* nev, int* nex, int* deg, double* tol,
             char* mode, char* opt) {
  chase_seq<std::complex<float>>(H, N, V, ritzv, nev, nex, deg, tol, mode,
                                   opt);
}

void schase_(float* H, int* N, float* V, float* ritzv, int* nev, int* nex,
             int* deg, double* tol, char* mode, char* opt) {
  chase_seq<float>(H, N, V, ritzv, nev, nex, deg, tol, mode, opt);
}

void pzchase_init(MPI_Fint* fcomm, int* N, int *nev, int *nex){

  chase_setup<std::complex<double>>(fcomm, N, nev, nex);

}

void pdchase_init(MPI_Fint* fcomm, int* N, int *nev, int *nex){

  chase_setup<double>(fcomm, N, nev, nex);

}

void pcchase_init(MPI_Fint* fcomm, int* N, int *nev, int *nex){

  chase_setup<std::complex<float>>(fcomm, N, nev, nex);

}

void pschase_init(MPI_Fint* fcomm, int* N, int *nev, int *nex){

  chase_setup<float>(fcomm, N, nev, nex);

}

void pzchase_init_block(MPI_Fint* fcomm, int* N, int *nev, int *nex, int* m, int* n,
                int *dim0, int *dim1, char *grid_major){

  chase_setup<std::complex<double>>(fcomm, N, nev, nex, m, n, dim0, dim1, grid_major);

}

void pdchase_init_block(MPI_Fint* fcomm, int* N, int *nev, int *nex, int* m, int* n,
                int *dim0, int *dim1, char *grid_major){

  chase_setup<double>(fcomm, N, nev, nex, m, n, dim0, dim1, grid_major);

}

void pcchase_init_block(MPI_Fint* fcomm, int* N, int *nev, int *nex, int* m, int* n,
                int *dim0, int *dim1, char *grid_major){

  chase_setup<std::complex<float>>(fcomm, N, nev, nex, m, n, dim0, dim1, grid_major);

}

void pschase_init_block(MPI_Fint* fcomm, int* N, int *nev, int *nex, int* m, int* n,
                int *dim0, int *dim1, char *grid_major){

  chase_setup<float>(fcomm, N, nev, nex, m, n, dim0, dim1, grid_major);

}

void pzchase_init_blockcyclic(MPI_Fint* fcomm, int* N, int *mbsize, int *nbsize, int* nev, int* nex, 
		int *dim0, int *dim1, char *grid_major, int *irsrc, int *icsrc){

  chase_setup<std::complex<double>>(fcomm, N, mbsize, nbsize, nev, nex, dim0, dim1,
		  			grid_major, irsrc, icsrc);

}

void pdchase_init_blockcyclic(MPI_Fint* fcomm, int* N, int *mbsize, int *nbsize, int* nev, int* nex,
                int *dim0, int *dim1, char *grid_major, int *irsrc, int *icsrc){

  chase_setup<double>(fcomm, N, mbsize, nbsize, nev, nex, dim0, dim1,
                                        grid_major, irsrc, icsrc);

}

void pcchase_init_blockcyclic(MPI_Fint* fcomm, int* N, int *mbsize, int *nbsize, int* nev, int* nex,
                int *dim0, int *dim1, char *grid_major, int *irsrc, int *icsrc){

  chase_setup<std::complex<float>>(fcomm, N, mbsize, nbsize, nev, nex, dim0, dim1,
                                        grid_major, irsrc, icsrc);

}

void pschase_init_blockcyclic(MPI_Fint* fcomm, int* N, int *mbsize, int *nbsize, int* nev, int* nex,
                int *dim0, int *dim1, char *grid_major, int *irsrc, int *icsrc){

  chase_setup<float>(fcomm, N, mbsize, nbsize, nev, nex, dim0, dim1,
                                        grid_major, irsrc, icsrc);

}

void pzchase_(std::complex<double>* H, int *ldh, std::complex<double>* V,
                  double* ritzv, int* deg, double* tol, char* mode, char* opt) {
  chase_solve<std::complex<double>>(H, ldh, V, ritzv, deg, tol, mode, opt);
}

void pdchase_(double* H, int *ldh, double* V, double* ritzv, int* deg, double* tol,
                  char* mode, char* opt) {
  chase_solve<double>(H, ldh, V, ritzv, deg, tol, mode, opt);
}

void pcchase_(std::complex<float>* H, int *ldh, std::complex<float>* V,
                  float* ritzv, int* deg, double* tol, char* mode, char* opt) {
  chase_solve<std::complex<float>>(H, ldh, V, ritzv, deg, tol, mode, opt);
}

void pschase_(float* H, int *ldh, float* V, float* ritzv, int* deg, double* tol,
                  char* mode, char* opt) {
  chase_solve<float>(H, ldh, V, ritzv, deg, tol, mode, opt);
}

#ifdef HAS_GPU
void pzchase_mgpu_(std::complex<double>* H, int *ldh, std::complex<double>* V,
                  double* ritzv, int* deg, double* tol, char* mode, char* opt) {
  chase_solve_mgpu<std::complex<double>>(H, ldh, V, ritzv, deg, tol, mode, opt);
}

void pdchase_mgpu_(double* H, int *ldh, double* V, double* ritzv, int* deg, double* tol,
                  char* mode, char* opt) {
  chase_solve_mgpu<double>(H, ldh, V, ritzv, deg, tol, mode, opt);
}

void pcchase_mgpu_(std::complex<float>* H, int *ldh, std::complex<float>* V,
                  float* ritzv, int* deg, double* tol, char* mode, char* opt) {
  chase_solve_mgpu<std::complex<float>>(H, ldh, V, ritzv, deg, tol, mode, opt);
}

void pschase_mgpu_(float* H, int *ldh, float* V, float* ritzv, int* deg, double* tol,
                  char* mode, char* opt) {
  chase_solve_mgpu<float>(H, ldh, V, ritzv, deg, tol, mode, opt);
}
#endif

}  // extern C 
