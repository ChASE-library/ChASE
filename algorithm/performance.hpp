/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials, 
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#ifndef CHASE_ALGORITHM_PERFORMANCE_HPP
#define CHASE_ALGORITHM_PERFORMANCE_HPP

#ifndef NO_MPI
    #include <mpi.h>
#endif

#include <chrono>
#include <iostream>
#include <vector>

#include "interface.hpp"
#include "types.hpp"

namespace chase {
  //! ChASE class for collecting data relative to FLOPs, timings, etc.
  /*! The ChasePerfData class collects and handles information relative to the execution of the eigensolver. It collects information about 
      - Number of subspace iterations
      - Number of filtered vectors 
      - Timings of each main algorithmic procedure (Lanczos, Filter, etc.)
      - Number of FLOPs executed
      
      The number of iterations and filtered vectors can be used to
      monitor the behavior of the algorithm as it attempts to converge
      all the desired eigenpairs. The timings and number of FLOPs are
      use to measure performance, especially parallel performance. The
      timings are stored in a vector of objects derived by the class
      template `std::chrono::duration`.
   */
class ChasePerfData {
 public:
  ChasePerfData()
      : chase_iteration_count(0),
        chase_filtered_vecs(0),
        timings(6),
        start_times(6),
        chase_iter_blocksizes(0) {}

  enum TimePtrs { All, Lanczos, Filter, Qr, Rr, Resids_Locking };

  void Reset() {
    chase_iteration_count = 0;
    chase_filtered_vecs = 0;
    chase_iter_blocksizes.clear();
    auto val = std::chrono::time_point<std::chrono::high_resolution_clock>();
    std::fill(start_times.begin(), start_times.end(), val);
    std::fill(timings.begin(), timings.end(), std::chrono::duration<double>());
  }

  //! Returns the number of total subspace iterations executed by ChASE.
  /*! The *S* in ChASE stands for *Subspace iteration*. The main
      engine under the hood of ChASE is a *loop* enveloping all the
      main routines executed by the code. Because of this structure,
      ChASE is a truly iterative algorithm based on subspace
      filtering. Counting the number of times such a loop is repeated
      gives a measure of the effectiveness of the algorithm and it is
      usually a non-linear function of the spectral distribution. For
      example, when using the flag ``approximate_ = 'true'`` to solve a
      sequence of eigenproblems, one can observe that the number of
      subspace iteration decreases as a function of sequences index.
      \return The total number of subspace iterations.
   */
  std::size_t get_iter_count() { return chase_iteration_count; }

  //! Returns the cumulative number of times each column vector is filtered by one degree.
  /*! The most computationally expensive routine of ChASE is the
      Chebyshev filter. Within the filter a matrix of vectors *V* is
      filtered with a varying degree each time a subspace iteration is
      executed. This counter return the total number of times each
      vector in *V* goes through a filtering step. For instance, when
      the flag `optim_ = false `, such a number roughly corresponds to
      rank(V) x degree x iter_count. When the `optim_` is set
      to `true` such a calculation is quite more complicated. Roughly
      speaking, this counter is useful to monitor the convergence
      ration of the filtered vectors and together with
      `get_iter_count` convey the effectiveness of the algorithm.
      \return Cumulative number of filtered vectors.
   */
  std::size_t get_filtered_vecs() { return chase_filtered_vecs; }

  std::vector<std::chrono::duration<double>> get_timings() { return timings; }

  //! Returns the total number of FLOPs executed by ChASE.
  /*! When measuring performance, it is fundamental to understand how
      many operations a routine executes against the total time to
      solutions. This counter returns the total amount of operations
      executed by ChASE and can be used to extract the performance of
      ChASE and compare it with theoretical peak performance of the
      platform where the code is executed.
      \param N Size of the eigenproblem matrix
      \return The total number of operations executed by ChASE.
   */ 
  std::size_t get_flops(std::size_t N) {
    std::size_t flop_count = 0;
    for (auto block : chase_iter_blocksizes) {
      // QR //

      // https://software.intel.com/en-us/mkl-developer-reference-fortran-2018-beta-geqrf
      // QR m=N n=block
      //(2/3)n^2(3m-n) *4
      flop_count +=
          (8. / 3.) * static_cast<double>(block * block * (3 * N - block));

      // https://software.intel.com/en-us/mkl-developer-reference-c-2018-beta-ungqr
      // m = N, n=k=block
      // (8/3)*n^2*(3m - n)
      flop_count +=
          (8. / 3.) * static_cast<double>(block * block * (3 * N - block) * 4);

      // RR //

      // W = H*V
      // https://software.intel.com/en-us/mkl-developer-reference-fortran-2018-beta-gemm
      // 8MNK + 18MN
      // m = N, k = N, n = block
      flop_count += 8 * N * block * N + 18 * N * block;

      // A = W' * W
      // 8MNK + 18MN
      // m = block, k = N, n = block
      flop_count += 8 * block * block * N + 18 * block * block;

      // reduction to tridiagonal
      // https://software.intel.com/en-us/mkl-developer-reference-fortran-2018-beta-hetrd
      // (16/3)n^3
      flop_count += 16. / 3. * static_cast<double>(block * block * block);

      // W = V*Z
      // 8MNK + 18MN
      flop_count += 8 * N * block * block + 18 * N * block;

      // resid //
      // W = H*V
      flop_count += 8 * N * block * N + 18 * N * block;

      // V[:,i] - lambda W[:,i]
      // 3*block
      flop_count += 3 * block * N;

      // ||V[:,i]||
      // 4*N
      flop_count += 4 * N * block;
    }

    // filter
    // 8MNK + 18MN
    flop_count +=
        8 * N * chase_filtered_vecs * N + 16 * N * chase_filtered_vecs;

    return flop_count / 1e9;
  }

  //! Returns the total number of FLOPs of the Chebyshev filter
  /*! Similar to `get_flops`, this counter return the total number of
      operations executed by the Chebyshev filter alone. Since the
      filter is the routine that executes, on average, 80% of the
      total FLOPs of ChASE, this counter is a good indicator of the
      performance of the entire algorithm. Because the filter executes
      almost exclusively BLAS-3 operations, this counter is quite
      useful to monitor how well the filter is close to the peak
      performance of the platform where ChASE is executed. This can be
      quite useful to fine tune the use of the computational resources
      used.
      \param N Size of the eigenproblem matrix
      \return The total number of operations executed by the polynomial filter.
   */
  std::size_t get_filter_flops(std::size_t N) {
    return (8 * N * chase_filtered_vecs * N + 16 * N * chase_filtered_vecs) /
           1e9;
  }

  void set_nprocs(int nProcs){
      nprocs = nProcs;
  }

  void add_iter_count(std::size_t add) { chase_iteration_count += add; }

  void add_iter_blocksize(std::size_t nevex) {
    chase_iter_blocksizes.push_back(nevex);
  }

  void add_filtered_vecs(std::size_t add) { chase_filtered_vecs += add; }

  void start_clock(TimePtrs t) {
    start_times[t] = std::chrono::high_resolution_clock::now();
  }

  void end_clock(TimePtrs t) {
    timings[t] += std::chrono::high_resolution_clock::now() - start_times[t];
  }

  //! Print function outputting counters and timings for all routines
  /*! It prints by default ( for N = 0) in the order,  
      - size of the eigenproblem
      - total number of subspace iterations executed
      - total number of filtered vectors
      - time-to-solution of the following 6 main sections of the ChASE algorithm:
         1. Total time-to-solution
         2. Estimates of the spectral bounds based on Lanczos, 
         3. Chebyshev filter, 
         4. QR decomposition, 
         5. Raleygh-Ritz procedure including the solution of the reduced dense problem, 
         6. Computation of the eigenpairs residuals

      When the parameter `N` is set to be a number else than zero, the
      function returns total FLOPs and filter FLOPs, respectively.
      \param N Control parameter. By default equal to *0*.
   */ 
  void print(std::size_t N = 0) {
    // std::cout << "resd: " << norm << "\torth:" << norm2 << std::endl;
    // std::cout << "Filtered Vectors\t\t" << chase_filtered_vecs << "\n";
    // std::cout << "Iteration Count\t\t" << chase_iteration_count << "\n";
    if (N != 0) {
      std::cout << " | GFLOPS | GFLOPS/s ";
    }

    std::cout << " | Size  | Iterations | Vecs   |  All       | Lanczos    | "
                 "Filter     | QR         | RR         | Resid      | "
              << std::endl;
    
    std::size_t width = 20;
    std::cout << std::setprecision(8);
    std::cout << std::setfill(' ');
//     std::cout << std::scientific;
    std::cout << std::right;

    if (N != 0) {
      std::size_t flops = get_flops(N);
      std::size_t filter_flops = get_filter_flops(N);
      std::cout << " | " << flops;
      std::cout << " | "
                << static_cast<double>(filter_flops) /
                       timings[TimePtrs::Filter].count() / 1e9;
    }

    std::cout << " | " << std::setw(5) << nprocs;
    std::cout << " | " << std::setw(10) << chase_iteration_count << " | " << std::setw(6) << chase_filtered_vecs;

    this->print_timings();
  }

 private:
  void print_timings() {
    // std::cout << "All\t\t" << timings[TimePtrs::All] << std::endl
    //           << "Lanczos\t\t" << timings[TimePtrs::Lanczos] << std::endl
    //           << "Filter\t\t" << timings[TimePtrs::Filter] << std::endl
    //           << "QR\t\t" << timings[TimePtrs::Qr] << std::endl
    //           << "RR\t\t" << timings[TimePtrs::Rr] << std::endl
    //           << "Resid+Locking\t" << timings[TimePtrs::Resids_Locking]
    //           << std::endl;
    
    std::size_t width = 10;
    std::cout << std::setprecision(6);
    std::cout << std::setfill(' ');
//     std::cout << std::scientific;
    std::cout << std::right;
    
    std::cout << " | " << std::setw(width) << timings[TimePtrs::All].count()
              << " | " << std::setw(width) << timings[TimePtrs::Lanczos].count()
              << " | " << std::setw(width) << timings[TimePtrs::Filter].count()
              << " | " << std::setw(width) << timings[TimePtrs::Qr].count()
              << " | " << std::setw(width) << timings[TimePtrs::Rr].count()
              << " | " << std::setw(width) << timings[TimePtrs::Resids_Locking].count()
              << " | " << std::endl;
  }

  std::size_t chase_iteration_count;
  std::size_t chase_filtered_vecs;
  std::vector<std::size_t> chase_iter_blocksizes;

  std::vector<std::chrono::duration<double>> timings;
  std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>>
      start_times;
  int nprocs;
};
  //! A derived class used to extract performance and configuration data.
  /*! This is a class derived from the Chase class which plays the
      role of interface for the kernels used by the library. All
      members of the Chase class are virtual functions. These
      functions are re-implemented in the PerformanceDecoratorChase
      class. All derived members that provide an interface to
      computational kernels are reimplemented by *decorating* the
      original function with time pointers which are members of the
      ChasePerfData class. All derived members that provide an
      interface to input or output data are called without any
      specific decoration. In addition to the virtual member of the
      Chase class, the PerformanceDecoratorChase class has also among
      its public members a reference to an object of type
      ChasePerfData. When using Chase to solve an eigenvalue problem,
      the members of the PerformanceDecoratorChase are called instead
      of the virtual functions members of the Chase class. In this
      way, all parameters and counters are automatically invoked and
      returned in the correct order.  
      \see Chase
   */ 
template <class T>
class PerformanceDecoratorChase : public chase::Chase<T> {
 public:
  PerformanceDecoratorChase(Chase<T> *chase) : chase_(chase), perf_() {}

  void initVecs(){
    chase_->initVecs();
  }

  void initRndVecs(T *V){
    chase_->initRndVecs(V);
  }

  void Shift(T c, bool isunshift = false) {
    if (isunshift)
      perf_.end_clock(ChasePerfData::TimePtrs::Filter);
    else
      perf_.start_clock(ChasePerfData::TimePtrs::Filter);

    chase_->Shift(c, isunshift);
  }
  void HEMM(std::size_t nev, T alpha, T beta, std::size_t offset) {
    chase_->HEMM(nev, alpha, beta, offset);
    perf_.add_filtered_vecs(nev);
  }
  void QR(std::size_t fixednev) {
    perf_.start_clock(ChasePerfData::TimePtrs::Qr);
    chase_->QR(fixednev);
    perf_.end_clock(ChasePerfData::TimePtrs::Qr);
  }

  void stabQR(std::size_t fixednev) {
    perf_.start_clock(ChasePerfData::TimePtrs::Qr);
    chase_->stabQR(fixednev);
    perf_.end_clock(ChasePerfData::TimePtrs::Qr);
  }

  void fastQR(std::size_t fixednev) {
    perf_.start_clock(ChasePerfData::TimePtrs::Qr);
    chase_->fastQR(fixednev);
    perf_.end_clock(ChasePerfData::TimePtrs::Qr);
  }

  void RR(Base<T> *ritzv, std::size_t block) {
    perf_.start_clock(ChasePerfData::TimePtrs::Rr);
    chase_->RR(ritzv, block);
    perf_.add_iter_blocksize(block);
    perf_.end_clock(ChasePerfData::TimePtrs::Rr);
  }
  void Resd(Base<T> *ritzv, Base<T> *resd, std::size_t fixednev) {
    perf_.start_clock(ChasePerfData::TimePtrs::Resids_Locking);
    chase_->Resd(ritzv, resd, fixednev);
    // We end with ->chase_->Lock()
  }
  void Lanczos(std::size_t m, Base<T> *upperb) {
    perf_.start_clock(ChasePerfData::TimePtrs::Lanczos);
    chase_->Lanczos(m, upperb);
    perf_.end_clock(ChasePerfData::TimePtrs::Lanczos);
  }
  void Lanczos(std::size_t M, std::size_t idx, Base<T> *upperb, Base<T> *ritzv,
               Base<T> *Tau, Base<T> *ritzV) {
    chase_->Lanczos(M, idx, upperb, ritzv, Tau, ritzV);
  }
  void LanczosDos(std::size_t idx, std::size_t m, T *ritzVc) {
    chase_->LanczosDos(idx, m, ritzVc);
    perf_.end_clock(ChasePerfData::TimePtrs::Lanczos);
  }

  void Swap(std::size_t i, std::size_t j) { chase_->Swap(i, j); }
  void Lock(std::size_t new_converged) {
    chase_->Lock(new_converged);
    perf_.end_clock(ChasePerfData::TimePtrs::Resids_Locking);
    perf_.add_iter_count(1);
  }
  void Start() {
    chase_->Start();
    perf_.Reset();
    perf_.start_clock(ChasePerfData::TimePtrs::All);
    perf_.start_clock(ChasePerfData::TimePtrs::Lanczos);
    perf_.set_nprocs(chase_->get_nprocs());
  }

  int get_nprocs() {
      return chase_->get_nprocs();
  }

  void End() {
    chase_->End();
    perf_.end_clock(ChasePerfData::TimePtrs::All);
  }

  std::size_t GetN() const { return chase_->GetN(); }
  std::size_t GetNev() { return chase_->GetNev(); }
  std::size_t GetNex() { return chase_->GetNex(); }
  Base<T> *GetRitzv() { return chase_->GetRitzv(); }
  Base<T> *GetResid() { return chase_->GetResid(); }
  ChaseConfig<T> &GetConfig() { return chase_->GetConfig(); }
  ChasePerfData &GetPerfData() { return perf_; }

#ifdef CHASE_OUTPUT
  void Output(std::string str) { chase_->Output(str); }
#endif

 private:
  Chase<T> *chase_;
  ChasePerfData perf_;
};

}  // namespace chase
#endif
