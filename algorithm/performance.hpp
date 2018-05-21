/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2018, Simulation Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany
// and
// Copyright (c) 2016-2018, Aachen Institute for Advanced Study in Computational
//   Engineering Science, RWTH Aachen University, Germany All rights reserved.
// License is 3-clause BSD:
// https://github.com/SimLabQuantumMaterials/ChASE/

#ifndef CHASE_ALGORITHM_PERFORMANCE_HPP
#define CHASE_ALGORITHM_PERFORMANCE_HPP

#include <mpi.h>
#include <chrono>
#include <iostream>
#include <vector>

#include "interface.hpp"
#include "types.hpp"

namespace chase {

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

  std::size_t get_iter_count() { return chase_iteration_count; }

  std::size_t get_filtered_vecs() { return chase_filtered_vecs; }

  std::vector<std::chrono::duration<double>> get_timings() { return timings; }

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
        8 * N * chase_filtered_vecs * N + 18 * N * chase_filtered_vecs;

    return flop_count / 1e9;
  }

  std::size_t get_filter_flops(std::size_t N) {
    return (8 * N * chase_filtered_vecs * N + 18 * N * chase_filtered_vecs) /
           1e9;
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

  void print(std::size_t N = 0) {
    // std::cout << "resd: " << norm << "\torth:" << norm2 << std::endl;
    // std::cout << "Filtered Vectors\t\t" << chase_filtered_vecs << "\n";
    // std::cout << "Iteration Count\t\t" << chase_iteration_count << "\n";
    if (N != 0) {
      std::cout << " | GFLOPS | GFLOPS/s ";
    }

    std::cout << " | Size  | Iterations | Vecs  |  All | Lanczos | "
                 "Filter | QR | RR | Resid | "
              << std::endl;

    if (N != 0) {
      std::size_t flops = get_flops(N);
      std::size_t filter_flops =
          8 * N * chase_filtered_vecs * N + 18 * N * chase_filtered_vecs;
      std::cout << " | " << flops;
      std::cout << " | "
                << static_cast<double>(filter_flops) /
                       timings[TimePtrs::Filter].count() / 1e9;
    }

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::cout << " | " << size;
    std::cout << " | " << chase_iteration_count << " | " << chase_filtered_vecs;

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

    std::cout << " | " << timings[TimePtrs::All].count() << " | "
              << timings[TimePtrs::Lanczos].count() << " | "
              << timings[TimePtrs::Filter].count() << " | "
              << timings[TimePtrs::Qr].count() << " | "
              << timings[TimePtrs::Rr].count() << " | "
              << timings[TimePtrs::Resids_Locking].count() << " | "
              << std::endl;
  }

  std::size_t chase_iteration_count;
  std::size_t chase_filtered_vecs;
  std::vector<std::size_t> chase_iter_blocksizes;

  std::vector<std::chrono::duration<double>> timings;
  std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>>
      start_times;
};

template <class T>
class PerformanceDecoratorChase : public chase::Chase<T> {
 public:
  PerformanceDecoratorChase(Chase<T> *chase) : chase_(chase), perf_() {}

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
