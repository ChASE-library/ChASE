/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set ts=8 sts=2 et sw=2 tw=80: */
/* TODO License */
#pragma once

#ifdef HAS_MPI
#include <mpi.h>
#endif
#include <chrono>
#include <iostream>
#include <vector>

#include "types.h"

namespace chase {

class ChasePerfData {
 public:
  ChasePerfData()
      : chase_iteration_count(0),
        chase_filtered_vecs(0),
        timings(7),
        start_times(7),
        chase_iter_blocksizes(0) {}

  enum TimePtrs { All, Lanczos, Filter, Qr, Rr, Resids_Locking, Degrees };

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

#ifdef HAS_MPI
    std::cout << " | Size ";
#endif

    std::cout << " | Iterations | Vecs  |  All | Lanczos | Degrees | Filter | "
                 "QR | RR | Resid | "
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

    int size = 1;
#ifdef HAS_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
    std::cout << " | " << size;
    std::cout << " | " << chase_iteration_count << " | " << chase_filtered_vecs;

    this->print_timings();
  }

 private:
  void print_timings() {
    // std::cout << "All\t\t" << timings[TimePtrs::All] << std::endl
    //           << "Lanczos\t\t" << timings[TimePtrs::Lanczos] << std::endl
    //           << "Degrees\t\t" << timings[TimePtrs::Degrees] << std::endl
    //           << "Filter\t\t" << timings[TimePtrs::Filter] << std::endl
    //           << "QR\t\t" << timings[TimePtrs::Qr] << std::endl
    //           << "RR\t\t" << timings[TimePtrs::Rr] << std::endl
    //           << "Resid+Locking\t" << timings[TimePtrs::Resids_Locking]
    //           << std::endl;

    std::cout << " | " << timings[TimePtrs::All].count() << " | "
              << timings[TimePtrs::Lanczos].count() << " | "
              << timings[TimePtrs::Degrees].count() << " | "
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
}
