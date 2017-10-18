/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set ts=8 sts=2 et sw=2 tw=80: */
/* TODO License */
#pragma once

#ifdef HAS_MPI
#include <mpi.h>
#endif
#include <iostream>
#include <vector>
#include <chrono>

namespace chase {

class ChasePerfData {
 public:
  ChasePerfData()
      : chase_iteration_count(0),
        chase_filtered_vecs(0),
        timings(7),
        start_times(7) {}

  enum TimePtrs { All, Lanczos, Filter, Qr, Rr, Resids_Locking, Degrees };

  std::size_t get_iter_count() { return chase_iteration_count; }

  std::size_t get_filtered_vecs() { return chase_filtered_vecs; }

  void add_iter_count(std::size_t add) { chase_iteration_count += add; }

  void add_filtered_vecs(std::size_t add) { chase_filtered_vecs += add; }

  void start_clock(TimePtrs t) { start_times[t] = std::chrono::high_resolution_clock::now(); }

  void end_clock(TimePtrs t) { timings[t] += std::chrono::high_resolution_clock::now() - start_times[t]; }

  void print() {
// std::cout << "resd: " << norm << "\torth:" << norm2 << std::endl;
// std::cout << "Filtered Vectors\t\t" << chase_filtered_vecs << "\n";
// std::cout << "Iteration Count\t\t" << chase_iteration_count << "\n";

#ifdef HAS_MPI
    std::cout << " | Size ";
#endif

    std::cout << " | Iterations | Vecs  |  All | Lanczos | Degrees | Filter | "
                 "QR | RR | Resid | "
              << std::endl;

#ifdef HAS_MPI
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::cout << " | " << size;
#endif
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
              << timings[TimePtrs::Filter].count() << " | " << timings[TimePtrs::Qr].count()
              << " | " << timings[TimePtrs::Rr].count() << " | "
              << timings[TimePtrs::Resids_Locking].count() << " | " << std::endl;
  }

  std::size_t chase_iteration_count;
  std::size_t chase_filtered_vecs;

  std::vector<std::chrono::duration<double>> timings;
  std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> start_times;
};
}
