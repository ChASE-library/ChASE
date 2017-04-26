/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
#pragma once

#include <omp.h>
#include <vector>

class ChASE_PerfData
{
public:
  ChASE_PerfData()
    : timings(7, 0)
    , start_times(7)
    , chase_iteration_count(0)
    , chase_filtered_vecs(0)
  {
  }

  enum TimePtrs
  {
    All,
    Lanczos,
    Filter,
    Qr,
    Rr,
    Resids_Locking,
    Degrees
  };

  std::size_t get_iter_count() { return chase_iteration_count; }

  std::size_t get_filtered_vecs() { return chase_filtered_vecs; }

  void add_iter_count(std::size_t add) { chase_iteration_count += add; }

  void add_filtered_vecs(std::size_t add) { chase_filtered_vecs += add; }

  void start_clock(TimePtrs t) { start_times[t] = omp_get_wtime(); }

  void end_clock(TimePtrs t) { timings[t] += omp_get_wtime() - start_times[t]; }

  void print_timings()
  {
    std::cout << "All\t\t" << timings[TimePtrs::All] << std::endl
              << "Lanczos\t\t" << timings[TimePtrs::Lanczos] << std::endl
              << "Degrees\t\t" << timings[TimePtrs::Degrees] << std::endl
              << "Filter\t\t" << timings[TimePtrs::Filter] << std::endl
              << "QR\t\t" << timings[TimePtrs::Qr] << std::endl
              << "RR\t\t" << timings[TimePtrs::Rr] << std::endl
              << "Resid+Locking\t" << timings[TimePtrs::Resids_Locking]
              << std::endl;
  }

  void print()
  {
    // std::cout << "resd: " << norm << "\torth:" << norm2 << std::endl;
    this->print_timings();
    std::cout << "Filtered Vectors\t\t" << chase_filtered_vecs << "\n";
    std::cout << "Iteration Count\t\t" << chase_iteration_count << "\n";
  }

private:
  std::vector<double> start_times;
  std::vector<double> timings;
  std::size_t chase_iteration_count;
  std::size_t chase_filtered_vecs;
};
