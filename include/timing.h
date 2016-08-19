#ifndef CHASE_TIMING_H
#define CHASE_TIMING_H

#include <omp.h>
#include <iostream>

enum TimePtrs {
  All,
  Lanczos,
  Filter,
  Qr,
  Rr,
  Resids_Locking,
  Degrees
};

void start_clock( TimePtrs t );

void end_clock( TimePtrs t );

double * get_timings();

void print_timings();

void reset_clock();

#endif // CHASE_TIMING_H
