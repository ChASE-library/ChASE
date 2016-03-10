#include "../include/timing.h"


double start_times[7];
double timings[7] = {0};


void start_clock( TimePtrs t ) {
  start_times[t] = omp_get_wtime();
}

void end_clock( TimePtrs t ) {
  timings[t] += omp_get_wtime() - start_times[t];
}

double * get_timings() {
  return timings;
}

void print_timings() {
  std::cout
    << "All\t\t" << timings[ TimePtrs::All ] << std::endl
    << "Lanczos\t\t" << timings[ TimePtrs::Lanczos ] << std::endl
    << "Degrees\t\t"<< timings[ TimePtrs::Degrees ] << std::endl
    << "Filter\t\t"<< timings[ TimePtrs::Filter ] << std::endl
    << "QR\t\t"<< timings[ TimePtrs::Qr ] << std::endl
    << "RR\t\t"<< timings[ TimePtrs::Rr ] << std::endl
    << "Resid+Locking\t"<< timings[ TimePtrs::Resids_Locking ] << std::endl
    ;
}
