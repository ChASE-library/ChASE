/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */

#include "chase_blas.hpp"

extern "C" {
void
c_chase_(std::complex<double>* H, int* N, std::complex<double>* V,
         std::complex<double>* W, double* ritzv, int* nev, int* nex, int* deg,
         double* tol, char* mode, char* opt)
{
  ChASE_Config config(*N, *nev, *nex);

  config.setTol(*tol);
  config.setDeg(*deg);
  config.setOpt(opt == "S" || opt == "s");

  ChASE_Blas<std::complex<double>>* single =
    new ChASE_Blas<std::complex<double>>(config, H, V, ritzv);
}
}
