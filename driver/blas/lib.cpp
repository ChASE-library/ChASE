/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
#include <random>

#include "chase_blas.hpp"

extern "C" {
void c_chase_(std::complex<float> *H, int *N, std::complex<float> *V,
              float *ritzv, int *nev, int *nex, int *deg, double *tol,
              char *mode, char *opt) {
  std::cout << "entering chase" << std::endl;
  std::cout << "tol: " << *tol << std::endl;
  ChASE_Config config(*N, *nev, *nex);

  config.setTol(*tol);
  config.setDeg(*deg);
  config.setOpt(opt == "S" || opt == "s");

  std::mt19937 gen(2342.0); // TODO
  std::normal_distribution<> d;

  for (std::size_t k = 0; k < *N * (*nev + *nex); ++k)
    V[k] = std::complex<float>(d(gen), d(gen));

  ChASE_Blas<std::complex<float>> *single =
      new ChASE_Blas<std::complex<float>>(config, H, V, ritzv);

  float normH = std::max<float>(t_lange('1', *N, *N, H, *N), float(1.0));
  single->setNorm(normH);


  single->solve();
  std::cout << ritzv[0] << "\n";

  delete single;
}
}
