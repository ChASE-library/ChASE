/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */

#include <complex.h>
#include <hdf5.h>
#include <mpi.h>
#include <random>

#include "genera/matrixfree/blas_templates.h"
#include "genera/matrixfree/factory.h"


using namespace chase;

using T = std::complex<float>;

extern "C" {
// void chase_write_hdf5(MPI_Comm comm, std::complex<float>* H, size_t N);
// void chase_read_matrix(MPI_Comm comm, size_t* dims_ret,
//                        std::complex<float>** data_ptr);

void chase_write_hdf5(MPI_Comm comm, std::complex<float>* H, size_t N);
void chase_read_matrix(MPI_Comm comm, std::size_t xoff, std::size_t yoff,
                       std::size_t xlen, std::size_t ylen,
                       std::complex<float>* H);

/*
void c_chase_(std::complex<float>* H, int* N, std::complex<float>* V,
    float* ritzv, int* nev, int* nex, int* deg, double* tol,
    char* mode, char* opt)
{
    std::cout << "entering chase" << std::endl;
    std::cout << "tol: " << *tol << std::endl;
    ChASE_Config config(*N, *nev, *nex);

    config.setTol(*tol);
    config.setDeg(*deg);
    config.setOpt(opt == "S" || opt == "s");
    config.setApprox(mode == "A" || mode == "a");

    std::mt19937 gen(2342.0); // TODO
    std::normal_distribution<> d;

    if (!config.use_approx())
        for (std::size_t k = 0; k < *N * (*nev + *nex); ++k)
            V[k] = std::complex<float>(d(gen), d(gen));

    ChASE_Blas<std::complex<float> >* single = new
ChASE_Blas<std::complex<float> >(config, H, V, ritzv);

    float normH = std::max<float>(t_lange('1', *N, *N, H, *N), float(1.0));
    single->setNorm(normH);

    single->solve();
    std::cout << ritzv[0] << "\n";

    delete single;
}
*/
void c_chase_(MPI_Fint* Fcomm, std::complex<float>* H, int* N,
              std::complex<float>* V, float* ritzv, int* nev, int* nex,
              int* deg, double* tol, char* mode, char* opt) {
  MPI_Comm comm = MPI_Comm_f2c(*Fcomm);
  int mpi_size = 0;
  MPI_Comm_size(comm, &mpi_size);

  std::mt19937 gen(2342.0);  // TODO
  std::normal_distribution<> d;
  size_t dims_ret[2];

  if (mpi_size > 1) {
    chase_write_hdf5(comm, H, *N);
  }
  MPI_Barrier(comm);

  // if (mpi_size > 1) {
  //   chase_write_hdf5(comm, H, *N);
  //   chase_read_matrix(comm, dims_ret, &HH);
  //   assert(dims_ret[0] == dims_ret[1]);
  // } else {
  //   dims_ret[0] = *N;
  //   dims_ret[1] = *N;
  //   HH = H;
  // }

  ChaseConfig<std::complex<float>> config(*N, *nev, *nex);
  config.setTol(*tol);
  config.setDeg(*deg);
  config.setOpt(*opt == 'S' || *opt == 's');
  config.setApprox(*mode == 'A' || *mode == 'a');
  config.setLanczosIter(25);

  std::unique_ptr<MatrixFreeChase<T>> single =
    constructChASE(config, static_cast<T*>(nullptr), V, ritzv, MPI_COMM_WORLD);

  CHASE_INT xoff;
  CHASE_INT yoff;
  CHASE_INT xlen;
  CHASE_INT ylen;

  single->GetOff(&xoff, &yoff, &xlen, &ylen);

  std::complex<float>* HH = single->GetMatrixPtr();
  if (mpi_size == 1) {
    for (auto i = 0; i < *N * *N; ++i) HH[i] = H[i];

  } else {
    chase_read_matrix(comm, xoff, yoff, xlen, ylen, HH);
  }

  // single = new ChASE_Blas<std::complex<float> >(config, HH, V, ritzv);

  if (!config.use_approx())
    for (std::size_t k = 0; k < *N * (*nev + *nex); ++k)
      V[k] = std::complex<float>(d(gen), d(gen));

  //    float normH = std::max<float>(t_lange('1', *N, *N, HH, *N), float(1.0));
  single->SetNorm(6);

  single->Solve();
}
}
