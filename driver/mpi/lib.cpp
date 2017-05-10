/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set ts=8 sts=2 et sw=2 tw=80: */
/* TODO License */
#include "chase_mpi.hpp"
#include <complex.h>
#include <hdf5.h>
#include <mpi.h>
#include <random>

extern "C" {
void chase_write_hdf5(MPI_Comm comm, std::complex<float>* H, size_t N);
void chase_read_matrix(MPI_Comm comm, size_t* dims_ret,
    std::complex<float>** data_ptr);

void c_chase_(MPI_Fint* Fcomm, std::complex<float>* H, int* N,
    std::complex<float>* V, float* ritzv, int* nev, int* nex,
    int* deg, double* tol, char* mode, char* opt)
{

    MPI_Comm comm = MPI_Comm_f2c(*Fcomm);
    std::cout << "entering chase H[0] = " << H[0] << std::endl;
    std::cout << "tol: " << *tol << std::endl;

    std::mt19937 gen(2342.0); // TODO
    std::normal_distribution<> d;

    int mpi_size = 0;
    MPI_Comm_size(comm, &mpi_size);

    ChASE_MPI<std::complex<float> >* single;
    std::complex<float>* HH;

    if (mpi_size > 1) {
        chase_write_hdf5(comm, H, *N);
        MPI_Barrier(comm);
        size_t dims_ret[2];
        chase_read_matrix(comm, dims_ret, &HH);
        assert(dims_ret[0] == dims_ret[1]);

        ChASE_Config config(dims_ret[0], *nev, *nex);
        config.setTol(*tol);
        config.setDeg(*deg);
        config.setOpt(opt == "S" || opt == "s");

        single = new ChASE_MPI<std::complex<float> >(config, MPI_COMM_SELF, HH, V,
            ritzv);
    } else {
        ChASE_Config config(*N, *nev, *nex);
        config.setTol(*tol);
        config.setDeg(*deg);
        config.setOpt(opt == "S" || opt == "s");
        HH = H;

        single = new ChASE_MPI<std::complex<float> >(config, MPI_COMM_SELF, HH, V,
            ritzv);
    }

    for (std::size_t k = 0; k < *N * (*nev + *nex); ++k)
        V[k] = std::complex<float>(d(gen), d(gen));

    float normH = std::max<float>(t_lange('1', *N, *N, HH, *N), float(1.0));
    single->setNorm(normH);

    single->solve();
    delete single;
}
}
