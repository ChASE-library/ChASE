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
void chase_read_matrix(MPI_Comm comm, std::size_t xoff, std::size_t yoff,
    std::size_t xlen, std::size_t ylen, std::complex<float>* H);

void c_chase_(MPI_Fint* Fcomm, std::complex<float>* H, int* N,
    std::complex<float>* V, float* ritzv, int* nev, int* nex,
    int* deg, double* tol, char* mode, char* opt)
{

    MPI_Comm comm = MPI_Comm_f2c(*Fcomm);
    ChASE_MPI<std::complex<float> >* single;

    double start_time = omp_get_wtime();
    double stop_time;

    CHASE_INT xoff;
    CHASE_INT yoff;
    CHASE_INT xlen;
    CHASE_INT ylen;

    std::mt19937 gen(2342.0); // TODO
    std::normal_distribution<> d;

    int rank;

    chase_write_hdf5(comm, H, *N);
    MPI_Barrier(comm);

    ChASE_Config config(*N, *nev, *nex);
    config.setTol(*tol);
    config.setDeg(*deg);
    config.setOpt(*opt == 'S' || *opt == 's');
    config.setApprox(*mode == 'A' || *mode == 'a');
    config.setLanczosIter(25);

    single = new ChASE_MPI<std::complex<float> >(config, comm, V, ritzv);
    single->get_off(&xoff, &yoff, &xlen, &ylen);

    std::complex<float>* HH = single->getMatrixPtr();
    chase_read_matrix(comm, xoff, yoff, xlen, ylen, HH);

    //float normH = std::max<float>(t_lange('1', *N, *N, HH, *N), float(1.0));
    single->setNorm(9);

    if (!config.use_approx())
        for (std::size_t k = 0; k < *N * (*nev + *nex); ++k)
            V[k] = std::complex<float>(d(gen), d(gen));

    MPI_Barrier(comm);
    stop_time = omp_get_wtime();
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0)
        std::cout << "time for reading writing matrix and init: "
                  << stop_time - start_time << "\n";

    single->solve();

    ChASE_PerfData perf = single->getPerfData();

    if (rank == 0)
        perf.print();

    delete single;
}
}
