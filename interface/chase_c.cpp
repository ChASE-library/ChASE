/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2018, Simulation Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany
// and
// Copyright (c) 2016-2018, Aachen Institute for Advanced Study in Computational
//   Engineering Science, RWTH Aachen University, Germany All rights reserved.
// License is 3-clause BSD:
// https://github.com/SimLabQuantumMaterials/ChASE/

#include "ChASE-MPI/blas_templates.hpp"
#include "ChASE-MPI/chase_mpi.hpp"
#include "ChASE-MPI/chase_mpi_properties.hpp"
#include "ChASE-MPI/impl/chase_mpidla_blaslapack.hpp"
#include "ChASE-MPI/impl/chase_mpidla_blaslapack_seq.hpp"
#include "ChASE-MPI/impl/chase_mpidla_blaslapack_seq_inplace.hpp"
#include "algorithm/performance.hpp"
#include <algorithm>
#include <chrono>
#include <complex.h>
#include <complex>
#include <fstream>
#include <mpi.h>
#include <random>
#include <sys/stat.h>

#ifdef HAS_GPU 
#include "ChASE-MPI/impl/chase_mpidla_cuda_seq.hpp"
#include "ChASE-MPI/impl/chase_mpidla_mgpu.hpp"
#endif

using namespace chase;
using namespace chase::mpi;

#ifdef HAS_GPU
template <typename T>
using dlaSeq = ChaseMpiDLACudaSeq<T>;
template <typename T>
using dlaDist = ChaseMpiDLAMultiGPU<T>;
#else
template <typename T>
using dlaSeq = ChaseMpiDLABlaslapackSeqInplace<T>;
template <typename T>
using dlaDist = ChaseMpiDLABlaslapack<T>;
#endif

class ChASE_SEQ
{
public:
    template <typename T>
    static void Initialize(int N, int nev, int nex, T* H, T* V, Base<T>* ritzv);

    template <typename T>
    static void Finalize();

    template <typename T>
    static ChaseMpi<dlaSeq, T>* getChase();

    static ChaseMpi<dlaSeq, double>* dchaseSeq;
    static ChaseMpi<dlaSeq, float>* schaseSeq;
    static ChaseMpi<dlaSeq, std::complex<double>>* zchaseSeq;
    static ChaseMpi<dlaSeq, std::complex<float>>* cchaseSeq;
};

ChaseMpi<dlaSeq, double>* ChASE_SEQ::dchaseSeq = nullptr;
ChaseMpi<dlaSeq, float>* ChASE_SEQ::schaseSeq = nullptr;
ChaseMpi<dlaSeq, std::complex<double>>* ChASE_SEQ::zchaseSeq = nullptr;
ChaseMpi<dlaSeq, std::complex<float>>* ChASE_SEQ::cchaseSeq = nullptr;

template <>
void ChASE_SEQ::Initialize(int N, int nev, int nex, double* H, double* V,
                           double* ritzv)
{
    dchaseSeq = new ChaseMpi<dlaSeq, double>(N, nev, nex, V, ritzv, H);
}

template <>
void ChASE_SEQ::Initialize(int N, int nev, int nex, float* H, float* V,
                           float* ritzv)
{
    schaseSeq = new ChaseMpi<dlaSeq, float>(N, nev, nex, V, ritzv, H);
}

template <>
void ChASE_SEQ::Initialize(int N, int nev, int nex, std::complex<double>* H,
                           std::complex<double>* V, double* ritzv)
{
    zchaseSeq =
        new ChaseMpi<dlaSeq, std::complex<double>>(N, nev, nex, V, ritzv, H);
}

template <>
void ChASE_SEQ::Initialize(int N, int nev, int nex, std::complex<float>* H,
                           std::complex<float>* V, float* ritzv)
{
    cchaseSeq =
        new ChaseMpi<dlaSeq, std::complex<float>>(N, nev, nex, V, ritzv, H);
}

template <>
void ChASE_SEQ::Finalize<double>()
{
    delete dchaseSeq;
}

template <>
void ChASE_SEQ::Finalize<float>()
{
    delete schaseSeq;
}

template <>
void ChASE_SEQ::Finalize<std::complex<float>>()
{
    delete cchaseSeq;
}

template <>
void ChASE_SEQ::Finalize<std::complex<double>>()
{
    delete zchaseSeq;
}

template <>
ChaseMpi<dlaSeq, double>* ChASE_SEQ::getChase()
{
    return dchaseSeq;
}

template <>
ChaseMpi<dlaSeq, float>* ChASE_SEQ::getChase()
{
    return schaseSeq;
}

template <>
ChaseMpi<dlaSeq, std::complex<float>>* ChASE_SEQ::getChase()
{
    return cchaseSeq;
}

template <>
ChaseMpi<dlaSeq, std::complex<double>>* ChASE_SEQ::getChase()
{
    return zchaseSeq;
}

template <typename T>
int ChASE_SEQ_Init(int N, int nev, int nex, T* H, T* V, Base<T>* ritzv)
{
    ChASE_SEQ::Initialize<T>(N, nev, nex, H, V, ritzv);
    return 1;
}

template <typename T>
int ChASE_SEQ_Finalize()
{
    ChASE_SEQ::Finalize<T>();
    return 0;
}

template <typename T>
void ChASE_SEQ_Solve(int* deg, Base<T>* tol, char* mode, char* opt)
{
    ChaseMpi<dlaSeq, T>* single = ChASE_SEQ::getChase<T>();

    ChaseConfig<T>& config = single->GetConfig();
    config.SetTol(*tol);
    config.SetDeg(*deg);
    config.SetOpt(*opt == 'S');
    config.SetApprox(*mode == 'A');
    PerformanceDecoratorChase<T> performanceDecorator(single);
    chase::Solve(&performanceDecorator);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef CHASE_OUTPUT
    Base<T>* ritzv = single->GetRitzv();
    Base<T>* resid = single->GetResid();

    if (rank == 0)
    {
        performanceDecorator.GetPerfData().print();
        std::cout << "\n\n";
        std::cout << "Printing first 5 eigenvalues and residuals\n";
        std::cout
            << "| Index |       Eigenvalue      |         Residual      |\n"
            << "|-------|-----------------------|-----------------------|"
               "\n";
        std::size_t width = 20;
        std::cout << std::setprecision(12);
        std::cout << std::setfill(' ');
        std::cout << std::scientific;
        std::cout << std::right;
        for (auto i = 0; i < std::min(single->GetNev(), std::size_t(5)); ++i)
            std::cout << "|  " << std::setw(4) << i + 1 << " | "
                      << std::setw(width) << ritzv[i] << "  | "
                      << std::setw(width) << resid[i] << "  |\n";
        std::cout << "\n\n\n";
    }
#endif
}

class ChASE_DIST
{
public:
    template <typename T>
    static void Initialize(int N, int nev, int nex, int m, int n, T* H, int ldh,
                           T* V, Base<T>* ritzv, int dim0, int dim1,
                           char* grid_major, MPI_Comm comm);

    template <typename T>
    static void Initialize(int N, int nev, int nex, int mbsize, int nbsize,
                           T* H, int ldh, T* V, Base<T>* ritzv, int dim0,
                           int dim1, char* grid_major, int irsrc, int icsrc,
                           MPI_Comm comm);

    template <typename T>
    static void Finalize();

    template <typename T>
    static ChaseMpi<dlaDist, T>* getChase();

    template <typename T>
    static ChaseMpiProperties<T>* getProperties();

    static ChaseMpiProperties<double>* d_props;
    static ChaseMpiProperties<std::complex<double>>* z_props;
    static ChaseMpiProperties<float>* s_props;
    static ChaseMpiProperties<std::complex<float>>* c_props;

    static ChaseMpi<dlaDist, double>* dchaseDist;
    static ChaseMpi<dlaDist, float>* schaseDist;
    static ChaseMpi<dlaDist, std::complex<double>>* zchaseDist;
    static ChaseMpi<dlaDist, std::complex<float>>* cchaseDist;
};

ChaseMpiProperties<double>* ChASE_DIST::d_props = nullptr;
ChaseMpiProperties<std::complex<double>>* ChASE_DIST::z_props = nullptr;
ChaseMpiProperties<float>* ChASE_DIST::s_props = nullptr;
ChaseMpiProperties<std::complex<float>>* ChASE_DIST::c_props = nullptr;

ChaseMpi<dlaDist, double>* ChASE_DIST::dchaseDist = nullptr;
ChaseMpi<dlaDist, float>* ChASE_DIST::schaseDist = nullptr;
ChaseMpi<dlaDist, std::complex<double>>* ChASE_DIST::zchaseDist = nullptr;
ChaseMpi<dlaDist, std::complex<float>>* ChASE_DIST::cchaseDist = nullptr;

template <>
void ChASE_DIST::Initialize(int N, int nev, int nex, int m, int n, double* H,
                            int ldh, double* V, double* ritzv, int dim0,
                            int dim1, char* grid_major, MPI_Comm comm)
{
    d_props = new ChaseMpiProperties<double>(N, nev, nex, m, n, dim0, dim1,
                                             grid_major, comm);
    dchaseDist = new ChaseMpi<dlaDist, double>(d_props, H, ldh, V, ritzv);
}

template <>
void ChASE_DIST::Initialize(int N, int nev, int nex, int mbsize, int nbsize,
                            double* H, int ldh, double* V, double* ritzv,
                            int dim0, int dim1, char* grid_major, int irsrc,
                            int icsrc, MPI_Comm comm)
{
    d_props =
        new ChaseMpiProperties<double>(N, mbsize, nbsize, nev, nex, dim0, dim1,
                                       grid_major, irsrc, icsrc, comm);
    dchaseDist = new ChaseMpi<dlaDist, double>(d_props, H, ldh, V, ritzv);
}

template <>
void ChASE_DIST::Initialize(int N, int nev, int nex, int m, int n, float* H,
                            int ldh, float* V, float* ritzv, int dim0, int dim1,
                            char* grid_major, MPI_Comm comm)
{
    s_props = new ChaseMpiProperties<float>(N, nev, nex, m, n, dim0, dim1,
                                            grid_major, comm);
    schaseDist = new ChaseMpi<dlaDist, float>(s_props, H, ldh, V, ritzv);
}

template <>
void ChASE_DIST::Initialize(int N, int nev, int nex, int mbsize, int nbsize,
                            float* H, int ldh, float* V, float* ritzv, int dim0,
                            int dim1, char* grid_major, int irsrc, int icsrc,
                            MPI_Comm comm)
{
    s_props =
        new ChaseMpiProperties<float>(N, mbsize, nbsize, nev, nex, dim0, dim1,
                                      grid_major, irsrc, icsrc, comm);
    schaseDist = new ChaseMpi<dlaDist, float>(s_props, H, ldh, V, ritzv);
}

template <>
void ChASE_DIST::Initialize(int N, int nev, int nex, int m, int n,
                            std::complex<float>* H, int ldh,
                            std::complex<float>* V, float* ritzv, int dim0,
                            int dim1, char* grid_major, MPI_Comm comm)
{
    c_props = new ChaseMpiProperties<std::complex<float>>(
        N, nev, nex, m, n, dim0, dim1, grid_major, comm);
    cchaseDist =
        new ChaseMpi<dlaDist, std::complex<float>>(c_props, H, ldh, V, ritzv);
}

template <>
void ChASE_DIST::Initialize(int N, int nev, int nex, int mbsize, int nbsize,
                            std::complex<float>* H, int ldh,
                            std::complex<float>* V, float* ritzv, int dim0,
                            int dim1, char* grid_major, int irsrc, int icsrc,
                            MPI_Comm comm)
{
    c_props = new ChaseMpiProperties<std::complex<float>>(
        N, mbsize, nbsize, nev, nex, dim0, dim1, grid_major, irsrc, icsrc,
        comm);
    cchaseDist =
        new ChaseMpi<dlaDist, std::complex<float>>(c_props, H, ldh, V, ritzv);
}

template <>
void ChASE_DIST::Initialize(int N, int nev, int nex, int m, int n,
                            std::complex<double>* H, int ldh,
                            std::complex<double>* V, double* ritzv, int dim0,
                            int dim1, char* grid_major, MPI_Comm comm)
{
    z_props = new ChaseMpiProperties<std::complex<double>>(
        N, nev, nex, m, n, dim0, dim1, grid_major, comm);
    zchaseDist =
        new ChaseMpi<dlaDist, std::complex<double>>(z_props, H, ldh, V, ritzv);
}

template <>
void ChASE_DIST::Initialize(int N, int nev, int nex, int mbsize, int nbsize,
                            std::complex<double>* H, int ldh,
                            std::complex<double>* V, double* ritzv, int dim0,
                            int dim1, char* grid_major, int irsrc, int icsrc,
                            MPI_Comm comm)
{
    z_props = new ChaseMpiProperties<std::complex<double>>(
        N, mbsize, nbsize, nev, nex, dim0, dim1, grid_major, irsrc, icsrc,
        comm);
    zchaseDist =
        new ChaseMpi<dlaDist, std::complex<double>>(z_props, H, ldh, V, ritzv);
}

template <>
void ChASE_DIST::Finalize<double>()
{
    delete dchaseDist;
}

template <>
void ChASE_DIST::Finalize<float>()
{
    delete schaseDist;
}

template <>
void ChASE_DIST::Finalize<std::complex<float>>()
{
    delete cchaseDist;
}

template <>
void ChASE_DIST::Finalize<std::complex<double>>()
{
    delete zchaseDist;
}

template <>
ChaseMpi<dlaDist, double>* ChASE_DIST::getChase()
{
    return dchaseDist;
}

template <>
ChaseMpi<dlaDist, float>* ChASE_DIST::getChase()
{
    return schaseDist;
}

template <>
ChaseMpi<dlaDist, std::complex<float>>* ChASE_DIST::getChase()
{
    return cchaseDist;
}

template <>
ChaseMpi<dlaDist, std::complex<double>>* ChASE_DIST::getChase()
{
    return zchaseDist;
}

template <>
ChaseMpiProperties<double>* ChASE_DIST::getProperties()
{
    return d_props;
}

template <>
ChaseMpiProperties<float>* ChASE_DIST::getProperties()
{
    return s_props;
}

template <>
ChaseMpiProperties<std::complex<float>>* ChASE_DIST::getProperties()
{
    return c_props;
}

template <>
ChaseMpiProperties<std::complex<double>>* ChASE_DIST::getProperties()
{
    return z_props;
}

template <typename T>
int ChASE_DIST_Init(int N, int nev, int nex, int m, int n, T* H, int ldh, T* V,
                    Base<T>* ritzv, int dim0, int dim1, char* grid_major,
                    MPI_Comm comm)
{
    ChASE_DIST::Initialize<T>(N, nev, nex, m, n, H, ldh, V, ritzv, dim0, dim1,
                              grid_major, comm);
    return 1;
}

template <typename T>
int ChASE_DIST_Init(int N, int nev, int nex, int mbsize, int nbsize, T* H,
                    int ldh, T* V, Base<T>* ritzv, int dim0, int dim1,
                    char* grid_major, int irsrc, int icsrc, MPI_Comm comm)
{
    ChASE_DIST::Initialize<T>(N, nev, nex, mbsize, nbsize, H, ldh, V, ritzv,
                              dim0, dim1, grid_major, irsrc, icsrc, comm);
    return 1;
}

template <typename T>
int ChASE_DIST_Finalize()
{
    ChASE_DIST::Finalize<T>();
    return 0;
}

template <typename T>
void ChASE_DIST_Solve(int* deg, Base<T>* tol, char* mode, char* opt)
{
    ChaseMpi<dlaDist, T>* single = ChASE_DIST::getChase<T>();

    ChaseConfig<T>& config = single->GetConfig();
    config.SetTol(*tol);
    config.SetDeg(*deg);
    config.SetOpt(*opt == 'S');
    config.SetApprox(*mode == 'A');
    PerformanceDecoratorChase<T> performanceDecorator(single);
    chase::Solve(&performanceDecorator);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef CHASE_OUTPUT
    Base<T>* ritzv = single->GetRitzv();
    Base<T>* resid = single->GetResid();

    if (rank == 0)
    {
        performanceDecorator.GetPerfData().print();
        std::cout << "\n\n";
        std::cout << "Printing first 5 eigenvalues and residuals\n";
        std::cout
            << "| Index |       Eigenvalue      |         Residual      |\n"
            << "|-------|-----------------------|-----------------------|"
               "\n";
        std::size_t width = 20;
        std::cout << std::setprecision(12);
        std::cout << std::setfill(' ');
        std::cout << std::scientific;
        std::cout << std::right;
        for (auto i = 0; i < std::min(single->GetNev(), std::size_t(5)); ++i)
            std::cout << "|  " << std::setw(4) << i + 1 << " | "
                      << std::setw(width) << ritzv[i] << "  | "
                      << std::setw(width) << resid[i] << "  |\n";
        std::cout << "\n\n\n";
    }
#endif
}

extern "C"
{
/** @defgroup chase-c ChASE C Interface
 *  @brief This module provides a C interface of ChASE  
 *  @{
 */

    //! Initialization of shared-memory ChASE with real scalar in double precison.
    //! It is linked to single-GPU ChASE when CUDA is detected.
    //!    
    //!
    //! @param[in] n global matrix size of the matrix to be diagonalized  
    //! @param[in] nev number of desired eigenpairs
    //! @param[in] nex extra searching space size      
    //! @param[in] h pointer to the matrix to be diagonalized
    //! @param[in,out] v `(nx(nev+nex))` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors
    //! @param[in,out] ritzv an array of size `nev` which contains the desired eigenvalues
    //! @param[in,out] init a flag to indicate if ChASE has been initialized
    void dchase_init_(int* N, int* nev, int* nex, double* H, double* V,
                      double* ritzv, int* init)
    {
        *init = ChASE_SEQ_Init<double>(*N, *nev, *nex, H, V, ritzv);
    }
    void schase_init_(int* N, int* nev, int* nex, float* H, float* V,
                      float* ritzv, int* init)
    {
        *init = ChASE_SEQ_Init<float>(*N, *nev, *nex, H, V, ritzv);
    }
    void cchase_init_(int* N, int* nev, int* nex, float _Complex* H,
                      float _Complex* V, float* ritzv, int* init)
    {
        *init = ChASE_SEQ_Init<std::complex<float>>(
            *N, *nev, *nex, reinterpret_cast<std::complex<float>*>(H),
            reinterpret_cast<std::complex<float>*>(V), ritzv);
    }
    void zchase_init_(int* N, int* nev, int* nex, double _Complex* H,
                      double _Complex* V, double* ritzv, int* init)
    {
        *init = ChASE_SEQ_Init<std::complex<double>>(
            *N, *nev, *nex, reinterpret_cast<std::complex<double>*>(H),
            reinterpret_cast<std::complex<double>*>(V), ritzv);
    }

    void dchase_finalize_(int* flag) { *flag = ChASE_SEQ_Finalize<double>(); }
    void schase_finalize_(int* flag) { *flag = ChASE_SEQ_Finalize<float>(); }
    void cchase_finalize_(int* flag)
    {
        *flag = ChASE_SEQ_Finalize<std::complex<float>>();
    }
    void zchase_finalize_(int* flag)
    {
        *flag = ChASE_SEQ_Finalize<std::complex<double>>();
    }

    void dchase_(int* deg, double* tol, char* mode, char* opt)
    {
        ChASE_SEQ_Solve<double>(deg, tol, mode, opt);
    }
    void schase_(int* deg, float* tol, char* mode, char* opt)
    {
        ChASE_SEQ_Solve<float>(deg, tol, mode, opt);
    }
    void zchase_(int* deg, double* tol, char* mode, char* opt)
    {
        ChASE_SEQ_Solve<std::complex<double>>(deg, tol, mode, opt);
    }
    void cchase_(int* deg, float* tol, char* mode, char* opt)
    {
        ChASE_SEQ_Solve<std::complex<float>>(deg, tol, mode, opt);
    }

    void pdchase_init_(int* N, int* nev, int* nex, int* m, int* n, double* H,
                       int* ldh, double* V, double* ritzv, int* dim0, int* dim1,
                       char* grid_major, MPI_Comm* comm, int* init)
    {
        *init = ChASE_DIST_Init<double>(*N, *nev, *nex, *m, *n, H, *ldh, V,
                                        ritzv, *dim0, *dim1, grid_major, *comm);
    }

    void pdchase_init_f_(int* N, int* nev, int* nex, int* m, int* n, double* H,
                         int* ldh, double* V, double* ritzv, int* dim0,
                         int* dim1, char* grid_major, MPI_Fint* fcomm,
                         int* init)
    {
        MPI_Comm comm = MPI_Comm_f2c(*fcomm);
        *init = ChASE_DIST_Init<double>(*N, *nev, *nex, *m, *n, H, *ldh, V,
                                        ritzv, *dim0, *dim1, grid_major, comm);
    }

    void pschase_init_(int* N, int* nev, int* nex, int* m, int* n, float* H,
                       int* ldh, float* V, float* ritzv, int* dim0, int* dim1,
                       char* grid_major, MPI_Comm* comm, int* init)
    {
        *init = ChASE_DIST_Init<float>(*N, *nev, *nex, *m, *n, H, *ldh, V,
                                       ritzv, *dim0, *dim1, grid_major, *comm);
    }

    void pschase_init_f_(int* N, int* nev, int* nex, int* m, int* n, float* H,
                         int* ldh, float* V, float* ritzv, int* dim0, int* dim1,
                         char* grid_major, MPI_Fint* fcomm, int* init)
    {
        MPI_Comm comm = MPI_Comm_f2c(*fcomm);
        *init = ChASE_DIST_Init<float>(*N, *nev, *nex, *m, *n, H, *ldh, V,
                                       ritzv, *dim0, *dim1, grid_major, comm);
    }

    void pzchase_init_(int* N, int* nev, int* nex, int* m, int* n,
                       double _Complex* H, int* ldh, double _Complex* V,
                       double* ritzv, int* dim0, int* dim1, char* grid_major,
                       MPI_Comm* comm, int* init)
    {
        *init = ChASE_DIST_Init<std::complex<double>>(
            *N, *nev, *nex, *m, *n, reinterpret_cast<std::complex<double>*>(H),
            *ldh, reinterpret_cast<std::complex<double>*>(V), ritzv, *dim0,
            *dim1, grid_major, *comm);
    }

    void pzchase_init_f_(int* N, int* nev, int* nex, int* m, int* n,
                         double _Complex* H, int* ldh, double _Complex* V,
                         double* ritzv, int* dim0, int* dim1, char* grid_major,
                         MPI_Fint* fcomm, int* init)
    {
        MPI_Comm comm = MPI_Comm_f2c(*fcomm);
        *init = ChASE_DIST_Init<std::complex<double>>(
            *N, *nev, *nex, *m, *n, reinterpret_cast<std::complex<double>*>(H),
            *ldh, reinterpret_cast<std::complex<double>*>(V), ritzv, *dim0,
            *dim1, grid_major, comm);
    }

    void pcchase_init_(int* N, int* nev, int* nex, int* m, int* n,
                       float _Complex* H, int* ldh, float _Complex* V,
                       float* ritzv, int* dim0, int* dim1, char* grid_major,
                       MPI_Comm* comm, int* init)
    {
        *init = ChASE_DIST_Init<std::complex<float>>(
            *N, *nev, *nex, *m, *n, reinterpret_cast<std::complex<float>*>(H),
            *ldh, reinterpret_cast<std::complex<float>*>(V), ritzv, *dim0,
            *dim1, grid_major, *comm);
    }

    void pcchase_init_f_(int* N, int* nev, int* nex, int* m, int* n,
                         float _Complex* H, int* ldh, float _Complex* V,
                         float* ritzv, int* dim0, int* dim1, char* grid_major,
                         MPI_Fint* fcomm, int* init)
    {
        MPI_Comm comm = MPI_Comm_f2c(*fcomm);
        *init = ChASE_DIST_Init<std::complex<float>>(
            *N, *nev, *nex, *m, *n, reinterpret_cast<std::complex<float>*>(H),
            *ldh, reinterpret_cast<std::complex<float>*>(V), ritzv, *dim0,
            *dim1, grid_major, comm);
    }

    void pdchase_init_blockcyclic_(int* N, int* nev, int* nex, int* mbsize,
                                   int* nbsize, double* H, int* ldh, double* V,
                                   double* ritzv, int* dim0, int* dim1,
                                   char* grid_major, int* irsrc, int* icsrc,
                                   MPI_Comm* comm, int* init)
    {
        *init = ChASE_DIST_Init<double>(*N, *nev, *nex, *mbsize, *nbsize, H,
                                        *ldh, V, ritzv, *dim0, *dim1,
                                        grid_major, *irsrc, *icsrc, *comm);
    }

    void pdchase_init_blockcyclic_f_(int* N, int* nev, int* nex, int* mbsize,
                                     int* nbsize, double* H, int* ldh,
                                     double* V, double* ritzv, int* dim0,
                                     int* dim1, char* grid_major, int* irsrc,
                                     int* icsrc, MPI_Fint* fcomm, int* init)
    {
        MPI_Comm comm = MPI_Comm_f2c(*fcomm);
        *init = ChASE_DIST_Init<double>(*N, *nev, *nex, *mbsize, *nbsize, H,
                                        *ldh, V, ritzv, *dim0, *dim1,
                                        grid_major, *irsrc, *icsrc, comm);
    }

    void pschase_init_blockcyclic_(int* N, int* nev, int* nex, int* mbsize,
                                   int* nbsize, float* H, int* ldh, float* V,
                                   float* ritzv, int* dim0, int* dim1,
                                   char* grid_major, int* irsrc, int* icsrc,
                                   MPI_Comm* comm, int* init)
    {
        *init = ChASE_DIST_Init<float>(*N, *nev, *nex, *mbsize, *nbsize, H,
                                       *ldh, V, ritzv, *dim0, *dim1, grid_major,
                                       *irsrc, *icsrc, *comm);
    }

    void pschase_init_blockcyclic_f_(int* N, int* nev, int* nex, int* mbsize,
                                     int* nbsize, float* H, int* ldh, float* V,
                                     float* ritzv, int* dim0, int* dim1,
                                     char* grid_major, int* irsrc, int* icsrc,
                                     MPI_Fint* fcomm, int* init)
    {
        MPI_Comm comm = MPI_Comm_f2c(*fcomm);
        *init = ChASE_DIST_Init<float>(*N, *nev, *nex, *mbsize, *nbsize, H,
                                       *ldh, V, ritzv, *dim0, *dim1, grid_major,
                                       *irsrc, *icsrc, comm);
    }

    void pcchase_init_blockcyclic_(int* N, int* nev, int* nex, int* mbsize,
                                   int* nbsize, float _Complex* H, int* ldh,
                                   float _Complex* V, float* ritzv, int* dim0,
                                   int* dim1, char* grid_major, int* irsrc,
                                   int* icsrc, MPI_Comm* comm, int* init)
    {
        *init = ChASE_DIST_Init<std::complex<float>>(
            *N, *nev, *nex, *mbsize, *nbsize,
            reinterpret_cast<std::complex<float>*>(H), *ldh,
            reinterpret_cast<std::complex<float>*>(V), ritzv, *dim0, *dim1,
            grid_major, *irsrc, *icsrc, *comm);
    }

    void pcchase_init_blockcyclic_f_(int* N, int* nev, int* nex, int* mbsize,
                                     int* nbsize, float _Complex* H, int* ldh,
                                     float _Complex* V, float* ritzv, int* dim0,
                                     int* dim1, char* grid_major, int* irsrc,
                                     int* icsrc, MPI_Fint* fcomm, int* init)
    {
        MPI_Comm comm = MPI_Comm_f2c(*fcomm);
        *init = ChASE_DIST_Init<std::complex<float>>(
            *N, *nev, *nex, *mbsize, *nbsize,
            reinterpret_cast<std::complex<float>*>(H), *ldh,
            reinterpret_cast<std::complex<float>*>(V), ritzv, *dim0, *dim1,
            grid_major, *irsrc, *icsrc, comm);
    }

    void pzchase_init_blockcyclic_(int* N, int* nev, int* nex, int* mbsize,
                                   int* nbsize, double _Complex* H, int* ldh,
                                   double _Complex* V, double* ritzv, int* dim0,
                                   int* dim1, char* grid_major, int* irsrc,
                                   int* icsrc, MPI_Comm* comm, int* init)
    {
        *init = ChASE_DIST_Init<std::complex<double>>(
            *N, *nev, *nex, *mbsize, *nbsize,
            reinterpret_cast<std::complex<double>*>(H), *ldh,
            reinterpret_cast<std::complex<double>*>(V), ritzv, *dim0, *dim1,
            grid_major, *irsrc, *icsrc, *comm);
    }

    void pzchase_init_blockcyclic_f_(int* N, int* nev, int* nex, int* mbsize,
                                     int* nbsize, double _Complex* H, int* ldh,
                                     double _Complex* V, double* ritzv,
                                     int* dim0, int* dim1, char* grid_major,
                                     int* irsrc, int* icsrc, MPI_Fint* fcomm,
                                     int* init)
    {
        MPI_Comm comm = MPI_Comm_f2c(*fcomm);
        *init = ChASE_DIST_Init<std::complex<double>>(
            *N, *nev, *nex, *mbsize, *nbsize,
            reinterpret_cast<std::complex<double>*>(H), *ldh,
            reinterpret_cast<std::complex<double>*>(V), ritzv, *dim0, *dim1,
            grid_major, *irsrc, *icsrc, comm);
    }

    void pdchase_finalize_(int* flag) { *flag = ChASE_DIST_Finalize<double>(); }
    void pschase_finalize_(int* flag) { *flag = ChASE_DIST_Finalize<float>(); }
    void pcchase_finalize_(int* flag)
    {
        *flag = ChASE_DIST_Finalize<std::complex<float>>();
    }
    void pzchase_finalize_(int* flag)
    {
        *flag = ChASE_DIST_Finalize<std::complex<double>>();
    }

    void pdchase_(int* deg, double* tol, char* mode, char* opt)
    {
        ChASE_DIST_Solve<double>(deg, tol, mode, opt);
    }
    void pschase_(int* deg, float* tol, char* mode, char* opt)
    {
        ChASE_DIST_Solve<float>(deg, tol, mode, opt);
    }
    void pzchase_(int* deg, double* tol, char* mode, char* opt)
    {
        ChASE_DIST_Solve<std::complex<double>>(deg, tol, mode, opt);
    }
    void pcchase_(int* deg, float* tol, char* mode, char* opt)
    {
        ChASE_DIST_Solve<std::complex<float>>(deg, tol, mode, opt);
    }
/** @} */ // end of chase-c
} // extern C
