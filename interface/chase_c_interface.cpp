// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <complex>
#include <iostream>
#include <iomanip>
#include <memory>
#include <type_traits>

#include "algorithm/performance.hpp"
#include "chase_c_interface.h"

#ifdef HAS_CUDA
#include "Impl/chase_gpu/chase_gpu.hpp"
template <typename T>
using SeqSolverType = chase::Impl::ChASEGPU<T>;
#else
#include "Impl/chase_cpu/chase_cpu.hpp"
template <typename T>
using SeqSolverType = chase::Impl::ChASECPU<T>;
#endif

#ifdef HAS_CUDA
#include "Impl/pchase_gpu/pchase_gpu.hpp"
using ARCH = chase::platform::GPU;
#ifdef HAS_NCCL
#ifdef CHASE_INTERFACE_DISABLE_NCCL
template<typename MatrixType>
using DistSolverType = chase::Impl::pChASEGPU<MatrixType, typename ColumnMultiVectorType<MatrixType>::type, chase::grid::backend::MPI>;
#else
template<typename MatrixType>
using DistSolverType = chase::Impl::pChASEGPU<MatrixType, typename ColumnMultiVectorType<MatrixType>::type>;
#endif
#else
template<typename MatrixType>
using DistSolverType = chase::Impl::pChASEGPU<MatrixType, typename ColumnMultiVectorType<MatrixType>::type, chase::grid::backend::MPI>;
#endif
#else
#include "Impl/pchase_cpu/pchase_cpu.hpp"
using ARCH = chase::platform::CPU;
template<typename MatrixType>
using DistSolverType = chase::Impl::pChASECPU<MatrixType, typename ColumnMultiVectorType<MatrixType>::type>; 
#endif

// Type aliases removed - using explicit types throughout


class ChASE_SEQ
{
public:
    template <typename T>
    static int Initialize(int N, int nev, int nex, T* H, int ldh, T* V, chase::Base<T>* ritzv);

    template <typename T>
    static int Finalize();

    template <typename T>
    static SeqSolverType<T>* getChase();

    static SeqSolverType<double>* dchaseSeq;
    static SeqSolverType<float>* schaseSeq;
    static SeqSolverType<std::complex<double>>* zchaseSeq;
    static SeqSolverType<std::complex<float>>* cchaseSeq;
};

SeqSolverType<double>* ChASE_SEQ::dchaseSeq = nullptr;
SeqSolverType<float>* ChASE_SEQ::schaseSeq = nullptr;
SeqSolverType<std::complex<double>>* ChASE_SEQ::zchaseSeq = nullptr;
SeqSolverType<std::complex<float>>* ChASE_SEQ::cchaseSeq = nullptr;

template <>
int ChASE_SEQ::Initialize(int N, int nev, int nex, double* H, int ldh, double* V,
                           double* ritzv)
{
    if (dchaseSeq) delete dchaseSeq;
    dchaseSeq = new SeqSolverType<double>(N, nev, nex, H, ldh, V, N, ritzv);
    return 1;
}

template <>
int ChASE_SEQ::Initialize(int N, int nev, int nex, float* H, int ldh, float* V,
                           float* ritzv)
{
    if (schaseSeq) delete schaseSeq;
    schaseSeq = new SeqSolverType<float>(N, nev, nex, H, ldh, V, N, ritzv);
    return 1;
}

template <>
int ChASE_SEQ::Initialize(int N, int nev, int nex, std::complex<double>* H, int ldh,
                           std::complex<double>* V, double* ritzv)
{
    if (zchaseSeq) delete zchaseSeq;
    zchaseSeq = new SeqSolverType<std::complex<double>>(N, nev, nex, H, ldh, V, N, ritzv);
    return 1;
}

template <>
int ChASE_SEQ::Initialize(int N, int nev, int nex, std::complex<float>* H, int ldh,
                           std::complex<float>* V, float* ritzv)
{
    if (cchaseSeq) delete cchaseSeq;
    cchaseSeq = new SeqSolverType<std::complex<float>>(N, nev, nex, H, ldh, V, N, ritzv);
    return 1;
}

template <>
int ChASE_SEQ::Finalize<double>()
{
    delete dchaseSeq;
    dchaseSeq = nullptr;
    return 0;
}

template <>
int ChASE_SEQ::Finalize<float>()
{
    delete schaseSeq;
    schaseSeq = nullptr;
    return 0;
}

template <>
int ChASE_SEQ::Finalize<std::complex<float>>()
{
    delete cchaseSeq;
    cchaseSeq = nullptr;
    return 0;
}

template <>
int ChASE_SEQ::Finalize<std::complex<double>>()
{
    delete zchaseSeq;
    zchaseSeq = nullptr;
    return 0;
}

template <>
SeqSolverType<double>* ChASE_SEQ::getChase()
{
    return dchaseSeq;
}

template <>
SeqSolverType<float>* ChASE_SEQ::getChase()
{
    return schaseSeq;
}

template <>
SeqSolverType<std::complex<float>>* ChASE_SEQ::getChase()
{
    return cchaseSeq;
}

template <>
SeqSolverType<std::complex<double>>* ChASE_SEQ::getChase()
{
    return zchaseSeq;
}

template <typename T>
void ChASE_SEQ_Solve(int* deg, chase::Base<T>* tol, char* mode, char* opt, char* qr )
{
    SeqSolverType<T>* single = ChASE_SEQ::getChase<T>();
    
    chase::ChaseConfig<T>& config = single->GetConfig();
    config.SetTol(*tol);
    config.SetDeg(*deg);
    config.SetOpt(*opt == 'S');
    config.SetApprox(*mode == 'A');
    config.SetCholQR(*qr == 'C');
    config.EnableSymCheck(false);

    chase::PerformanceDecoratorChase<T> performanceDecorator(single);
    chase::Solve(&performanceDecorator);   

    chase::Base<T>* ritzv = single->GetRitzv();
    chase::Base<T>* resid = single->GetResid();

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

template<typename MatrixType>
class ChASE_DIST
{
    using T = typename MatrixType::value_type;
public:

    static int Initialize(int N, int nev, int nex, int mbsize, int nbsize,
                                T* H, int ldh, T* V, chase::Base<T>* ritzv,
                                int dim0, int dim1, char* grid_major, int irsrc,
                                int icsrc, MPI_Comm comm);
    
    static int Initialize(int N, int nev, int nex, int m, int n, T* H, int ldh,
               T* V, chase::Base<T>* ritzv, int dim0, int dim1,
               char* grid_major, MPI_Comm comm);

    static DistSolverType<chase::distMatrix::BlockCyclicMatrix<std::complex<double>, ARCH>> *zchaseDist_cyclic;
    static DistSolverType<chase::distMatrix::BlockCyclicMatrix<std::complex<float>, ARCH>> *cchaseDist_cyclic;
    static DistSolverType<chase::distMatrix::BlockCyclicMatrix<float, ARCH>> *schaseDist_cyclic;
    static DistSolverType<chase::distMatrix::BlockCyclicMatrix<double, ARCH>> *dchaseDist_cyclic;

    static DistSolverType<chase::distMatrix::BlockBlockMatrix<std::complex<double>, ARCH>> *zchaseDist_block;
    static DistSolverType<chase::distMatrix::BlockBlockMatrix<std::complex<float>, ARCH>> *cchaseDist_block;
    static DistSolverType<chase::distMatrix::BlockBlockMatrix<float, ARCH>> *schaseDist_block;
    static DistSolverType<chase::distMatrix::BlockBlockMatrix<double, ARCH>> *dchaseDist_block;

    static chase::distMatrix::BlockCyclicMatrix<std::complex<double>, ARCH>* zHmat_cyclic;
    static chase::distMatrix::BlockCyclicMatrix<std::complex<float>, ARCH>* cHmat_cyclic;
    static chase::distMatrix::BlockCyclicMatrix<double, ARCH>* dHmat_cyclic;
    static chase::distMatrix::BlockCyclicMatrix<float, ARCH>* sHmat_cyclic;

    static chase::distMatrix::BlockBlockMatrix<std::complex<double>, ARCH>* zHmat_block;
    static chase::distMatrix::BlockBlockMatrix<std::complex<float>, ARCH>* cHmat_block;
    static chase::distMatrix::BlockBlockMatrix<double, ARCH>* dHmat_block;
    static chase::distMatrix::BlockBlockMatrix<float, ARCH>* sHmat_block;

    // PseudoHermitian matrix pointers (separate from base class pointers)
    static chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<double>, ARCH>* zHmat_block_pseudo;
    static chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<float>, ARCH>* cHmat_block_pseudo;
    static chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<double>, ARCH>* zHmat_cyclic_pseudo;
    static chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<float>, ARCH>* cHmat_cyclic_pseudo;

    static chase::distMultiVector::DistMultiVectorBlockCyclic1D<std::complex<double>, chase::distMultiVector::CommunicatorType::column, ARCH>* zVec_cyclic;
    static chase::distMultiVector::DistMultiVectorBlockCyclic1D<std::complex<float>, chase::distMultiVector::CommunicatorType::column, ARCH>* cVec_cyclic;
    static chase::distMultiVector::DistMultiVectorBlockCyclic1D<double, chase::distMultiVector::CommunicatorType::column, ARCH>* dVec_cyclic;
    static chase::distMultiVector::DistMultiVectorBlockCyclic1D<float, chase::distMultiVector::CommunicatorType::column, ARCH>* sVec_cyclic;

    static chase::distMultiVector::DistMultiVector1D<std::complex<double>, chase::distMultiVector::CommunicatorType::column, ARCH>* zVec_block;
    static chase::distMultiVector::DistMultiVector1D<std::complex<float>, chase::distMultiVector::CommunicatorType::column, ARCH>* cVec_block;
    static chase::distMultiVector::DistMultiVector1D<double, chase::distMultiVector::CommunicatorType::column, ARCH>* dVec_block;
    static chase::distMultiVector::DistMultiVector1D<float, chase::distMultiVector::CommunicatorType::column, ARCH>* sVec_block;

    // PseudoHermitian solver pointers (separate because DistSolverType is templated on exact type)
    // Matrices are stored in base class pointers above (polymorphism)
    static DistSolverType<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<double>, ARCH>> *zchaseDist_pseudo_block;
    static DistSolverType<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<float>, ARCH>> *cchaseDist_pseudo_block;
    static DistSolverType<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<double>, ARCH>> *zchaseDist_pseudo_cyclic;
    static DistSolverType<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<float>, ARCH>> *cchaseDist_pseudo_cyclic;

    static DistSolverType<MatrixType>* getChase();
    static int Finalize();
    static void Solve(int* deg, chase::Base<T>* tol, char* mode, char* opt, char *qr);
    static void WrteHam(const std::string& filename);
    static void readHam(const std::string& filename);

};

// BlockCyclic static member definitions
template <typename MatrixType>
DistSolverType<chase::distMatrix::BlockCyclicMatrix<std::complex<double>, ARCH>> *ChASE_DIST<MatrixType>::zchaseDist_cyclic = nullptr;
template <typename MatrixType>
DistSolverType<chase::distMatrix::BlockCyclicMatrix<std::complex<float>, ARCH>> *ChASE_DIST<MatrixType>::cchaseDist_cyclic = nullptr;
template <typename MatrixType>
DistSolverType<chase::distMatrix::BlockCyclicMatrix<float, ARCH>> *ChASE_DIST<MatrixType>::schaseDist_cyclic = nullptr;
template <typename MatrixType>
DistSolverType<chase::distMatrix::BlockCyclicMatrix<double, ARCH>> *ChASE_DIST<MatrixType>::dchaseDist_cyclic = nullptr;

template <typename MatrixType>
chase::distMatrix::BlockCyclicMatrix<std::complex<double>, ARCH>* ChASE_DIST<MatrixType>::zHmat_cyclic = nullptr;
template <typename MatrixType>
chase::distMatrix::BlockCyclicMatrix<std::complex<float>, ARCH>* ChASE_DIST<MatrixType>::cHmat_cyclic = nullptr;
template <typename MatrixType>
chase::distMatrix::BlockCyclicMatrix<double, ARCH>* ChASE_DIST<MatrixType>::dHmat_cyclic = nullptr;
template <typename MatrixType>
chase::distMatrix::BlockCyclicMatrix<float, ARCH>* ChASE_DIST<MatrixType>::sHmat_cyclic = nullptr;

template <typename MatrixType>
chase::distMultiVector::DistMultiVectorBlockCyclic1D<std::complex<double>, chase::distMultiVector::CommunicatorType::column, ARCH>* ChASE_DIST<MatrixType>::zVec_cyclic = nullptr;
template <typename MatrixType>
chase::distMultiVector::DistMultiVectorBlockCyclic1D<std::complex<float>, chase::distMultiVector::CommunicatorType::column, ARCH>* ChASE_DIST<MatrixType>::cVec_cyclic = nullptr;
template <typename MatrixType>
chase::distMultiVector::DistMultiVectorBlockCyclic1D<double, chase::distMultiVector::CommunicatorType::column, ARCH>* ChASE_DIST<MatrixType>::dVec_cyclic = nullptr;
template <typename MatrixType>
chase::distMultiVector::DistMultiVectorBlockCyclic1D<float, chase::distMultiVector::CommunicatorType::column, ARCH>* ChASE_DIST<MatrixType>::sVec_cyclic = nullptr;

// BlockBlock static member definitions
template <typename MatrixType>
DistSolverType<chase::distMatrix::BlockBlockMatrix<std::complex<double>, ARCH>> *ChASE_DIST<MatrixType>::zchaseDist_block = nullptr;
template <typename MatrixType>
DistSolverType<chase::distMatrix::BlockBlockMatrix<std::complex<float>, ARCH>> *ChASE_DIST<MatrixType>::cchaseDist_block = nullptr;
template <typename MatrixType>
DistSolverType<chase::distMatrix::BlockBlockMatrix<float, ARCH>> *ChASE_DIST<MatrixType>::schaseDist_block = nullptr;
template <typename MatrixType>
DistSolverType<chase::distMatrix::BlockBlockMatrix<double, ARCH>> *ChASE_DIST<MatrixType>::dchaseDist_block = nullptr;

template <typename MatrixType>
chase::distMatrix::BlockBlockMatrix<std::complex<double>, ARCH>* ChASE_DIST<MatrixType>::zHmat_block = nullptr;
template <typename MatrixType>
chase::distMatrix::BlockBlockMatrix<std::complex<float>, ARCH>* ChASE_DIST<MatrixType>::cHmat_block = nullptr;
template <typename MatrixType>
chase::distMatrix::BlockBlockMatrix<double, ARCH>* ChASE_DIST<MatrixType>::dHmat_block = nullptr;
template <typename MatrixType>
chase::distMatrix::BlockBlockMatrix<float, ARCH>* ChASE_DIST<MatrixType>::sHmat_block = nullptr;

template <typename MatrixType>
chase::distMultiVector::DistMultiVector1D<std::complex<double>, chase::distMultiVector::CommunicatorType::column, ARCH>* ChASE_DIST<MatrixType>::zVec_block = nullptr;
template <typename MatrixType>
chase::distMultiVector::DistMultiVector1D<std::complex<float>, chase::distMultiVector::CommunicatorType::column, ARCH>* ChASE_DIST<MatrixType>::cVec_block = nullptr;
template <typename MatrixType>
chase::distMultiVector::DistMultiVector1D<double, chase::distMultiVector::CommunicatorType::column, ARCH>* ChASE_DIST<MatrixType>::dVec_block = nullptr;
template <typename MatrixType>
chase::distMultiVector::DistMultiVector1D<float, chase::distMultiVector::CommunicatorType::column, ARCH>* ChASE_DIST<MatrixType>::sVec_block = nullptr;

// PseudoHermitian matrix static member definitions
template <typename MatrixType>
chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<double>, ARCH>* ChASE_DIST<MatrixType>::zHmat_block_pseudo = nullptr;
template <typename MatrixType>
chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<float>, ARCH>* ChASE_DIST<MatrixType>::cHmat_block_pseudo = nullptr;
template <typename MatrixType>
chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<double>, ARCH>* ChASE_DIST<MatrixType>::zHmat_cyclic_pseudo = nullptr;
template <typename MatrixType>
chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<float>, ARCH>* ChASE_DIST<MatrixType>::cHmat_cyclic_pseudo = nullptr;

// PseudoHermitian solver static member definitions
template <typename MatrixType>
DistSolverType<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<double>, ARCH>> *ChASE_DIST<MatrixType>::zchaseDist_pseudo_block = nullptr;
template <typename MatrixType>
DistSolverType<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<float>, ARCH>> *ChASE_DIST<MatrixType>::cchaseDist_pseudo_block = nullptr;
template <typename MatrixType>
DistSolverType<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<double>, ARCH>> *ChASE_DIST<MatrixType>::zchaseDist_pseudo_cyclic = nullptr;
template <typename MatrixType>
DistSolverType<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<float>, ARCH>> *ChASE_DIST<MatrixType>::cchaseDist_pseudo_cyclic = nullptr;

template <>
int ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<std::complex<double>, ARCH>>::Initialize(int N, int nev, int nex, int mbsize, int nbsize,
                            std::complex<double>* H, int ldh, std::complex<double>* V, chase::Base<std::complex<double>>* ritzv,
                            int dim0, int dim1, char* grid_major, int irsrc,
                            int icsrc, MPI_Comm comm)
{
    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid;

    if (*grid_major == 'R')
    {
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::RowMajor>>(dim0, dim1, comm);
    }else if(*grid_major == 'C')
    {
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(dim0, dim1, comm);
    }else {
        throw std::runtime_error("Invalid grid major type, expected 'C' or 'R'.");
    }

    int *coord = mpi_grid->get_coords();
    int *dim   = mpi_grid->get_dims();

    std::size_t m, n, mblocks, nblocks;
    std::tie(m, mblocks) = chase::numroc(N, mbsize, coord[0], dim[0]);
    std::tie(n, nblocks) = chase::numroc(N, nbsize, coord[1], dim[1]);

    zHmat_cyclic = new chase::distMatrix::BlockCyclicMatrix<std::complex<double>, ARCH>(N, N, m, n, mbsize, nbsize, ldh, H, mpi_grid);    
    zVec_cyclic = new chase::distMultiVector::DistMultiVectorBlockCyclic1D<std::complex<double>, chase::distMultiVector::CommunicatorType::column, ARCH>(N, m, nev + nex, mbsize, m, V, mpi_grid);  

    zchaseDist_cyclic = new DistSolverType<chase::distMatrix::BlockCyclicMatrix<std::complex<double>, ARCH>>(nev, nex, zHmat_cyclic, zVec_cyclic, ritzv);
    
    return 1;    
}

template <>
int ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<std::complex<float>, ARCH>>::Initialize(int N, int nev, int nex, int mbsize, int nbsize,
                            std::complex<float>* H, int ldh, std::complex<float>* V, chase::Base<std::complex<float>>* ritzv,
                            int dim0, int dim1, char* grid_major, int irsrc,
                            int icsrc, MPI_Comm comm)
{
    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid;

    if (*grid_major == 'R')
    {
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::RowMajor>>(dim0, dim1, comm);
    }else if(*grid_major == 'C')
    {
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(dim0, dim1, comm);
    }else {
        throw std::runtime_error("Invalid grid major type, expected 'C' or 'R'.");
    }

    int *coord = mpi_grid->get_coords();
    int *dim   = mpi_grid->get_dims();

    std::size_t m, n, mblocks, nblocks;
    std::tie(m, mblocks) = chase::numroc(N, mbsize, coord[0], dim[0]);
    std::tie(n, nblocks) = chase::numroc(N, nbsize, coord[1], dim[1]);

    cHmat_cyclic = new chase::distMatrix::BlockCyclicMatrix<std::complex<float>, ARCH>(N, N, m, n, mbsize, nbsize, ldh, H, mpi_grid);    
    cVec_cyclic = new chase::distMultiVector::DistMultiVectorBlockCyclic1D<std::complex<float>, chase::distMultiVector::CommunicatorType::column, ARCH>(N, m, nev + nex, mbsize, m, V, mpi_grid);  

    cchaseDist_cyclic = new DistSolverType<chase::distMatrix::BlockCyclicMatrix<std::complex<float>, ARCH>>(nev, nex, cHmat_cyclic, cVec_cyclic, ritzv);
    
    return 1;    
}

template <>
int ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<float, ARCH>>::Initialize(int N, int nev, int nex, int mbsize, int nbsize,
                           float* H, int ldh, float* V, float* ritzv,
                            int dim0, int dim1, char* grid_major, int irsrc,
                            int icsrc, MPI_Comm comm)
{
    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid;

    if (*grid_major == 'R')
    {
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::RowMajor>>(dim0, dim1, comm);
    }else if(*grid_major == 'C')
    {
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(dim0, dim1, comm);
    }else {
        throw std::runtime_error("Invalid grid major type, expected 'C' or 'R'.");
    }

    int *coord = mpi_grid->get_coords();
    int *dim   = mpi_grid->get_dims();

    std::size_t m, n, mblocks, nblocks;
    std::tie(m, mblocks) = chase::numroc(N, mbsize, coord[0], dim[0]);
    std::tie(n, nblocks) = chase::numroc(N, nbsize, coord[1], dim[1]);

    sHmat_cyclic = new chase::distMatrix::BlockCyclicMatrix<float, ARCH>(N, N, m, n, mbsize, nbsize, ldh, H, mpi_grid);    
    sVec_cyclic = new chase::distMultiVector::DistMultiVectorBlockCyclic1D<float, chase::distMultiVector::CommunicatorType::column, ARCH>(N, m, nev + nex, mbsize, m, V, mpi_grid);  

    schaseDist_cyclic = new DistSolverType<chase::distMatrix::BlockCyclicMatrix<float, ARCH>>(nev, nex, sHmat_cyclic, sVec_cyclic, ritzv);
        
    return 1;    
}

template <>
int ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<double, ARCH>>::Initialize(int N, int nev, int nex, int mbsize, int nbsize,
                           double* H, int ldh, double* V, double* ritzv,
                            int dim0, int dim1, char* grid_major, int irsrc,
                            int icsrc, MPI_Comm comm)
{
    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid;

    if (*grid_major == 'R')
    {
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::RowMajor>>(dim0, dim1, comm);
    }else if(*grid_major == 'C')
    {
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(dim0, dim1, comm);
    }else {
        throw std::runtime_error("Invalid grid major type, expected 'C' or 'R'.");
    }

    int *coord = mpi_grid->get_coords();
    int *dim   = mpi_grid->get_dims();

    std::size_t m, n, mblocks, nblocks;
    std::tie(m, mblocks) = chase::numroc(N, mbsize, coord[0], dim[0]);
    std::tie(n, nblocks) = chase::numroc(N, nbsize, coord[1], dim[1]);

    dHmat_cyclic = new chase::distMatrix::BlockCyclicMatrix<double, ARCH>(N, N, m, n, mbsize, nbsize, ldh, H, mpi_grid);    
    dVec_cyclic = new chase::distMultiVector::DistMultiVectorBlockCyclic1D<double, chase::distMultiVector::CommunicatorType::column, ARCH>(N, m, nev + nex, mbsize, m, V, mpi_grid);  
    
    dchaseDist_cyclic = new DistSolverType<chase::distMatrix::BlockCyclicMatrix<double, ARCH>>(nev, nex, dHmat_cyclic, dVec_cyclic, ritzv);

    return 1;    
}

template <>
int ChASE_DIST<chase::distMatrix::BlockBlockMatrix<std::complex<double>, ARCH>>::Initialize(int N, int nev, int nex, int m, int n, std::complex<double>* H, int ldh,
                           std::complex<double>* V, chase::Base<std::complex<double>>* ritzv, int dim0, int dim1,
                           char* grid_major, MPI_Comm comm) {

    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid;

    if (*grid_major == 'R')
    {
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::RowMajor>>(dim0, dim1, comm);
    }else if(*grid_major == 'C')
    {
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(dim0, dim1, comm);
    }else {
        throw std::runtime_error("Invalid grid major type, expected 'C' or 'R'.");
    }

    zHmat_block = new chase::distMatrix::BlockBlockMatrix<std::complex<double>, ARCH>(m, n, ldh, H, mpi_grid);    
    zVec_block = new chase::distMultiVector::DistMultiVector1D<std::complex<double>, chase::distMultiVector::CommunicatorType::column, ARCH>(m, nev + nex, m, V, mpi_grid);  

    zchaseDist_block = new DistSolverType<chase::distMatrix::BlockBlockMatrix<std::complex<double>, ARCH>>(nev, nex, zHmat_block, zVec_block, ritzv);

    return 1;
}

template <>
int ChASE_DIST<chase::distMatrix::BlockBlockMatrix<std::complex<float>, ARCH>>::Initialize(int N, int nev, int nex, int m, int n, std::complex<float>* H, int ldh,
                           std::complex<float>* V, chase::Base<std::complex<float>>* ritzv, int dim0, int dim1,
                           char* grid_major, MPI_Comm comm) {

    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid;

    if (*grid_major == 'R')
    {
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::RowMajor>>(dim0, dim1, comm);
    }else if(*grid_major == 'C')
    {
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(dim0, dim1, comm);
    }else {
        throw std::runtime_error("Invalid grid major type, expected 'C' or 'R'.");
    }

    cHmat_block = new chase::distMatrix::BlockBlockMatrix<std::complex<float>, ARCH>(m, n, ldh, H, mpi_grid);    
    cVec_block = new chase::distMultiVector::DistMultiVector1D<std::complex<float>, chase::distMultiVector::CommunicatorType::column, ARCH>(m, nev + nex, m, V, mpi_grid);  

    cchaseDist_block = new DistSolverType<chase::distMatrix::BlockBlockMatrix<std::complex<float>, ARCH>>(nev, nex, cHmat_block, cVec_block, ritzv);

    return 1;
}

template <>
int ChASE_DIST<chase::distMatrix::BlockBlockMatrix<float, ARCH>>::Initialize(int N, int nev, int nex, int m, int n, float* H, int ldh,
                           float* V, chase::Base<float>* ritzv, int dim0, int dim1,
                           char* grid_major, MPI_Comm comm) {

    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid;

    if (*grid_major == 'R')
    {
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::RowMajor>>(dim0, dim1, comm);
    }else if(*grid_major == 'C')
    {
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(dim0, dim1, comm);
    }else {
        throw std::runtime_error("Invalid grid major type, expected 'C' or 'R'.");
    }

    sHmat_block = new chase::distMatrix::BlockBlockMatrix<float, ARCH>(m, n, ldh, H, mpi_grid);    
    sVec_block = new chase::distMultiVector::DistMultiVector1D<float, chase::distMultiVector::CommunicatorType::column, ARCH>(m, nev + nex, m, V, mpi_grid);  

    schaseDist_block = new DistSolverType<chase::distMatrix::BlockBlockMatrix<float, ARCH>>(nev, nex, sHmat_block, sVec_block, ritzv);
      
    return 1;
}

template <>
int ChASE_DIST<chase::distMatrix::BlockBlockMatrix<double, ARCH>>::Initialize(int N, int nev, int nex, int m, int n, double* H, int ldh,
                           double* V, chase::Base<double>* ritzv, int dim0, int dim1,
                           char* grid_major, MPI_Comm comm) {

    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid;

    if (*grid_major == 'R')
    {
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::RowMajor>>(dim0, dim1, comm);
    }else if(*grid_major == 'C')
    {
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(dim0, dim1, comm);
    }else {
        throw std::runtime_error("Invalid grid major type, expected 'C' or 'R'.");
    }

    dHmat_block = new chase::distMatrix::BlockBlockMatrix<double, ARCH>(m, n, ldh, H, mpi_grid);    
    dVec_block = new chase::distMultiVector::DistMultiVector1D<double, chase::distMultiVector::CommunicatorType::column, ARCH>(m, nev + nex, m, V, mpi_grid);  

    dchaseDist_block = new DistSolverType<chase::distMatrix::BlockBlockMatrix<double, ARCH>>(nev, nex, dHmat_block, dVec_block, ritzv);

    return 1;
}

// Initialize specializations for PseudoHermitianBlockBlockMatrix
// Store in separate PseudoHermitian-specific pointers to distinguish from regular matrices
template <>
int ChASE_DIST<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<double>, ARCH>>::Initialize(int N, int nev, int nex, int m, int n, std::complex<double>* H, int ldh,
                           std::complex<double>* V, chase::Base<std::complex<double>>* ritzv, int dim0, int dim1,
                           char* grid_major, MPI_Comm comm) {

    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid;

    if (*grid_major == 'R')
    {
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::RowMajor>>(dim0, dim1, comm);
    }else if(*grid_major == 'C')
    {
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(dim0, dim1, comm);
    }else {
        throw std::runtime_error("Invalid grid major type, expected 'C' or 'R'.");
    }

    // Store in PseudoHermitian-specific pointer
    zHmat_block_pseudo = new chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<double>, ARCH>(m, n, ldh, H, mpi_grid);  
    zVec_block = new chase::distMultiVector::DistMultiVector1D<std::complex<double>, chase::distMultiVector::CommunicatorType::column, ARCH>(m, nev + nex, m, V, mpi_grid);  

    // Create solver with PseudoHermitianBlockBlockMatrix type - solver will detect PseudoHermitian at compile time
    zchaseDist_pseudo_block = new DistSolverType<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<double>, ARCH>>(nev, nex, 
        zHmat_block_pseudo, zVec_block, ritzv);

    return 1;
}

template <>
int ChASE_DIST<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<float>, ARCH>>::Initialize(int N, int nev, int nex, int m, int n, std::complex<float>* H, int ldh,
                           std::complex<float>* V, chase::Base<std::complex<float>>* ritzv, int dim0, int dim1,
                           char* grid_major, MPI_Comm comm) {

    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid;

    if (*grid_major == 'R')
    {
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::RowMajor>>(dim0, dim1, comm);
    }else if(*grid_major == 'C')
    {
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(dim0, dim1, comm);
    }else {
        throw std::runtime_error("Invalid grid major type, expected 'C' or 'R'.");
    }
    // Store in PseudoHermitian-specific pointer
    cHmat_block_pseudo = new chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<float>, ARCH>(m, n, ldh, H, mpi_grid);
    cVec_block = new chase::distMultiVector::DistMultiVector1D<std::complex<float>, chase::distMultiVector::CommunicatorType::column, ARCH>(m, nev + nex, m, V, mpi_grid);  

    // Create solver with PseudoHermitianBlockBlockMatrix type - solver will detect PseudoHermitian at compile time
    cchaseDist_pseudo_block = new DistSolverType<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<float>, ARCH>>(nev, nex, 
        cHmat_block_pseudo, cVec_block, ritzv);

    return 1;
}

// Initialize specializations for PseudoHermitianBlockCyclicMatrix
// Store in separate PseudoHermitian-specific pointers to distinguish from regular matrices
template <>
int ChASE_DIST<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<double>, ARCH>>::Initialize(int N, int nev, int nex, int mbsize, int nbsize,
                            std::complex<double>* H, int ldh, std::complex<double>* V, chase::Base<std::complex<double>>* ritzv,
                            int dim0, int dim1, char* grid_major, int irsrc,
                            int icsrc, MPI_Comm comm)
{
    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid;

    if (*grid_major == 'R')
    {
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::RowMajor>>(dim0, dim1, comm);
    }else if(*grid_major == 'C')
    {
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(dim0, dim1, comm);
    }else {
        throw std::runtime_error("Invalid grid major type, expected 'C' or 'R'.");
    }

    int *coord = mpi_grid->get_coords();
    int *dim   = mpi_grid->get_dims();

    std::size_t m, n, mblocks, nblocks;
    std::tie(m, mblocks) = chase::numroc(N, mbsize, coord[0], dim[0]);
    std::tie(n, nblocks) = chase::numroc(N, nbsize, coord[1], dim[1]);

    // Store in PseudoHermitian-specific pointer
    zHmat_cyclic_pseudo = new chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<double>, ARCH>(N, N, m, n, mbsize, nbsize, ldh, H, mpi_grid);
    zVec_cyclic = new chase::distMultiVector::DistMultiVectorBlockCyclic1D<std::complex<double>, chase::distMultiVector::CommunicatorType::column, ARCH>(N, m, nev + nex, mbsize, m, V, mpi_grid);  

    // Create solver with PseudoHermitianBlockCyclicMatrix type - solver will detect PseudoHermitian at compile time
    zchaseDist_pseudo_cyclic = new DistSolverType<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<double>, ARCH>>(nev, nex, 
        zHmat_cyclic_pseudo, zVec_cyclic, ritzv);
    
    return 1;    
}

template <>
int ChASE_DIST<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<float>, ARCH>>::Initialize(int N, int nev, int nex, int mbsize, int nbsize,
                            std::complex<float>* H, int ldh, std::complex<float>* V, chase::Base<std::complex<float>>* ritzv,
                            int dim0, int dim1, char* grid_major, int irsrc,
                            int icsrc, MPI_Comm comm)
{
    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid;

    if (*grid_major == 'R')
    {
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::RowMajor>>(dim0, dim1, comm);
    }else if(*grid_major == 'C')
    {
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(dim0, dim1, comm);
    }else {
        throw std::runtime_error("Invalid grid major type, expected 'C' or 'R'.");
    }

    int *coord = mpi_grid->get_coords();
    int *dim   = mpi_grid->get_dims();

    std::size_t m, n, mblocks, nblocks;
    std::tie(m, mblocks) = chase::numroc(N, mbsize, coord[0], dim[0]);
    std::tie(n, nblocks) = chase::numroc(N, nbsize, coord[1], dim[1]);

    // Store in PseudoHermitian-specific pointer
    cHmat_cyclic_pseudo = new chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<float>, ARCH>(N, N, m, n, mbsize, nbsize, ldh, H, mpi_grid);
    cVec_cyclic = new chase::distMultiVector::DistMultiVectorBlockCyclic1D<std::complex<float>, chase::distMultiVector::CommunicatorType::column, ARCH>(N, m, nev + nex, mbsize, m, V, mpi_grid);  

    // Create solver with PseudoHermitianBlockCyclicMatrix type - solver will detect PseudoHermitian at compile time
    cchaseDist_pseudo_cyclic = new DistSolverType<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<float>, ARCH>>(nev, nex, 
        cHmat_cyclic_pseudo, cVec_cyclic, ritzv);
    
    return 1;    
}

// getChase specializations for BlockCyclicMatrix
template<>
DistSolverType<chase::distMatrix::BlockCyclicMatrix<std::complex<double>, ARCH>> *ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<std::complex<double>, ARCH>>::getChase()
{
    return zchaseDist_cyclic;
}

template<>
DistSolverType<chase::distMatrix::BlockCyclicMatrix<std::complex<float>, ARCH>> *ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<std::complex<float>, ARCH>>::getChase()
{
    return cchaseDist_cyclic;
}

template<>
DistSolverType<chase::distMatrix::BlockCyclicMatrix<double, ARCH>> *ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<double, ARCH>>::getChase()
{
    return dchaseDist_cyclic;
}

template<>
DistSolverType<chase::distMatrix::BlockCyclicMatrix<float, ARCH>> *ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<float, ARCH>>::getChase()
{
    return schaseDist_cyclic;
}

// getChase specializations for BlockBlockMatrix
template<>
DistSolverType<chase::distMatrix::BlockBlockMatrix<std::complex<double>, ARCH>> *ChASE_DIST<chase::distMatrix::BlockBlockMatrix<std::complex<double>, ARCH>>::getChase()
{
    return zchaseDist_block;
}

template<>
DistSolverType<chase::distMatrix::BlockBlockMatrix<std::complex<float>, ARCH>> *ChASE_DIST<chase::distMatrix::BlockBlockMatrix<std::complex<float>, ARCH>>::getChase()
{
    return cchaseDist_block;
}

template<>
DistSolverType<chase::distMatrix::BlockBlockMatrix<double, ARCH>> *ChASE_DIST<chase::distMatrix::BlockBlockMatrix<double, ARCH>>::getChase()
{
    return dchaseDist_block;
}

template<>
DistSolverType<chase::distMatrix::BlockBlockMatrix<float, ARCH>> *ChASE_DIST<chase::distMatrix::BlockBlockMatrix<float, ARCH>>::getChase()
{
    return schaseDist_block;
}

// getChase specializations for PseudoHermitianBlockBlockMatrix
template<>
DistSolverType<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<double>, ARCH>> *ChASE_DIST<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<double>, ARCH>>::getChase()
{
    return zchaseDist_pseudo_block;
}

template<>
DistSolverType<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<float>, ARCH>> *ChASE_DIST<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<float>, ARCH>>::getChase()
{
    return cchaseDist_pseudo_block;
}

// getChase specializations for PseudoHermitianBlockCyclicMatrix
template<>
DistSolverType<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<double>, ARCH>> *ChASE_DIST<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<double>, ARCH>>::getChase()
{
    return zchaseDist_pseudo_cyclic;
}

template<>
DistSolverType<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<float>, ARCH>> *ChASE_DIST<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<float>, ARCH>>::getChase()
{
    return cchaseDist_pseudo_cyclic;
}

// Finalize specializations for BlockCyclicMatrix
template<>
int ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<std::complex<double>, ARCH>>::Finalize()
{
    delete zchaseDist_cyclic;
    zchaseDist_cyclic = nullptr;
    delete zHmat_cyclic;
    zHmat_cyclic = nullptr;
    delete zVec_cyclic;
    zVec_cyclic = nullptr;    
    return 0;
}

template<>
int ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<std::complex<float>, ARCH>>::Finalize()
{
    delete cchaseDist_cyclic;
    cchaseDist_cyclic = nullptr;
    delete cHmat_cyclic;
    cHmat_cyclic = nullptr;
    delete cVec_cyclic;
    cVec_cyclic = nullptr;    
    return 0;
}

template<>
int ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<double, ARCH>>::Finalize()
{
    delete dchaseDist_cyclic;
    dchaseDist_cyclic = nullptr;
    delete dHmat_cyclic;
    dHmat_cyclic = nullptr;
    delete dVec_cyclic;
    dVec_cyclic = nullptr;    
    return 0;
}

template<>
int ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<float, ARCH>>::Finalize()
{
    delete schaseDist_cyclic;
    schaseDist_cyclic = nullptr;
    delete sHmat_cyclic;
    sHmat_cyclic = nullptr;
    delete sVec_cyclic;
    sVec_cyclic = nullptr;    
    return 0;
}

// Finalize specializations for BlockBlockMatrix
template<>
int ChASE_DIST<chase::distMatrix::BlockBlockMatrix<std::complex<double>, ARCH>>::Finalize()
{
    delete zchaseDist_block;
    zchaseDist_block = nullptr;
    delete zHmat_block;
    zHmat_block = nullptr;
    delete zVec_block;
    zVec_block = nullptr;    
    return 0;
}

template<>
int ChASE_DIST<chase::distMatrix::BlockBlockMatrix<std::complex<float>, ARCH>>::Finalize()
{
    delete cchaseDist_block;
    cchaseDist_block = nullptr;
    delete cHmat_block;
    cHmat_block = nullptr;
    delete cVec_block;
    cVec_block = nullptr;    
    return 0;
}

template<>
int ChASE_DIST<chase::distMatrix::BlockBlockMatrix<double, ARCH>>::Finalize()
{
    delete dchaseDist_block;
    dchaseDist_block = nullptr;
    delete dHmat_block;
    dHmat_block = nullptr;
    delete dVec_block;
    dVec_block = nullptr;    
    return 0;
}

template<>
int ChASE_DIST<chase::distMatrix::BlockBlockMatrix<float, ARCH>>::Finalize()
{
    delete schaseDist_block;
    schaseDist_block = nullptr;
    delete sHmat_block;
    sHmat_block = nullptr;
    delete sVec_block;
    sVec_block = nullptr;    
    return 0;
}

// Finalize specializations for PseudoHermitianBlockBlockMatrix
template<>
int ChASE_DIST<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<double>, ARCH>>::Finalize()
{
    delete zchaseDist_pseudo_block;
    zchaseDist_pseudo_block = nullptr;
    delete zHmat_block_pseudo;
    zHmat_block_pseudo = nullptr;
    delete zVec_block;
    zVec_block = nullptr;    
    return 0;
}

template<>
int ChASE_DIST<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<float>, ARCH>>::Finalize()
{
    delete cchaseDist_pseudo_block;
    cchaseDist_pseudo_block = nullptr;
    delete cHmat_block_pseudo;
    cHmat_block_pseudo = nullptr;
    delete cVec_block;
    cVec_block = nullptr;    
    return 0;
}

// Finalize specializations for PseudoHermitianBlockCyclicMatrix
template<>
int ChASE_DIST<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<double>, ARCH>>::Finalize()
{
    delete zchaseDist_pseudo_cyclic;
    zchaseDist_pseudo_cyclic = nullptr;
    delete zHmat_cyclic_pseudo;
    zHmat_cyclic_pseudo = nullptr;
    delete zVec_cyclic;
    zVec_cyclic = nullptr;    
    return 0;
}

template<>
int ChASE_DIST<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<float>, ARCH>>::Finalize()
{
    delete cchaseDist_pseudo_cyclic;
    cchaseDist_pseudo_cyclic = nullptr;
    delete cHmat_cyclic_pseudo;
    cHmat_cyclic_pseudo = nullptr;
    delete cVec_cyclic;
    cVec_cyclic = nullptr;    
    return 0;
}

template<typename MatrixType>
void ChASE_DIST<MatrixType>::readHam(const std::string& filename)
{
    auto single = ChASE_DIST<MatrixType>::getChase();
    single->loadProblemFromFile(filename);
}

template<typename MatrixType>
void ChASE_DIST<MatrixType>::WrteHam(const std::string& filename)
{
    auto single = ChASE_DIST<MatrixType>::getChase();
    single->saveProblemToFile(filename);
}

template<typename MatrixType>
void ChASE_DIST<MatrixType>::Solve(int* deg, chase::Base<typename MatrixType::value_type>* tol, char* mode, char* opt, char *qr)
{
    using T = typename MatrixType::value_type;
    auto single = ChASE_DIST<MatrixType>::getChase();
    
    chase::ChaseConfig<T>& config = single->GetConfig();
    config.SetTol(*tol);
    config.SetDeg(*deg);
    config.SetOpt(*opt == 'S');
    config.SetApprox(*mode == 'A');
    config.SetCholQR(*qr == 'C');
    config.EnableSymCheck(false);
    config.SetDegExtra(0);
    
    // Set lowerb parameter to 0.97 for PseudoHermitian matrices
    // Check if MatrixType is a PseudoHermitian type by checking the hermitian_type trait
    if constexpr (std::is_same_v<typename MatrixType::hermitian_type, chase::matrix::PseudoHermitian>) {
        config.SetDecayingRate(0.975);
        config.SetLanczosIter(25);
        config.SetNumLanczos(100);
        config.SetMaxDeg(36);
        config.SetClusterAwareDegrees(true);
    }
#ifdef CHASE_OUTPUT
    if(single->get_rank() == 0)
    {
        std::cout << config << std::endl;
    }  
#endif

    chase::PerformanceDecoratorChase<T> performanceDecorator(single);

    chase::Solve(&performanceDecorator);

    chase::Base<T>* ritzv = single->GetRitzv();
    chase::Base<T>* resid = single->GetResid();

#ifdef CHASE_OUTPUT
    if(single->get_rank() == 0)
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

extern "C" {

void dchase_init_(int* N, int* nev, int* nex, double* H, int *ldh, double* V,
                    double* ritzv, int* init)
{
    *init = ChASE_SEQ::Initialize<double>(*N, *nev, *nex, H, *ldh, V, ritzv);
}
void schase_init_(int* N, int* nev, int* nex, float* H,  int *ldh, float* V,
                    float* ritzv, int* init)
{
    *init = ChASE_SEQ::Initialize<float>(*N, *nev, *nex, H, *ldh, V, ritzv);
}

void cchase_init_(int* N, int* nev, int* nex, float _Complex* H, int *ldh,
                    float _Complex* V, float* ritzv, int* init)
{
    *init = ChASE_SEQ::Initialize<std::complex<float>>(
        *N, *nev, *nex, reinterpret_cast<std::complex<float>*>(H), *ldh,
        reinterpret_cast<std::complex<float>*>(V), ritzv);
}

void zchase_init_(int* N, int* nev, int* nex, double _Complex* H, int *ldh,
                    double _Complex* V, double* ritzv, int* init)
{
    *init = ChASE_SEQ::Initialize<std::complex<double>>(
        *N, *nev, *nex, reinterpret_cast<std::complex<double>*>(H), *ldh,
        reinterpret_cast<std::complex<double>*>(V), ritzv);    
}

void dchase_finalize_(int* flag) { *flag = ChASE_SEQ::Finalize<double>(); }
void schase_finalize_(int* flag) { *flag = ChASE_SEQ::Finalize<float>(); }
void cchase_finalize_(int* flag)
{
    *flag = ChASE_SEQ::Finalize<std::complex<float>>();
}
void zchase_finalize_(int* flag)
{
    *flag = ChASE_SEQ::Finalize<std::complex<double>>();
}
    
void dchase_(int* deg, double* tol, char* mode, char* opt, char *qr)
{
    ChASE_SEQ_Solve<double>(deg, tol, mode, opt, qr);
}   
void schase_(int* deg, float* tol, char* mode, char* opt, char *qr)
{
    ChASE_SEQ_Solve<float>(deg, tol, mode, opt, qr);
}   
void zchase_(int* deg, double* tol, char* mode, char* opt, char *qr)
{
    ChASE_SEQ_Solve<std::complex<double>>(deg, tol, mode, opt, qr);
}    
void cchase_(int* deg, float* tol, char* mode, char* opt, char *qr)
{
    ChASE_SEQ_Solve<std::complex<float>>(deg, tol, mode, opt, qr);
}

void pdchase_init_blockcyclic_(int* N, int* nev, int* nex, int* mbsize,
                                int* nbsize, double* H, int* ldh, double* V,
                                double* ritzv, int* dim0, int* dim1,
                                char* grid_major, int* irsrc, int* icsrc,
                                MPI_Comm* comm, int* init)
{
    *init =  ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<double, ARCH>>::Initialize(*N, *nev, *nex, *mbsize, *nbsize, H,
                                    *ldh, V, ritzv, *dim0, *dim1, grid_major,
                                    *irsrc, *icsrc, *comm);    
}                                

void pdchase_init_blockcyclic_f_(int* N, int* nev, int* nex, int* mbsize,
                                    int* nbsize, double* H, int* ldh,
                                    double* V, double* ritzv, int* dim0,
                                    int* dim1, char* grid_major, int* irsrc,
                                    int* icsrc, MPI_Fint* fcomm, int* init)
{
    MPI_Comm comm = MPI_Comm_f2c(*fcomm);
    *init = ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<double, ARCH>>::Initialize(*N, *nev, *nex, *mbsize, *nbsize, H,
                                    *ldh, V, ritzv, *dim0, *dim1,
                                    grid_major, *irsrc, *icsrc, comm);
}

void pschase_init_blockcyclic_(int* N, int* nev, int* nex, int* mbsize,
                                int* nbsize, float* H, int* ldh, float* V,
                                float* ritzv, int* dim0, int* dim1,
                                char* grid_major, int* irsrc, int* icsrc,
                                MPI_Comm* comm, int* init)
{
    *init =  ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<float, ARCH>>::Initialize(*N, *nev, *nex, *mbsize, *nbsize, H,
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
    *init = ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<float, ARCH>>::Initialize(*N, *nev, *nex, *mbsize, *nbsize, H,
                                    *ldh, V, ritzv, *dim0, *dim1, grid_major,
                                    *irsrc, *icsrc, comm);
}


void pcchase_init_blockcyclic_(int* N, int* nev, int* nex, int* mbsize,
                                int* nbsize, float _Complex* H, int* ldh,
                                float _Complex* V, float* ritzv, int* dim0,
                                int* dim1, char* grid_major, int* irsrc,
                                int* icsrc, MPI_Comm* comm, int* init)
{
    *init = ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<std::complex<float>, ARCH>>::Initialize(
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
    *init = ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<std::complex<float>, ARCH>>::Initialize(
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
    *init = ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<std::complex<double>, ARCH>>::Initialize(
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
    *init = ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<std::complex<double>, ARCH>>::Initialize(
        *N, *nev, *nex, *mbsize, *nbsize,
        reinterpret_cast<std::complex<double>*>(H), *ldh,
        reinterpret_cast<std::complex<double>*>(V), ritzv, *dim0, *dim1,
        grid_major, *irsrc, *icsrc, comm);
}

void pdchase_init_(int* N, int* nev, int* nex, int* m, int* n, double* H,
                    int* ldh, double* V, double* ritzv, int* dim0, int* dim1,
                    char* grid_major, MPI_Comm* comm, int* init)
{
    *init = ChASE_DIST<chase::distMatrix::BlockBlockMatrix<double, ARCH>>::Initialize(*N, *nev, *nex, *m, *n, H, *ldh, V,
                                    ritzv, *dim0, *dim1, grid_major, *comm);
}

void pdchase_init_f_(int* N, int* nev, int* nex, int* m, int* n, double* H,
                        int* ldh, double* V, double* ritzv, int* dim0,
                        int* dim1, char* grid_major, MPI_Fint* fcomm,
                        int* init)
{
    MPI_Comm comm = MPI_Comm_f2c(*fcomm);
    *init = ChASE_DIST<chase::distMatrix::BlockBlockMatrix<double, ARCH>>::Initialize(*N, *nev, *nex, *m, *n, H, *ldh, V,
                                    ritzv, *dim0, *dim1, grid_major, comm);
}

void pschase_init_(int* N, int* nev, int* nex, int* m, int* n, float* H,
                    int* ldh, float* V, float* ritzv, int* dim0, int* dim1,
                    char* grid_major, MPI_Comm* comm, int* init)
{
    *init = ChASE_DIST<chase::distMatrix::BlockBlockMatrix<float, ARCH>>::Initialize(*N, *nev, *nex, *m, *n, H, *ldh, V,
                                    ritzv, *dim0, *dim1, grid_major, *comm);
}                    

void pschase_init_f_(int* N, int* nev, int* nex, int* m, int* n, float* H,
                    int* ldh, float* V, float* ritzv, int* dim0, int* dim1,
                    char* grid_major, MPI_Fint* fcomm, int* init)
{
    MPI_Comm comm = MPI_Comm_f2c(*fcomm);
    *init = ChASE_DIST<chase::distMatrix::BlockBlockMatrix<float, ARCH>>::Initialize(*N, *nev, *nex, *m, *n, H, *ldh, V,
                                    ritzv, *dim0, *dim1, grid_major, comm);   
}


void pzchase_init_(int* N, int* nev, int* nex, int* m, int* n,
                    double _Complex* H, int* ldh, double _Complex* V,
                    double* ritzv, int* dim0, int* dim1, char* grid_major,
                    MPI_Comm* comm, int* init)
{
    *init = ChASE_DIST<chase::distMatrix::BlockBlockMatrix<std::complex<double>, ARCH>>::Initialize(*N, *nev, *nex, *m, *n, 
                                                         reinterpret_cast<std::complex<double>*>(H), *ldh, 
                                                         reinterpret_cast<std::complex<double>*>(V),
                                                         ritzv, *dim0, *dim1, grid_major, *comm);
}                    

void pzchase_init_f_(int* N, int* nev, int* nex, int* m, int* n,
                        double _Complex* H, int* ldh, double _Complex* V,
                        double* ritzv, int* dim0, int* dim1, char* grid_major,
                        MPI_Fint* fcomm, int* init)
{
    MPI_Comm comm = MPI_Comm_f2c(*fcomm);
    *init = ChASE_DIST<chase::distMatrix::BlockBlockMatrix<std::complex<double>, ARCH>>::Initialize(
        *N, *nev, *nex, *m, *n, reinterpret_cast<std::complex<double>*>(H),
        *ldh, reinterpret_cast<std::complex<double>*>(V), ritzv, *dim0,
        *dim1, grid_major, comm);
}

void pcchase_init_(int* N, int* nev, int* nex, int* m, int* n,
                    float _Complex* H, int* ldh, float _Complex* V,
                    float* ritzv, int* dim0, int* dim1, char* grid_major,
                    MPI_Comm* comm, int* init)
{
    *init = ChASE_DIST<chase::distMatrix::BlockBlockMatrix<std::complex<float>, ARCH>>::Initialize(*N, *nev, *nex, *m, *n, 
                                                         reinterpret_cast<std::complex<float>*>(H), *ldh, 
                                                         reinterpret_cast<std::complex<float>*>(V),
                                                         ritzv, *dim0, *dim1, grid_major, *comm);
}                     

void pcchase_init_f_(int* N, int* nev, int* nex, int* m, int* n,
                        float _Complex* H, int* ldh, float _Complex* V,
                        float* ritzv, int* dim0, int* dim1, char* grid_major,
                        MPI_Fint* fcomm, int* init)
{
    MPI_Comm comm = MPI_Comm_f2c(*fcomm);
    *init = ChASE_DIST<chase::distMatrix::BlockBlockMatrix<std::complex<float>, ARCH>>::Initialize(
        *N, *nev, *nex, *m, *n, reinterpret_cast<std::complex<float>*>(H),
        *ldh, reinterpret_cast<std::complex<float>*>(V), ritzv, *dim0,
        *dim1, grid_major, comm);
}

// PseudoHermitian initialization functions (BlockBlockMatrix)
void pzchase_init_pseudo_(int* N, int* nev, int* nex, int* m, int* n,
                    double _Complex* H, int* ldh, double _Complex* V,
                    double* ritzv, int* dim0, int* dim1, char* grid_major,
                    MPI_Comm* comm, int* init)
{
    *init = ChASE_DIST<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<double>, ARCH>>::Initialize(*N, *nev, *nex, *m, *n, 
                                                         reinterpret_cast<std::complex<double>*>(H), *ldh, 
                                                         reinterpret_cast<std::complex<double>*>(V),
                                                         ritzv, *dim0, *dim1, grid_major, *comm);
}                     

void pzchase_init_pseudo_f_(int* N, int* nev, int* nex, int* m, int* n,
                        double _Complex* H, int* ldh, double _Complex* V,
                        double* ritzv, int* dim0, int* dim1, char* grid_major,
                        MPI_Fint* fcomm, int* init)
{
    MPI_Comm comm = MPI_Comm_f2c(*fcomm);
    *init = ChASE_DIST<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<double>, ARCH>>::Initialize(
        *N, *nev, *nex, *m, *n, reinterpret_cast<std::complex<double>*>(H),
        *ldh, reinterpret_cast<std::complex<double>*>(V), ritzv, *dim0,
        *dim1, grid_major, comm);
}

void pcchase_init_pseudo_(int* N, int* nev, int* nex, int* m, int* n,
                    float _Complex* H, int* ldh, float _Complex* V,
                    float* ritzv, int* dim0, int* dim1, char* grid_major,
                    MPI_Comm* comm, int* init)
{
    *init = ChASE_DIST<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<float>, ARCH>>::Initialize(*N, *nev, *nex, *m, *n, 
                                                         reinterpret_cast<std::complex<float>*>(H), *ldh, 
                                                         reinterpret_cast<std::complex<float>*>(V),
                                                         ritzv, *dim0, *dim1, grid_major, *comm);
}                     

void pcchase_init_pseudo_f_(int* N, int* nev, int* nex, int* m, int* n,
                        float _Complex* H, int* ldh, float _Complex* V,
                        float* ritzv, int* dim0, int* dim1, char* grid_major,
                        MPI_Fint* fcomm, int* init)
{
    MPI_Comm comm = MPI_Comm_f2c(*fcomm);
    *init = ChASE_DIST<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<float>, ARCH>>::Initialize(
        *N, *nev, *nex, *m, *n, reinterpret_cast<std::complex<float>*>(H),
        *ldh, reinterpret_cast<std::complex<float>*>(V), ritzv, *dim0,
        *dim1, grid_major, comm);
}

// PseudoHermitian initialization functions (BlockCyclicMatrix)
void pzchase_init_pseudo_blockcyclic_(int* N, int* nev, int* nex, int* mbsize,
                                int* nbsize, double _Complex* H, int* ldh,
                                double _Complex* V, double* ritzv, int* dim0,
                                int* dim1, char* grid_major, int* irsrc,
                                int* icsrc, MPI_Comm* comm, int* init)
{
    *init = ChASE_DIST<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<double>, ARCH>>::Initialize(
        *N, *nev, *nex, *mbsize, *nbsize,
        reinterpret_cast<std::complex<double>*>(H), *ldh,
        reinterpret_cast<std::complex<double>*>(V), ritzv, *dim0, *dim1,
        grid_major, *irsrc, *icsrc, *comm);
}

void pzchase_init_pseudo_blockcyclic_f_(int* N, int* nev, int* nex, int* mbsize,
                                    int* nbsize, double _Complex* H, int* ldh,
                                    double _Complex* V, double* ritzv,
                                    int* dim0, int* dim1, char* grid_major,
                                    int* irsrc, int* icsrc, MPI_Fint* fcomm,
                                    int* init)
{
    MPI_Comm comm = MPI_Comm_f2c(*fcomm);
    *init = ChASE_DIST<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<double>, ARCH>>::Initialize(
        *N, *nev, *nex, *mbsize, *nbsize,
        reinterpret_cast<std::complex<double>*>(H), *ldh,
        reinterpret_cast<std::complex<double>*>(V), ritzv, *dim0, *dim1,
        grid_major, *irsrc, *icsrc, comm);
}

void pcchase_init_pseudo_blockcyclic_(int* N, int* nev, int* nex, int* mbsize,
                                int* nbsize, float _Complex* H, int* ldh,
                                float _Complex* V, float* ritzv, int* dim0,
                                int* dim1, char* grid_major, int* irsrc,
                                int* icsrc, MPI_Comm* comm, int* init)
{
    *init = ChASE_DIST<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<float>, ARCH>>::Initialize(
        *N, *nev, *nex, *mbsize, *nbsize,
        reinterpret_cast<std::complex<float>*>(H), *ldh,
        reinterpret_cast<std::complex<float>*>(V), ritzv, *dim0, *dim1,
        grid_major, *irsrc, *icsrc, *comm);

}

void pcchase_init_pseudo_blockcyclic_f_(int* N, int* nev, int* nex, int* mbsize,
                                    int* nbsize, float _Complex* H, int* ldh,
                                    float _Complex* V, float* ritzv, int* dim0,
                                    int* dim1, char* grid_major, int* irsrc,
                                    int* icsrc, MPI_Fint* fcomm, int* init)
{
    MPI_Comm comm = MPI_Comm_f2c(*fcomm);
    *init = ChASE_DIST<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<float>, ARCH>>::Initialize(
        *N, *nev, *nex, *mbsize, *nbsize,
        reinterpret_cast<std::complex<float>*>(H), *ldh,
        reinterpret_cast<std::complex<float>*>(V), ritzv, *dim0, *dim1,
        grid_major, *irsrc, *icsrc, comm);
}

// BlockCyclicMatrix functions (with _blockcyclic suffix)
// Note: solve, write, and read functions are unified below - no separate _blockcyclic versions needed

// Unified finalize functions - work for both BlockBlockMatrix and BlockCyclicMatrix
// Finalizes whichever type was initialized (deleting nullptr is safe in C++)
void pdchase_finalize_(int* flag) 
{ 
    ChASE_DIST<chase::distMatrix::BlockBlockMatrix<double, ARCH>>::Finalize();
    ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<double, ARCH>>::Finalize();
    *flag = 0;
}
void pschase_finalize_(int* flag) 
{ 
    ChASE_DIST<chase::distMatrix::BlockBlockMatrix<float, ARCH>>::Finalize();
    ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<float, ARCH>>::Finalize();
    *flag = 0;
}
void pcchase_finalize_(int* flag) 
{ 
    // Finalize pseudo-Hermitian types first (inheritance allows unified interface)
    ChASE_DIST<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<float>, ARCH>>::Finalize();
    ChASE_DIST<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<float>, ARCH>>::Finalize();
    // Then finalize regular types (deleting nullptr is safe)
    ChASE_DIST<chase::distMatrix::BlockBlockMatrix<std::complex<float>, ARCH>>::Finalize();
    ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<std::complex<float>, ARCH>>::Finalize();
    *flag = 0;
}
void pzchase_finalize_(int* flag) 
{ 
    // Finalize pseudo-Hermitian types first (inheritance allows unified interface)
    ChASE_DIST<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<double>, ARCH>>::Finalize();
    ChASE_DIST<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<double>, ARCH>>::Finalize();
    // Then finalize regular types (deleting nullptr is safe)
    ChASE_DIST<chase::distMatrix::BlockBlockMatrix<std::complex<double>, ARCH>>::Finalize();
    ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<std::complex<double>, ARCH>>::Finalize();
    *flag = 0;
}

// Unified solve functions - work for both BlockBlockMatrix and BlockCyclicMatrix
// Checks which type was initialized and calls the appropriate solve function
void pdchase_(int* deg, double* tol, char* mode, char* opt, char *qr)
{
    // Check which type was initialized by checking the static members
    // Access through any template instantiation since static members are shared
    if (ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<double, ARCH>>::dchaseDist_cyclic != nullptr) {
        ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<double, ARCH>>::Solve(deg, tol, mode, opt, qr);
    } else if (ChASE_DIST<chase::distMatrix::BlockBlockMatrix<double, ARCH>>::dchaseDist_block != nullptr) {
        ChASE_DIST<chase::distMatrix::BlockBlockMatrix<double, ARCH>>::Solve(deg, tol, mode, opt, qr);
    }
}
void pschase_(int* deg, float* tol, char* mode, char* opt, char *qr)
{
    if (ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<float, ARCH>>::schaseDist_cyclic != nullptr) {
        ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<float, ARCH>>::Solve(deg, tol, mode, opt, qr);
    } else if (ChASE_DIST<chase::distMatrix::BlockBlockMatrix<float, ARCH>>::schaseDist_block != nullptr) {
        ChASE_DIST<chase::distMatrix::BlockBlockMatrix<float, ARCH>>::Solve(deg, tol, mode, opt, qr);
    }
}
void pzchase_(int* deg, double* tol, char* mode, char* opt, char *qr)
{
    // Check pseudo-Hermitian solvers first (inheritance allows unified interface)
    if (ChASE_DIST<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<double>, ARCH>>::zchaseDist_pseudo_cyclic != nullptr) {
        ChASE_DIST<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<double>, ARCH>>::Solve(deg, tol, mode, opt, qr);
    } else if (ChASE_DIST<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<double>, ARCH>>::zchaseDist_pseudo_block != nullptr) {
        ChASE_DIST<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<double>, ARCH>>::Solve(deg, tol, mode, opt, qr);
    } else if (ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<std::complex<double>, ARCH>>::zchaseDist_cyclic != nullptr) {
        ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<std::complex<double>, ARCH>>::Solve(deg, tol, mode, opt, qr);
    } else if (ChASE_DIST<chase::distMatrix::BlockBlockMatrix<std::complex<double>, ARCH>>::zchaseDist_block != nullptr) {
        ChASE_DIST<chase::distMatrix::BlockBlockMatrix<std::complex<double>, ARCH>>::Solve(deg, tol, mode, opt, qr);
    }
}
void pcchase_(int* deg, float* tol, char* mode, char* opt, char *qr)
{
    // Check pseudo-Hermitian solvers first (inheritance allows unified interface)
    if (ChASE_DIST<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<float>, ARCH>>::cchaseDist_pseudo_cyclic != nullptr) {
        ChASE_DIST<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<float>, ARCH>>::Solve(deg, tol, mode, opt, qr);
    } else if (ChASE_DIST<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<float>, ARCH>>::cchaseDist_pseudo_block != nullptr) {
        ChASE_DIST<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<float>, ARCH>>::Solve(deg, tol, mode, opt, qr);
    } else if (ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<std::complex<float>, ARCH>>::cchaseDist_cyclic != nullptr) {
        ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<std::complex<float>, ARCH>>::Solve(deg, tol, mode, opt, qr);
    } else if (ChASE_DIST<chase::distMatrix::BlockBlockMatrix<std::complex<float>, ARCH>>::cchaseDist_block != nullptr) {
        ChASE_DIST<chase::distMatrix::BlockBlockMatrix<std::complex<float>, ARCH>>::Solve(deg, tol, mode, opt, qr);
    }
}

// Unified write functions - work for both BlockBlockMatrix and BlockCyclicMatrix
// Checks which type was initialized and calls the appropriate write function
void pschase_wrtHam_(const char* filename)
{
    std::string filename_str(filename);
    if (ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<float, ARCH>>::schaseDist_cyclic != nullptr) {
        ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<float, ARCH>>::WrteHam(filename_str);
    } else if (ChASE_DIST<chase::distMatrix::BlockBlockMatrix<float, ARCH>>::schaseDist_block != nullptr) {
        ChASE_DIST<chase::distMatrix::BlockBlockMatrix<float, ARCH>>::WrteHam(filename_str);
    }
}

void pdchase_wrtHam_(const char* filename)
{
    std::string filename_str(filename);
    if (ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<double, ARCH>>::dchaseDist_cyclic != nullptr) {
        ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<double, ARCH>>::WrteHam(filename_str);
    } else if (ChASE_DIST<chase::distMatrix::BlockBlockMatrix<double, ARCH>>::dchaseDist_block != nullptr) {
        ChASE_DIST<chase::distMatrix::BlockBlockMatrix<double, ARCH>>::WrteHam(filename_str);
    }
}

void pcchase_wrtHam_(const char* filename)
{
    std::string filename_str(filename);
    // Check pseudo-Hermitian solvers first (inheritance allows unified interface)
    if (ChASE_DIST<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<float>, ARCH>>::cchaseDist_pseudo_cyclic != nullptr) {
        ChASE_DIST<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<float>, ARCH>>::WrteHam(filename_str);
    } else if (ChASE_DIST<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<float>, ARCH>>::cchaseDist_pseudo_block != nullptr) {
        ChASE_DIST<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<float>, ARCH>>::WrteHam(filename_str);
    } else if (ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<std::complex<float>, ARCH>>::cchaseDist_cyclic != nullptr) {
        ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<std::complex<float>, ARCH>>::WrteHam(filename_str);
    } else if (ChASE_DIST<chase::distMatrix::BlockBlockMatrix<std::complex<float>, ARCH>>::cchaseDist_block != nullptr) {
        ChASE_DIST<chase::distMatrix::BlockBlockMatrix<std::complex<float>, ARCH>>::WrteHam(filename_str);
    }
}

void pzchase_wrtHam_(const char* filename)
{
    std::string filename_str(filename);
    // Check pseudo-Hermitian solvers first (inheritance allows unified interface)
    if (ChASE_DIST<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<double>, ARCH>>::zchaseDist_pseudo_cyclic != nullptr) {
        ChASE_DIST<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<double>, ARCH>>::WrteHam(filename_str);
    } else if (ChASE_DIST<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<double>, ARCH>>::zchaseDist_pseudo_block != nullptr) {
        ChASE_DIST<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<double>, ARCH>>::WrteHam(filename_str);
    } else if (ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<std::complex<double>, ARCH>>::zchaseDist_cyclic != nullptr) {
        ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<std::complex<double>, ARCH>>::WrteHam(filename_str);
    } else if (ChASE_DIST<chase::distMatrix::BlockBlockMatrix<std::complex<double>, ARCH>>::zchaseDist_block != nullptr) {
        ChASE_DIST<chase::distMatrix::BlockBlockMatrix<std::complex<double>, ARCH>>::WrteHam(filename_str);
    }
}

// Unified read functions - work for both BlockBlockMatrix and BlockCyclicMatrix
// Checks which type was initialized and calls the appropriate read function
void pschase_readHam_(const char* filename)
{
    std::string filename_str(filename);
    if (ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<float, ARCH>>::schaseDist_cyclic != nullptr) {
        ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<float, ARCH>>::readHam(filename_str);
    } else if (ChASE_DIST<chase::distMatrix::BlockBlockMatrix<float, ARCH>>::schaseDist_block != nullptr) {
        ChASE_DIST<chase::distMatrix::BlockBlockMatrix<float, ARCH>>::readHam(filename_str);
    }
}

void pdchase_readHam_(const char* filename)
{
    std::string filename_str(filename);
    if (ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<double, ARCH>>::dchaseDist_cyclic != nullptr) {
        ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<double, ARCH>>::readHam(filename_str);
    } else if (ChASE_DIST<chase::distMatrix::BlockBlockMatrix<double, ARCH>>::dchaseDist_block != nullptr) {
        ChASE_DIST<chase::distMatrix::BlockBlockMatrix<double, ARCH>>::readHam(filename_str);
    }
}

void pcchase_readHam_(const char* filename)
{
    std::string filename_str(filename);
    // Check pseudo-Hermitian solvers first (inheritance allows unified interface)
    if (ChASE_DIST<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<float>, ARCH>>::cchaseDist_pseudo_cyclic != nullptr) {
        ChASE_DIST<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<float>, ARCH>>::readHam(filename_str);
    } else if (ChASE_DIST<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<float>, ARCH>>::cchaseDist_pseudo_block != nullptr) {
        ChASE_DIST<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<float>, ARCH>>::readHam(filename_str);
    } else if (ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<std::complex<float>, ARCH>>::cchaseDist_cyclic != nullptr) {
        ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<std::complex<float>, ARCH>>::readHam(filename_str);
    } else if (ChASE_DIST<chase::distMatrix::BlockBlockMatrix<std::complex<float>, ARCH>>::cchaseDist_block != nullptr) {
        ChASE_DIST<chase::distMatrix::BlockBlockMatrix<std::complex<float>, ARCH>>::readHam(filename_str);
    }
}

void pzchase_readHam_(const char* filename)
{
    std::string filename_str(filename);
    // Check pseudo-Hermitian solvers first (inheritance allows unified interface)
    if (ChASE_DIST<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<double>, ARCH>>::zchaseDist_pseudo_cyclic != nullptr) {
        ChASE_DIST<chase::distMatrix::PseudoHermitianBlockCyclicMatrix<std::complex<double>, ARCH>>::readHam(filename_str);
    } else if (ChASE_DIST<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<double>, ARCH>>::zchaseDist_pseudo_block != nullptr) {
        ChASE_DIST<chase::distMatrix::PseudoHermitianBlockBlockMatrix<std::complex<double>, ARCH>>::readHam(filename_str);
    } else if (ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<std::complex<double>, ARCH>>::zchaseDist_cyclic != nullptr) {
        ChASE_DIST<chase::distMatrix::BlockCyclicMatrix<std::complex<double>, ARCH>>::readHam(filename_str);
    } else if (ChASE_DIST<chase::distMatrix::BlockBlockMatrix<std::complex<double>, ARCH>>::zchaseDist_block != nullptr) {
        ChASE_DIST<chase::distMatrix::BlockBlockMatrix<std::complex<double>, ARCH>>::readHam(filename_str);
    }
}

// Note: The _pseudo functions have been unified into the regular functions above.
// The regular pzchase_, pcchase_, pzchase_finalize_, pcchase_finalize_, etc. 
// now handle both regular and pseudo-Hermitian types by checking pseudo-Hermitian solvers first.

}
