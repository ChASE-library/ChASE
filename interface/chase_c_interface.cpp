#include <complex>
#include <iostream>
#include <iomanip>
#include <memory>

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

#ifdef INTERFACE_BLOCK_CYCLIC
template <typename T>
using BlockMatrixType = chase::distMatrix::BlockCyclicMatrix<T, ARCH>;
template<typename T>
using DistMultiVector1DColumn = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column, ARCH>;
#else
template <typename T>
using BlockMatrixType = chase::distMatrix::BlockBlockMatrix<T, ARCH>;
template<typename T>
using DistMultiVector1DColumn = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, ARCH>;
#endif


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
#ifdef INTERFACE_BLOCK_CYCLIC
    static typename std::enable_if<std::is_same<MatrixType, chase::distMatrix::BlockCyclicMatrix<T, ARCH>>::value, int>::type
    Initialize(int N, int nev, int nex, int mbsize, int nbsize,
                                T* H, int ldh, T* V, chase::Base<T>* ritzv,
                                int dim0, int dim1, char* grid_major, int irsrc,
                                int icsrc, MPI_Comm comm);
#else
    static typename std::enable_if<std::is_same<MatrixType, chase::distMatrix::BlockBlockMatrix<T, ARCH>>::value, int>::type
    Initialize(int N, int nev, int nex, int m, int n, T* H, int ldh,
               T* V, chase::Base<T>* ritzv, int dim0, int dim1,
               char* grid_major, MPI_Comm comm);
#endif

    static DistSolverType<BlockMatrixType<std::complex<double>>> *zchaseDist;
    static DistSolverType<BlockMatrixType<std::complex<float>>> *cchaseDist;
    static DistSolverType<BlockMatrixType<float>> *schaseDist;
    static DistSolverType<BlockMatrixType<double>> *dchaseDist;

    static BlockMatrixType<std::complex<double>>* zHmat;
    static BlockMatrixType<std::complex<float>>* cHmat;
    static BlockMatrixType<double>* dHmat;
    static BlockMatrixType<float>* sHmat;

    static DistMultiVector1DColumn<std::complex<double>>* zVec;
    static DistMultiVector1DColumn<std::complex<float>>* cVec;
    static DistMultiVector1DColumn<double>* dVec;
    static DistMultiVector1DColumn<float>* sVec;

    static DistSolverType<MatrixType>* getChase();
    static int Finalize();
    static void Solve(int* deg, chase::Base<T>* tol, char* mode, char* opt, char *qr);
};

template <typename MatrixType>
DistSolverType<BlockMatrixType<std::complex<double>>> *ChASE_DIST<MatrixType>::zchaseDist = nullptr;
template <typename MatrixType>
DistSolverType<BlockMatrixType<std::complex<float>>> *ChASE_DIST<MatrixType>::cchaseDist = nullptr;
template <typename MatrixType>
DistSolverType<BlockMatrixType<float>> *ChASE_DIST<MatrixType>::schaseDist = nullptr;
template <typename MatrixType>
DistSolverType<BlockMatrixType<double>> *ChASE_DIST<MatrixType>::dchaseDist = nullptr;

template <typename MatrixType>
BlockMatrixType<std::complex<double>>* ChASE_DIST<MatrixType>::zHmat = nullptr;
template <typename MatrixType>
BlockMatrixType<std::complex<float>>* ChASE_DIST<MatrixType>::cHmat = nullptr;
template <typename MatrixType>
BlockMatrixType<double>* ChASE_DIST<MatrixType>::dHmat = nullptr;
template <typename MatrixType>
BlockMatrixType<float>* ChASE_DIST<MatrixType>::sHmat = nullptr;

template <typename MatrixType>
DistMultiVector1DColumn<std::complex<double>>* ChASE_DIST<MatrixType>::zVec = nullptr;
template <typename MatrixType>
DistMultiVector1DColumn<std::complex<float>>*  ChASE_DIST<MatrixType>::cVec = nullptr;
template <typename MatrixType>
DistMultiVector1DColumn<double>*  ChASE_DIST<MatrixType>::dVec = nullptr;
template <typename MatrixType>
DistMultiVector1DColumn<float>*  ChASE_DIST<MatrixType>::sVec = nullptr;

#ifdef INTERFACE_BLOCK_CYCLIC
template <>
int ChASE_DIST<BlockMatrixType<std::complex<double>>>::Initialize(int N, int nev, int nex, int mbsize, int nbsize,
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

    zHmat = new BlockMatrixType<std::complex<double>>(N, N, m, n, mbsize, nbsize, ldh, H, mpi_grid);    
    zVec = new DistMultiVector1DColumn<std::complex<double>>(N, m, nev + nex, mbsize, m, V, mpi_grid);  

    zchaseDist = new DistSolverType<BlockMatrixType<std::complex<double>>>(nev, nex, zHmat, zVec, ritzv);
    
    return 1;    
}

template <>
int ChASE_DIST<BlockMatrixType<std::complex<float>>>::Initialize(int N, int nev, int nex, int mbsize, int nbsize,
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

    cHmat = new BlockMatrixType<std::complex<float>>(N, N, m, n, mbsize, nbsize, ldh, H, mpi_grid);    
    cVec = new DistMultiVector1DColumn<std::complex<float>>(N, m, nev + nex, mbsize, m, V, mpi_grid);  

    cchaseDist = new DistSolverType<BlockMatrixType<std::complex<float>>>(nev, nex, cHmat, cVec, ritzv);
    
    return 1;    
}

template <>
int ChASE_DIST<BlockMatrixType<float>>::Initialize(int N, int nev, int nex, int mbsize, int nbsize,
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

    sHmat = new BlockMatrixType<float>(N, N, m, n, mbsize, nbsize, ldh, H, mpi_grid);    
    sVec = new DistMultiVector1DColumn<float>(N, m, nev + nex, mbsize, m, V, mpi_grid);  

    schaseDist = new DistSolverType<BlockMatrixType<float>>(nev, nex, sHmat, sVec, ritzv);
        
    return 1;    
}

template <>
int ChASE_DIST<BlockMatrixType<double>>::Initialize(int N, int nev, int nex, int mbsize, int nbsize,
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

    dHmat = new BlockMatrixType<double>(N, N, m, n, mbsize, nbsize, ldh, H, mpi_grid);    
    dVec = new DistMultiVector1DColumn<double>(N, m, nev + nex, mbsize, m, V, mpi_grid);  
    
    dchaseDist = new DistSolverType<BlockMatrixType<double>>(nev, nex, dHmat, dVec, ritzv);

    return 1;    
}

#else
template <>
int ChASE_DIST<BlockMatrixType<std::complex<double>>>::Initialize(int N, int nev, int nex, int m, int n, std::complex<double>* H, int ldh,
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

    zHmat = new BlockMatrixType<std::complex<double>>(m, n, ldh, H, mpi_grid);    
    zVec = new DistMultiVector1DColumn<std::complex<double>>(m, nev + nex, m, V, mpi_grid);  

    zchaseDist = new DistSolverType<BlockMatrixType<std::complex<double>>>(nev, nex, zHmat, zVec, ritzv);

    return 1;
}

template <>
int ChASE_DIST<BlockMatrixType<std::complex<float>>>::Initialize(int N, int nev, int nex, int m, int n, std::complex<float>* H, int ldh,
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

    cHmat = new BlockMatrixType<std::complex<float>>(m, n, ldh, H, mpi_grid);    
    cVec = new DistMultiVector1DColumn<std::complex<float>>(m, nev + nex, m, V, mpi_grid);  

    cchaseDist = new DistSolverType<BlockMatrixType<std::complex<float>>>(nev, nex, cHmat, cVec, ritzv);

    return 1;
}

template <>
int ChASE_DIST<BlockMatrixType<float>>::Initialize(int N, int nev, int nex, int m, int n, float* H, int ldh,
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

    sHmat = new BlockMatrixType<float>(m, n, ldh, H, mpi_grid);    
    sVec = new DistMultiVector1DColumn<float>(m, nev + nex, m, V, mpi_grid);  

    schaseDist = new DistSolverType<BlockMatrixType<float>>(nev, nex, sHmat, sVec, ritzv);
      
    return 1;
}

template <>
int ChASE_DIST<BlockMatrixType<double>>::Initialize(int N, int nev, int nex, int m, int n, double* H, int ldh,
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

    dHmat = new BlockMatrixType<double>(m, n, ldh, H, mpi_grid);    
    dVec = new DistMultiVector1DColumn<double>(m, nev + nex, m, V, mpi_grid);  

    dchaseDist = new DistSolverType<BlockMatrixType<double>>(nev, nex, dHmat, dVec, ritzv);

    return 1;
}
#endif
/*
*/
template<>
DistSolverType<BlockMatrixType<std::complex<double>>> *ChASE_DIST<BlockMatrixType<std::complex<double>>>::getChase()
{
    return zchaseDist;
}

template<>
DistSolverType<BlockMatrixType<std::complex<float>>> *ChASE_DIST<BlockMatrixType<std::complex<float>>>::getChase()
{
    return cchaseDist;
}

template<>
DistSolverType<BlockMatrixType<double>> *ChASE_DIST<BlockMatrixType<double>>::getChase()
{
    return dchaseDist;
}

template<>
DistSolverType<BlockMatrixType<float>> *ChASE_DIST<BlockMatrixType<float>>::getChase()
{
    return schaseDist;
}

template<>
int ChASE_DIST<BlockMatrixType<std::complex<double>>>::Finalize()
{
    delete zchaseDist;
    zchaseDist = nullptr;
    delete zHmat;
    zHmat = nullptr;
    delete zVec;
    zVec = nullptr;    
    return 0;
}

template<>
int ChASE_DIST<BlockMatrixType<std::complex<float>>>::Finalize()
{
    delete cchaseDist;
    cchaseDist = nullptr;
    delete cHmat;
    cHmat = nullptr;
    delete cVec;
    cVec = nullptr;    
    return 0;
}

template<>
int ChASE_DIST<BlockMatrixType<double>>::Finalize()
{
    delete dchaseDist;
    dchaseDist = nullptr;
    delete dHmat;
    dHmat = nullptr;
    delete dVec;
    dVec = nullptr;    
    return 0;
}

template<>
int ChASE_DIST<BlockMatrixType<float>>::Finalize()
{
    delete schaseDist;
    schaseDist = nullptr;
    delete sHmat;
    sHmat = nullptr;
    delete sVec;
    sVec = nullptr;    
    return 0;
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
    
    chase::PerformanceDecoratorChase<T> performanceDecorator(single);

    chase::Solve(&performanceDecorator);

    chase::Base<T>* ritzv = single->GetRitzv();
    chase::Base<T>* resid = single->GetResid();

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

#ifdef INTERFACE_BLOCK_CYCLIC
void pdchase_init_blockcyclic_(int* N, int* nev, int* nex, int* mbsize,
                                int* nbsize, double* H, int* ldh, double* V,
                                double* ritzv, int* dim0, int* dim1,
                                char* grid_major, int* irsrc, int* icsrc,
                                MPI_Comm* comm, int* init)
{
    *init =  ChASE_DIST<BlockMatrixType<double>>::Initialize(*N, *nev, *nex, *mbsize, *nbsize, H,
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
    *init = ChASE_DIST<BlockMatrixType<double>>::Initialize(*N, *nev, *nex, *mbsize, *nbsize, H,
                                    *ldh, V, ritzv, *dim0, *dim1,
                                    grid_major, *irsrc, *icsrc, comm);
}

void pschase_init_blockcyclic_(int* N, int* nev, int* nex, int* mbsize,
                                int* nbsize, float* H, int* ldh, float* V,
                                float* ritzv, int* dim0, int* dim1,
                                char* grid_major, int* irsrc, int* icsrc,
                                MPI_Comm* comm, int* init)
{
    *init =  ChASE_DIST<BlockMatrixType<float>>::Initialize(*N, *nev, *nex, *mbsize, *nbsize, H,
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
    *init = ChASE_DIST<BlockMatrixType<float>>::Initialize(*N, *nev, *nex, *mbsize, *nbsize, H,
                                    *ldh, V, ritzv, *dim0, *dim1, grid_major,
                                    *irsrc, *icsrc, comm);
}


void pcchase_init_blockcyclic_(int* N, int* nev, int* nex, int* mbsize,
                                int* nbsize, float _Complex* H, int* ldh,
                                float _Complex* V, float* ritzv, int* dim0,
                                int* dim1, char* grid_major, int* irsrc,
                                int* icsrc, MPI_Comm* comm, int* init)
{
    *init = ChASE_DIST<BlockMatrixType<std::complex<float>>>::Initialize(
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
    *init = ChASE_DIST<BlockMatrixType<std::complex<float>>>::Initialize(
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
    *init = ChASE_DIST<BlockMatrixType<std::complex<double>>>::Initialize(
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
    *init = ChASE_DIST<BlockMatrixType<std::complex<double>>>::Initialize(
        *N, *nev, *nex, *mbsize, *nbsize,
        reinterpret_cast<std::complex<double>*>(H), *ldh,
        reinterpret_cast<std::complex<double>*>(V), ritzv, *dim0, *dim1,
        grid_major, *irsrc, *icsrc, comm);
}

void pdchase_init_(int* N, int* nev, int* nex, int* mbsize,
                                int* nbsize, double* H, int* ldh, double* V,
                                double* ritzv, int* dim0, int* dim1,
                                char* grid_major, int* irsrc, int* icsrc,
                                MPI_Comm* comm, int* init)
{
    *init =  ChASE_DIST<BlockMatrixType<double>>::Initialize(*N, *nev, *nex, *mbsize, *nbsize, H,
                                    *ldh, V, ritzv, *dim0, *dim1, grid_major,
                                    *irsrc, *icsrc, *comm);    
}                                

void pschase_init_(int* N, int* nev, int* nex, int* mbsize,
                                int* nbsize, float* H, int* ldh, float* V,
                                float* ritzv, int* dim0, int* dim1,
                                char* grid_major, int* irsrc, int* icsrc,
                                MPI_Comm* comm, int* init)
{
    *init =  ChASE_DIST<BlockMatrixType<float>>::Initialize(*N, *nev, *nex, *mbsize, *nbsize, H,
                                    *ldh, V, ritzv, *dim0, *dim1, grid_major,
                                    *irsrc, *icsrc, *comm);
}                                

void pcchase_init_(int* N, int* nev, int* nex, int* mbsize,
                                int* nbsize, float _Complex* H, int* ldh,
                                float _Complex* V, float* ritzv, int* dim0,
                                int* dim1, char* grid_major, int* irsrc,
                                int* icsrc, MPI_Comm* comm, int* init)
{
    *init = ChASE_DIST<BlockMatrixType<std::complex<float>>>::Initialize(
        *N, *nev, *nex, *mbsize, *nbsize,
        reinterpret_cast<std::complex<float>*>(H), *ldh,
        reinterpret_cast<std::complex<float>*>(V), ritzv, *dim0, *dim1,
        grid_major, *irsrc, *icsrc, *comm);

}

void pzchase_init_(int* N, int* nev, int* nex, int* mbsize,
                                int* nbsize, double _Complex* H, int* ldh,
                                double _Complex* V, double* ritzv, int* dim0,
                                int* dim1, char* grid_major, int* irsrc,
                                int* icsrc, MPI_Comm* comm, int* init)
{
    *init = ChASE_DIST<BlockMatrixType<std::complex<double>>>::Initialize(
        *N, *nev, *nex, *mbsize, *nbsize,
        reinterpret_cast<std::complex<double>*>(H), *ldh,
        reinterpret_cast<std::complex<double>*>(V), ritzv, *dim0, *dim1,
        grid_major, *irsrc, *icsrc, *comm);
}

#else
void pdchase_init_(int* N, int* nev, int* nex, int* m, int* n, double* H,
                    int* ldh, double* V, double* ritzv, int* dim0, int* dim1,
                    char* grid_major, MPI_Comm* comm, int* init)
{
    *init = ChASE_DIST<BlockMatrixType<double>>::Initialize(*N, *nev, *nex, *m, *n, H, *ldh, V,
                                    ritzv, *dim0, *dim1, grid_major, *comm);
}

void pdchase_init_f_(int* N, int* nev, int* nex, int* m, int* n, double* H,
                        int* ldh, double* V, double* ritzv, int* dim0,
                        int* dim1, char* grid_major, MPI_Fint* fcomm,
                        int* init)
{
    MPI_Comm comm = MPI_Comm_f2c(*fcomm);
    *init = ChASE_DIST<BlockMatrixType<double>>::Initialize(*N, *nev, *nex, *m, *n, H, *ldh, V,
                                    ritzv, *dim0, *dim1, grid_major, comm);
}

void pschase_init_(int* N, int* nev, int* nex, int* m, int* n, float* H,
                    int* ldh, float* V, float* ritzv, int* dim0, int* dim1,
                    char* grid_major, MPI_Comm* comm, int* init)
{
    *init = ChASE_DIST<BlockMatrixType<float>>::Initialize(*N, *nev, *nex, *m, *n, H, *ldh, V,
                                    ritzv, *dim0, *dim1, grid_major, *comm);
}                    

void pschase_init_f_(int* N, int* nev, int* nex, int* m, int* n, float* H,
                    int* ldh, float* V, float* ritzv, int* dim0, int* dim1,
                    char* grid_major, MPI_Fint* fcomm, int* init)
{
    MPI_Comm comm = MPI_Comm_f2c(*fcomm);
    *init = ChASE_DIST<BlockMatrixType<float>>::Initialize(*N, *nev, *nex, *m, *n, H, *ldh, V,
                                    ritzv, *dim0, *dim1, grid_major, comm);   
}


void pzchase_init_(int* N, int* nev, int* nex, int* m, int* n,
                    double _Complex* H, int* ldh, double _Complex* V,
                    double* ritzv, int* dim0, int* dim1, char* grid_major,
                    MPI_Comm* comm, int* init)
{
    *init = ChASE_DIST<BlockMatrixType<std::complex<double>>>::Initialize(*N, *nev, *nex, *m, *n, 
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
    *init = ChASE_DIST<BlockMatrixType<std::complex<double>>>::Initialize(
        *N, *nev, *nex, *m, *n, reinterpret_cast<std::complex<double>*>(H),
        *ldh, reinterpret_cast<std::complex<double>*>(V), ritzv, *dim0,
        *dim1, grid_major, comm);
}

void pcchase_init_(int* N, int* nev, int* nex, int* m, int* n,
                    float _Complex* H, int* ldh, float _Complex* V,
                    float* ritzv, int* dim0, int* dim1, char* grid_major,
                    MPI_Comm* comm, int* init)
{
    *init = ChASE_DIST<BlockMatrixType<std::complex<float>>>::Initialize(*N, *nev, *nex, *m, *n, 
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
    *init = ChASE_DIST<BlockMatrixType<std::complex<float>>>::Initialize(
        *N, *nev, *nex, *m, *n, reinterpret_cast<std::complex<float>*>(H),
        *ldh, reinterpret_cast<std::complex<float>*>(V), ritzv, *dim0,
        *dim1, grid_major, comm);
}
#endif

void pdchase_finalize_(int* flag) { *flag = ChASE_DIST<BlockMatrixType<double>>::Finalize(); }
void pschase_finalize_(int* flag) { *flag = ChASE_DIST<BlockMatrixType<float>>::Finalize(); }
void pcchase_finalize_(int* flag) { *flag = ChASE_DIST<BlockMatrixType<std::complex<float>>>::Finalize(); }
void pzchase_finalize_(int* flag) { *flag =ChASE_DIST<BlockMatrixType<std::complex<double>>>::Finalize(); }

void pdchase_(int* deg, double* tol, char* mode, char* opt, char *qr)
{
    ChASE_DIST<BlockMatrixType<double>>::Solve(deg, tol, mode, opt, qr);
}
void pschase_(int* deg, float* tol, char* mode, char* opt, char *qr)
{
    ChASE_DIST<BlockMatrixType<float>>::Solve(deg, tol, mode, opt, qr);
}
void pzchase_(int* deg, double* tol, char* mode, char* opt, char *qr)
{
    ChASE_DIST<BlockMatrixType<std::complex<double>>>::Solve(deg, tol, mode, opt, qr);
}
void pcchase_(int* deg, float* tol, char* mode, char* opt, char *qr)
{
    ChASE_DIST<BlockMatrixType<std::complex<float>>>::Solve(deg, tol, mode, opt, qr);
}


}
