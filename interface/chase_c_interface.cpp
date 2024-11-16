#include <complex>
#include <iostream>
#include <iomanip>
#include <memory>

#include "algorithm/performance.hpp"

#ifdef HAS_CUDA
#include "Impl/cuda/chase_seq_gpu.hpp"
template <typename T>
using SeqSolverType = chase::Impl::ChaseGPUSeq<T>;
#else
#include "Impl/cpu/chase_seq_cpu.hpp"
template <typename T>
using SeqSolverType = chase::Impl::ChaseCPUSeq<T>;
#endif

#ifdef HAS_NCCL
#include "Impl/nccl/chase_nccl_gpu.hpp"
using ARCH = chase::platform::GPU;
template<typename T>
using DistSolverBlockType = chase::Impl::ChaseNCCLGPU<chase::distMatrix::BlockBlockMatrix<T, ARCH>,                                                      
                                                     chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, ARCH>>;
#else
#include "Impl/mpi/chase_mpi_cpu.hpp"
using ARCH = chase::platform::CPU;
template<typename T>
using DistSolverBlockType = chase::Impl::ChaseMPICPU<chase::distMatrix::BlockBlockMatrix<T, ARCH>,                                                      
                                                     chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, ARCH>>;
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

class ChASE_DIST_BLOCK
{
public:
    template <typename T>
    static int Initialize(int N, int nev, int nex, int m, int n, T* H, int ldh,
                           T* V, chase::Base<T>* ritzv, int dim0, int dim1,
                           char* grid_major, MPI_Comm comm);

    template <typename T>
    static void Solve(int* deg, chase::Base<T>* tol, char* mode, char* opt, char *qr);
                     
    template <typename T>
    static int Finalize();

    template <typename T>
    static DistSolverBlockType<T>* getChase();

    static DistSolverBlockType<std::complex<double>> *zchaseDist;
    static DistSolverBlockType<std::complex<float>> *cchaseDist;
    static DistSolverBlockType<double> *dchaseDist;
    static DistSolverBlockType<float> *schaseDist;    

    static chase::distMatrix::BlockBlockMatrix<std::complex<double>, ARCH>* zHmat;
    static chase::distMatrix::BlockBlockMatrix<std::complex<float>, ARCH>* cHmat;
    static chase::distMatrix::BlockBlockMatrix<double, ARCH>* dHmat;
    static chase::distMatrix::BlockBlockMatrix<float, ARCH>* sHmat;

    static chase::distMultiVector::DistMultiVector1D<std::complex<double>, chase::distMultiVector::CommunicatorType::column, ARCH>* zVec;
    static chase::distMultiVector::DistMultiVector1D<std::complex<float>, chase::distMultiVector::CommunicatorType::column, ARCH>* cVec;
    static chase::distMultiVector::DistMultiVector1D<double, chase::distMultiVector::CommunicatorType::column, ARCH>* dVec;
    static chase::distMultiVector::DistMultiVector1D<float, chase::distMultiVector::CommunicatorType::column, ARCH>* sVec;
};

DistSolverBlockType<std::complex<double>> *ChASE_DIST_BLOCK::zchaseDist = nullptr;
DistSolverBlockType<std::complex<float>> *ChASE_DIST_BLOCK::cchaseDist = nullptr;
DistSolverBlockType<double> *ChASE_DIST_BLOCK::dchaseDist = nullptr;
DistSolverBlockType<float> *ChASE_DIST_BLOCK::schaseDist = nullptr;   

chase::distMatrix::BlockBlockMatrix<std::complex<double>, ARCH>* ChASE_DIST_BLOCK::zHmat = nullptr;
chase::distMatrix::BlockBlockMatrix<std::complex<float>, ARCH>* ChASE_DIST_BLOCK::cHmat = nullptr;
chase::distMatrix::BlockBlockMatrix<double, ARCH>* ChASE_DIST_BLOCK::dHmat = nullptr;
chase::distMatrix::BlockBlockMatrix<float, ARCH>* ChASE_DIST_BLOCK::sHmat = nullptr;

chase::distMultiVector::DistMultiVector1D<std::complex<double>, chase::distMultiVector::CommunicatorType::column, ARCH>*  ChASE_DIST_BLOCK::zVec = nullptr;
chase::distMultiVector::DistMultiVector1D<std::complex<float>, chase::distMultiVector::CommunicatorType::column, ARCH>*  ChASE_DIST_BLOCK::cVec = nullptr;
chase::distMultiVector::DistMultiVector1D<double, chase::distMultiVector::CommunicatorType::column, ARCH>*  ChASE_DIST_BLOCK::dVec = nullptr;
chase::distMultiVector::DistMultiVector1D<float, chase::distMultiVector::CommunicatorType::column, ARCH>*  ChASE_DIST_BLOCK::sVec = nullptr;

template <>
int ChASE_DIST_BLOCK::Initialize(int N, int nev, int nex, int m, int n, double* H, int ldh,
                           double* V, double* ritzv, int dim0, int dim1,
                           char* grid_major, MPI_Comm comm)
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

    dHmat = new chase::distMatrix::BlockBlockMatrix<double, ARCH>(m, n, ldh, H, mpi_grid);    
    dVec = new chase::distMultiVector::DistMultiVector1D<double, chase::distMultiVector::CommunicatorType::column, ARCH>(m, nev + nex, m, V, mpi_grid);  

    dchaseDist = new DistSolverBlockType<double>(nev, nex, dHmat, dVec, ritzv);

    return 1;
}

template <>
int ChASE_DIST_BLOCK::Initialize(int N, int nev, int nex, int m, int n, float* H, int ldh,
                           float* V, float* ritzv, int dim0, int dim1,
                           char* grid_major, MPI_Comm comm)
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

    sHmat = new chase::distMatrix::BlockBlockMatrix<float, ARCH>(m, n, ldh, H, mpi_grid);    
    sVec = new chase::distMultiVector::DistMultiVector1D<float, chase::distMultiVector::CommunicatorType::column, ARCH>(m, nev + nex, m, V, mpi_grid);  

    schaseDist = new DistSolverBlockType<float>(nev, nex, sHmat, sVec, ritzv);
    return 1;
}

template <>
int ChASE_DIST_BLOCK::Initialize(int N, int nev, int nex, int m, int n, std::complex<float>* H, int ldh,
                           std::complex<float>* V, float* ritzv, int dim0, int dim1,
                           char* grid_major, MPI_Comm comm)
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

    cHmat = new chase::distMatrix::BlockBlockMatrix<std::complex<float>, ARCH>(m, n, ldh, H, mpi_grid);    
    cVec = new chase::distMultiVector::DistMultiVector1D<std::complex<float>, chase::distMultiVector::CommunicatorType::column, ARCH>(m, nev + nex, m, V, mpi_grid);  

    cchaseDist = new DistSolverBlockType<std::complex<float>>(nev, nex, cHmat, cVec, ritzv);
    return 1;
}

template <>
int ChASE_DIST_BLOCK::Initialize(int N, int nev, int nex, int m, int n, std::complex<double>* H, int ldh,
                           std::complex<double>* V, double* ritzv, int dim0, int dim1,
                           char* grid_major, MPI_Comm comm)
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

    zHmat = new chase::distMatrix::BlockBlockMatrix<std::complex<double>, ARCH>(m, n, ldh, H, mpi_grid);    
    zVec = new chase::distMultiVector::DistMultiVector1D<std::complex<double>, chase::distMultiVector::CommunicatorType::column, ARCH>(m, nev + nex, m, V, mpi_grid);  

    zchaseDist = new DistSolverBlockType<std::complex<double>>(nev, nex, zHmat, zVec, ritzv);
    return 1;
}

template <>
int ChASE_DIST_BLOCK::Finalize<double>()
{
    delete dchaseDist;
    dchaseDist = nullptr;
    delete dHmat;
    dHmat = nullptr;
    delete dVec;
    dVec = nullptr; 

    return 0;
}

template <>
int ChASE_DIST_BLOCK::Finalize<float>()
{
    delete schaseDist;
    schaseDist = nullptr;
    delete sHmat;
    sHmat = nullptr;
    delete sVec;
    sVec = nullptr; 

    return 0;
}

template <>
int ChASE_DIST_BLOCK::Finalize<std::complex<float>>()
{
    delete cchaseDist;
    cchaseDist = nullptr;
    delete cHmat;
    cHmat = nullptr;
    delete cVec;
    cVec = nullptr; 
        
    return 0;
}

template <>
int ChASE_DIST_BLOCK::Finalize<std::complex<double>>()
{
    delete zchaseDist;
    zchaseDist = nullptr;
    delete zHmat;
    zHmat = nullptr;
    delete zVec;
    zVec = nullptr;    
    return 0;
}

template <>
DistSolverBlockType<std::complex<double>>* ChASE_DIST_BLOCK::getChase<std::complex<double>>() {
    return zchaseDist;
}

template <>
DistSolverBlockType<std::complex<float>>* ChASE_DIST_BLOCK::getChase<std::complex<float>>() {
    return cchaseDist;
}

template <>
DistSolverBlockType<double>* ChASE_DIST_BLOCK::getChase<double>() {
    return dchaseDist;
}

template <>
DistSolverBlockType<float>* ChASE_DIST_BLOCK::getChase<float>() {
    return schaseDist;
}

template<typename T>
void ChASE_DIST_BLOCK::Solve(int* deg, chase::Base<T>* tol, char* mode, char* opt, char *qr)
{
    auto single = ChASE_DIST_BLOCK::getChase<T>();
    
    chase::ChaseConfig<T>& config = single->GetConfig();
    config.SetTol(*tol);
    config.SetDeg(*deg);
    config.SetOpt(*opt == 'S');
    config.SetApprox(*mode == 'A');
    config.SetCholQR(*qr == 'C');

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

void pdchase_init_(int* N, int* nev, int* nex, int* m, int* n, double* H,
                    int* ldh, double* V, double* ritzv, int* dim0, int* dim1,
                    char* grid_major, MPI_Comm* comm, int* init)
{
    *init = ChASE_DIST_BLOCK::Initialize<double>(*N, *nev, *nex, *m, *n, H, *ldh, V,
                                    ritzv, *dim0, *dim1, grid_major, *comm);
}

void pdchase_init_f_(int* N, int* nev, int* nex, int* m, int* n, double* H,
                        int* ldh, double* V, double* ritzv, int* dim0,
                        int* dim1, char* grid_major, MPI_Fint* fcomm,
                        int* init)
{
    MPI_Comm comm = MPI_Comm_f2c(*fcomm);
    *init = ChASE_DIST_BLOCK::Initialize<double>(*N, *nev, *nex, *m, *n, H, *ldh, V,
                                    ritzv, *dim0, *dim1, grid_major, comm);
}

void pschase_init_(int* N, int* nev, int* nex, int* m, int* n, float* H,
                    int* ldh, float* V, float* ritzv, int* dim0, int* dim1,
                    char* grid_major, MPI_Comm* comm, int* init)
{
    *init = ChASE_DIST_BLOCK::Initialize<float>(*N, *nev, *nex, *m, *n, H, *ldh, V,
                                    ritzv, *dim0, *dim1, grid_major, *comm);
}                    

void pschase_init_f_(int* N, int* nev, int* nex, int* m, int* n, float* H,
                    int* ldh, float* V, float* ritzv, int* dim0, int* dim1,
                    char* grid_major, MPI_Fint* fcomm, int* init)
{
    MPI_Comm comm = MPI_Comm_f2c(*fcomm);
    *init = ChASE_DIST_BLOCK::Initialize<float>(*N, *nev, *nex, *m, *n, H, *ldh, V,
                                    ritzv, *dim0, *dim1, grid_major, comm);   
}


void pzchase_init_(int* N, int* nev, int* nex, int* m, int* n,
                    double _Complex* H, int* ldh, double _Complex* V,
                    double* ritzv, int* dim0, int* dim1, char* grid_major,
                    MPI_Comm* comm, int* init)
{
    *init = ChASE_DIST_BLOCK::Initialize<std::complex<double>>(*N, *nev, *nex, *m, *n, 
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
    *init = ChASE_DIST_BLOCK::Initialize<std::complex<double>>(
        *N, *nev, *nex, *m, *n, reinterpret_cast<std::complex<double>*>(H),
        *ldh, reinterpret_cast<std::complex<double>*>(V), ritzv, *dim0,
        *dim1, grid_major, comm);
}

void pcchase_init_(int* N, int* nev, int* nex, int* m, int* n,
                    float _Complex* H, int* ldh, float _Complex* V,
                    float* ritzv, int* dim0, int* dim1, char* grid_major,
                    MPI_Comm* comm, int* init)
{
    *init = ChASE_DIST_BLOCK::Initialize<std::complex<float>>(*N, *nev, *nex, *m, *n, 
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
    *init = ChASE_DIST_BLOCK::Initialize<std::complex<float>>(
        *N, *nev, *nex, *m, *n, reinterpret_cast<std::complex<float>*>(H),
        *ldh, reinterpret_cast<std::complex<float>*>(V), ritzv, *dim0,
        *dim1, grid_major, comm);
}


void pdchase_finalize_(int* flag) { *flag = ChASE_DIST_BLOCK::Finalize<double>(); }

void pschase_finalize_(int* flag) { *flag = ChASE_DIST_BLOCK::Finalize<float>(); }
void pcchase_finalize_(int* flag) { *flag = ChASE_DIST_BLOCK::Finalize<std::complex<float>>(); }
void pzchase_finalize_(int* flag) { *flag = ChASE_DIST_BLOCK::Finalize<std::complex<double>>(); }

void pdchase_(int* deg, double* tol, char* mode, char* opt, char *qr)
{
    ChASE_DIST_BLOCK::Solve<double>(deg, tol, mode, opt, qr);
}
void pschase_(int* deg, float* tol, char* mode, char* opt, char *qr)
{
    ChASE_DIST_BLOCK::Solve<float>(deg, tol, mode, opt, qr);
}
void pzchase_(int* deg, double* tol, char* mode, char* opt, char *qr)
{
    ChASE_DIST_BLOCK::Solve<std::complex<double>>(deg, tol, mode, opt, qr);
}
void pcchase_(int* deg, float* tol, char* mode, char* opt, char *qr)
{
    ChASE_DIST_BLOCK::Solve<std::complex<float>>(deg, tol, mode, opt, qr);
}


}
