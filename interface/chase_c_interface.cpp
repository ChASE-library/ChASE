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

}
