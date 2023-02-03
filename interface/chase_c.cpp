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
#include <chrono>
#include <complex>
#include <complex.h>
#include <fstream>
#include <mpi.h>
#include <random>
#include <sys/stat.h>
#include <algorithm>

#ifdef INTERFACE_WITH_MGPU
#include "ChASE-MPI/impl/chase_mpidla_mgpu.hpp"
#include "ChASE-MPI/impl/chase_mpidla_cuda_seq.hpp"
#endif

using namespace chase;
using namespace chase::mpi;

#ifdef INTERFACE_WITH_MGPU
using dlaSeq = ChaseMpiDLACudaSeq<T>;
#else
template<typename T>
using dlaSeq = ChaseMpiDLABlaslapackSeqInplace<T>;
#endif

class ChASE_SEQ{
    public:
         template<typename T>
         static ChaseMpi<dlaSeq,T> *Initialize(int N, int nev, int nex, T *H, T *V, Base<T> *ritzv);
         
         template<typename T>
         static void Finalize();

         template<typename T>
         static ChaseMpi<dlaSeq,T> *getChase();

         static ChaseMpi<dlaSeq,double> *dchaseSeq;  
         static ChaseMpi<dlaSeq,float>  *schaseSeq;
         static ChaseMpi<dlaSeq,std::complex<double>> *zchaseSeq;  
         static ChaseMpi<dlaSeq,std::complex<float>> *cchaseSeq;  
    
};

ChaseMpi<dlaSeq,double> *ChASE_SEQ::dchaseSeq = nullptr;
ChaseMpi<dlaSeq,float> *ChASE_SEQ::schaseSeq = nullptr;
ChaseMpi<dlaSeq,std::complex<double>> *ChASE_SEQ::zchaseSeq = nullptr;
ChaseMpi<dlaSeq,std::complex<float>> *ChASE_SEQ::cchaseSeq = nullptr;

template<>
ChaseMpi<dlaSeq,double> *ChASE_SEQ::Initialize(int N, int nev, int nex, double *H, double *V, double *ritzv){
    dchaseSeq = new ChaseMpi<dlaSeq,double>(N, nev, nex, V, ritzv, H);
    return dchaseSeq;
}

template<>
ChaseMpi<dlaSeq,float> *ChASE_SEQ::Initialize(int N, int nev, int nex, float *H, float *V, float *ritzv){
    schaseSeq = new ChaseMpi<dlaSeq,float>(N, nev, nex, V, ritzv, H);
    return schaseSeq;
}


template<>
ChaseMpi<dlaSeq,std::complex<double>> *ChASE_SEQ::Initialize(int N, int nev, int nex, std::complex<double> *H, std::complex<double> *V, double *ritzv){
    zchaseSeq = new ChaseMpi<dlaSeq,std::complex<double>>(N, nev, nex, V, ritzv, H);
    return zchaseSeq;
}

template<>
ChaseMpi<dlaSeq,std::complex<float>> *ChASE_SEQ::Initialize(int N, int nev, int nex, std::complex<float> *H, std::complex<float> *V, float *ritzv){
    cchaseSeq = new ChaseMpi<dlaSeq,std::complex<float>>(N, nev, nex, V, ritzv, H);
    return cchaseSeq;
}

template<>
void ChASE_SEQ::Finalize<double>(){
    delete dchaseSeq;
}

template<>
void ChASE_SEQ::Finalize<float>(){
    delete schaseSeq;
}

template<>
void ChASE_SEQ::Finalize<std::complex<float>>(){
    delete cchaseSeq;
}

template<>
void ChASE_SEQ::Finalize<std::complex<double>>(){
    delete zchaseSeq;
}

template<>
ChaseMpi<dlaSeq,double> *ChASE_SEQ::getChase(){
    return dchaseSeq;
}

template<>
ChaseMpi<dlaSeq,float> *ChASE_SEQ::getChase(){
    return schaseSeq;
}

template<>
ChaseMpi<dlaSeq,std::complex<float>> *ChASE_SEQ::getChase(){
    return cchaseSeq;
}


template<>
ChaseMpi<dlaSeq,std::complex<double>> *ChASE_SEQ::getChase(){
    return zchaseSeq;
}

template<typename T>
int ChASE_SEQ_Init(int N, int nev, int nex, T *H, T *V, Base<T> *ritzv){
    auto single = ChASE_SEQ::Initialize<T>(N, nev, nex, H, V, ritzv);
    return 1;
}

template<typename T>
int ChASE_SEQ_Finalize(){
    ChASE_SEQ::Finalize<T>();
    return 0;
}

template<typename T>
void ChASE_SEQ_Solve(int* deg, Base<T>* tol, char* mode, char* opt){
    ChaseMpi<dlaSeq,T> *single = ChASE_SEQ::getChase<T>();

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
    Base<T> *ritzv = single->GetRitzv();
    Base<T> *resid = single->GetResid();

    if(rank == 0){
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
    int dchase_init_(int *N, int *nev, int *nex, double *H, double *V, double *ritzv){
        return ChASE_SEQ_Init<double>(*N, *nev, *nex, H, V, ritzv);
    } 
    int schase_init_(int *N, int *nev, int *nex, float *H, float *V, float *ritzv){
        return ChASE_SEQ_Init<float>(*N, *nev, *nex, H, V, ritzv);
    } 
    int cchase_init_(int *N, int *nev, int *nex, float _Complex *H, float _Complex *V, float *ritzv){
        return ChASE_SEQ_Init<std::complex<float>>(*N, *nev, *nex, reinterpret_cast<std::complex<float> *>(H), reinterpret_cast<std::complex<float> *>(V), ritzv);
    } 
    int zchase_init_(int *N, int *nev, int *nex, double _Complex *H, double _Complex *V, double *ritzv){
        return ChASE_SEQ_Init<std::complex<double>>(*N, *nev, *nex, reinterpret_cast<std::complex<double> *>(H), reinterpret_cast<std::complex<double> *>(V), ritzv);
    } 

    int dchase_finalize_(){
        return ChASE_SEQ_Finalize<double>();
    }
    int schase_finalize_(){
        return ChASE_SEQ_Finalize<float>();
    }
    int cchase_finalize_(){
        return ChASE_SEQ_Finalize<std::complex<float>>();
    }
    int zchase_finalize_(){
        return ChASE_SEQ_Finalize<std::complex<double>>();
    }

    void dchase_(int* deg, double* tol, char* mode, char* opt){
        ChASE_SEQ_Solve<double>(deg, tol, mode, opt);   
    }
    void schase_(int* deg, float* tol, char* mode, char* opt){
        ChASE_SEQ_Solve<float>(deg, tol, mode, opt);   
    }
    void zchase_(int* deg, double* tol, char* mode, char* opt){
        ChASE_SEQ_Solve<std::complex<double>>(deg, tol, mode, opt);   
    }
    void cchase_(int* deg, float* tol, char* mode, char* opt){
        ChASE_SEQ_Solve<std::complex<float>>(deg, tol, mode, opt);   
    }

}  // extern C 