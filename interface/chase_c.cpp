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
#include <complex.h>
#include <fstream>
#include <mpi.h>
#include <random>
#include <sys/stat.h>

#ifdef INTERFACE_WITH_MGPU
#include "ChASE-MPI/impl/chase_mpidla_mgpu.hpp"
#include "ChASE-MPI/impl/chase_mpidla_cuda_seq.hpp"
#endif

using namespace chase;
using namespace chase::mpi;

class ChASE_SEQ{
    public:
         static ChaseMpi<ChaseMpiDLABlaslapackSeqInplace,double> *Initialize(int N, int nev, int nex, double *H, double *V, double *ritzv);
         static void  Finalize();

         static ChaseMpi<ChaseMpiDLABlaslapackSeqInplace,double> *getChase();
         static ChaseMpi<ChaseMpiDLABlaslapackSeqInplace,double> *dchaseSeq;      
};

ChaseMpi<ChaseMpiDLABlaslapackSeqInplace,double> *ChASE_SEQ::dchaseSeq = nullptr;

ChaseMpi<ChaseMpiDLABlaslapackSeqInplace,double> * ChASE_SEQ::Initialize(int N, int nev, int nex, double *H, double *V, double *ritzv){
    dchaseSeq = new ChaseMpi<ChaseMpiDLABlaslapackSeqInplace,double> (N, nev, nex, V, ritzv, H);
    return dchaseSeq;
}

void ChASE_SEQ::Finalize(){
    delete dchaseSeq;
}

ChaseMpi<ChaseMpiDLABlaslapackSeqInplace,double> *ChASE_SEQ::getChase(){
    return dchaseSeq;
}

int ChASE_SEQ_Init(int N, int nev, int nex, double *H, double *V, double *ritzv){
    auto single = ChASE_SEQ::Initialize(N, nev, nex, H, V, ritzv);
    return 1;
}

void ChASE_SEQ_Finalize(){
    ChASE_SEQ::Finalize();
}

void ChASE_SEQ_Solve(int* deg, double* tol, char* mode, char* opt){
    ChaseMpi<ChaseMpiDLABlaslapackSeqInplace,double> *single = ChASE_SEQ::getChase();

    ChaseConfig<double>& config = single->GetConfig();
    config.SetTol(*tol);
    config.SetDeg(*deg);
    config.SetOpt(*opt == 'S');
    config.SetApprox(*mode == 'A');

    PerformanceDecoratorChase<double> performanceDecorator(single);
    chase::Solve(&performanceDecorator);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef CHASE_OUTPUT    
    double *ritzv = single->GetRitzv();
    double *resid = single->GetResid();

    if(rank == 0){
        performanceDecorator.GetPerfData().print();
        printf("Printing first 5 eigenvalues and residuals\n");
        printf("|-----------------------------------------|\n");

        for (int i = 0; i < 5; ++i)
            printf("| %.10e | %.15e |\n", ritzv[i], resid[i]);          
        printf("\n\n\n"); 
    }


#endif    
}

extern "C" {
    int chaseSeqInit(int *N, int *nev, int *nex, double *H, double *V, double *ritzv){
        return ChASE_SEQ_Init(*N, *nev, *nex, H, V, ritzv);
    }

    void chaseSeqFinalize(){
        ChASE_SEQ_Finalize();   
    }

    void chaseSeqSolve(int* deg, double* tol, char* mode, char* opt){
        ChASE_SEQ_Solve(deg, tol, mode, opt);
    }

}  // extern C 