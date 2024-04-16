#pragma once

#include <limits>

#include <gtest/gtest.h>

#include "ChASE-MPI/chase_mpi.hpp"
#include "algorithm/performance.hpp"

#ifdef SERIAL_TEST
#ifdef HAS_CUDA
#include "ChASE-MPI/impl/chase_mpidla_cuda_seq.hpp"
#else
#include "ChASE-MPI/impl/chase_mpidla_blaslapack_seq.hpp"
#include "ChASE-MPI/impl/chase_mpidla_blaslapack_seq_inplace.hpp"
#endif
#elif defined(HAS_CUDA)
#include "ChASE-MPI/impl/chase_mpidla_mgpu.hpp"
#elif defined(USE_MPI)
#include "ChASE-MPI/impl/chase_mpidla_blaslapack.hpp"
#endif


#include "util.h"

using namespace chase::mpi;


typedef ::testing::Types<
#ifdef SERIAL_TEST
#ifdef HAS_CUDA
    std::tuple<ChaseMpiDLACudaSeq<float>, float>,
    std::tuple<ChaseMpiDLACudaSeq<double>, double>,
    std::tuple<ChaseMpiDLACudaSeq<std::complex<float>>, std::complex<float>>,
    std::tuple<ChaseMpiDLACudaSeq<std::complex<double>>, std::complex<double>>
#else
    std::tuple<ChaseMpiDLABlaslapackSeqInplace<float>, float>,
    std::tuple<ChaseMpiDLABlaslapackSeqInplace<double>, double>,
    std::tuple<ChaseMpiDLABlaslapackSeqInplace<std::complex<float>>, std::complex<float>>,
    std::tuple<ChaseMpiDLABlaslapackSeqInplace<std::complex<double>>, std::complex<double>>,
    std::tuple<ChaseMpiDLABlaslapackSeq<float>, float>,
    std::tuple<ChaseMpiDLABlaslapackSeq<double>, double>,
    std::tuple<ChaseMpiDLABlaslapackSeq<std::complex<float>>, std::complex<float>>,
    std::tuple<ChaseMpiDLABlaslapackSeq<std::complex<double>>, std::complex<double>>    
#endif
#elif defined(HAS_CUDA)
    std::tuple<ChaseMpiDLAMultiGPU<float>, float>,
    std::tuple<ChaseMpiDLAMultiGPU<double>, double>,
    std::tuple<ChaseMpiDLAMultiGPU<std::complex<float>>, std::complex<float>>,
    std::tuple<ChaseMpiDLAMultiGPU<std::complex<double>>, std::complex<double>>
#elif defined(USE_MPI)
    std::tuple<ChaseMpiDLABlaslapack<float>, float>,
    std::tuple<ChaseMpiDLABlaslapack<double>, double>,
    std::tuple<ChaseMpiDLABlaslapack<std::complex<float>>, std::complex<float>>,
    std::tuple<ChaseMpiDLABlaslapack<std::complex<double>>, std::complex<double>>
#endif
> MyTypes;

template <typename T>
chase::Base<T> orthogonality(std::size_t m, std::size_t nevex, T* C, MPI_Comm comm)
{
    T one = T(1.0);
    T negone = T(-1.0);
    T zero = T(0.0);

    std::vector<T> A(nevex*nevex);
    
    t_gemm<T>(1, CblasConjTrans, CblasNoTrans,
              nevex, nevex, m, 
              &one, C, m, C, m, 
              &zero, A.data(), nevex);

    MPI_Allreduce(MPI_IN_PLACE, A.data(), nevex*nevex, getMPI_Type<T>(), MPI_SUM, comm);

    std::size_t incx = 1, incy = nevex+1;
    std::vector<T> hI(nevex, T(1.0));

    t_axpy(nevex, &negone, hI.data(), incx, A.data(), incy);

    chase::Base<T> nrmf = t_nrm2(nevex * nevex, A.data(), 1);
    return (nrmf / std::sqrt(nevex));
}

template <class T>
class QRfixture : public testing::Test {
    protected:
        typedef typename std::tuple_element<0, T>::type MF;
        typedef typename std::tuple_element<1, T>::type T2;

    void SetUp() override {   
#ifdef SERIAL_TEST
        n = N;
        m = N;
        xoff = 0;
        yoff = 0;
        xlen = N;
        ylen = N;
        column_comm = MPI_COMM_SELF;
        MPI_Comm_rank(column_comm, &column_rank);
        
        V1.resize(m*(nev+nex));
        ritzv.resize(nev+nex);

        DLA = new MF(H.data(), N, V1.data(), ritzv.data(), N, nev, nex);
        Matrices = DLA->getChaseMatrices();
#else    
        properties = new ChaseMpiProperties<T2>(N, nev, nex, MPI_COMM_WORLD);
        n = properties->get_n();
        m = properties->get_m();

        properties->get_off(&xoff, &yoff, &xlen, &ylen);
        column_comm = properties->get_col_comm();
        
        MPI_Comm_rank(column_comm, &column_rank);

        V1.resize(m*(nev+nex));
        ritzv.resize(nev+nex);

        DLAblaslapack = new MF(properties, H.data(), m, V1.data(), ritzv.data());
        DLA = new ChaseMpiDLA<T2>(properties, DLAblaslapack);
        Matrices = DLAblaslapack->getChaseMatrices();
#endif
    }

    void TearDown() override {
        //delete DLAblaslapack; // dont have to delete it because DLA will it delete it automaticaly becase DLAblaslapack is unique_ptr inside DLA
        delete DLA;
#ifndef SERIAL_TEST
        delete properties;
#endif        
    }
#ifndef SERIAL_TEST
    ChaseMpiProperties<T2>* properties;
    ChaseMpiDLAInterface<T2>* DLAblaslapack;
#endif
    ChaseMpiDLAInterface<T2>* DLA;
    ChaseMpiMatrices<T2>* Matrices;

    // I need ChaseMpiProperties class for constructor of chase_
    std::size_t N   = 100;
    std::size_t nev = 30;
    std::size_t nex = 20;
    std::size_t n,m;

    std::size_t xoff;
    std::size_t yoff;
    std::size_t xlen;
    std::size_t ylen;

    // memory allocation
    std::vector<T2> H;
    std::vector<T2> V1;
    std::vector<chase::Base<T2>> ritzv;

    MPI_Comm column_comm;
    int column_rank;
};
