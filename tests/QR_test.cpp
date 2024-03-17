#include <limits>

#include <gtest/gtest.h>

#include "ChASE-MPI/chase_mpi.hpp"
#include "algorithm/performance.hpp"

#include "ChASE-MPI/impl/chase_mpidla_blaslapack_seq.hpp"
#include "ChASE-MPI/impl/chase_mpidla_blaslapack_seq_inplace.hpp"

#include "ChASE-MPI/impl/chase_mpidla_blaslapack.hpp"

#include "util.h"

using namespace chase::mpi;

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

template <typename T>
class QRfixture : public testing::Test {
    protected:
    void SetUp() override {   
    
        properties = new ChaseMpiProperties<T>(N, nev, nex, MPI_COMM_WORLD);
        n = properties->get_n();
        m = properties->get_m();

        properties->get_off(&xoff, &yoff, &xlen, &ylen);
        column_comm = properties->get_col_comm();
        
        MPI_Comm_rank(column_comm, &column_rank);

        H.resize(n*m);
        V1.resize(m*(nev+nex));
        ritzv.resize(nev+nex);

        DLAblaslapack = new ChaseMpiDLABlaslapack<T>(properties, H.data(), m, V1.data(), ritzv.data());
        DLA = new ChaseMpiDLA<T>(properties, DLAblaslapack);

    }

    void TearDown() override {
        delete properties;
        delete DLA;
    }

    ChaseMpiProperties<T>* properties;
    ChaseMpiDLABlaslapack<T>* DLAblaslapack;
    ChaseMpiDLA<T>* DLA;

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
    std::vector<T> H;
    std::vector<T> V1;
    std::vector<chase::Base<T>> ritzv;

    MPI_Comm column_comm;
    int column_rank;
};

typedef ::testing::Types<float, double, std::complex<float>, std::complex<double>> MyTypes;
TYPED_TEST_SUITE(QRfixture, MyTypes);

TYPED_TEST(QRfixture, NumberOfProcs)
{
    ASSERT_EQ(this->DLA->get_nprocs(), 4);    
}

TYPED_TEST(QRfixture, cholQR1)
{
    auto machineEpsilon = MachineEpsilon<TypeParam>::value();

    read_vectors(this->V1.data(), GetFileName<TypeParam>() + "cond_10.bin", this->xoff, this->xlen, this->N, this->nev + this->nex, this->column_rank);
    this->DLA->cholQR1(0);
    auto orth = orthogonality<TypeParam>(this->m, this->nev + this->nex, this->V1.data(), this->column_comm);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon + 1e1);
}

TYPED_TEST(QRfixture, cholQR1BadlyCond)
{
    auto machineEpsilon = MachineEpsilon<TypeParam>::value();

    read_vectors(this->V1.data(), GetFileName<TypeParam>() + "cond_1e4.bin", this->xoff, this->xlen, this->N, this->nev+this->nex, this->column_rank);
    this->DLA->cholQR1(0);
    auto orth = orthogonality<TypeParam>(this->m, this->nev+this->nex, this->V1.data(), this->column_comm);
    EXPECT_GT(orth, machineEpsilon );
    EXPECT_LT(orth, 1.0);
}

TYPED_TEST(QRfixture, cholQR1IllCond)
{
    read_vectors(this->V1.data(), GetFileName<TypeParam>() + "cond_1e8.bin", this->xoff, this->xlen, this->N, this->nev+this->nex, this->column_rank);
    int info = this->DLA->cholQR1(0);
    EXPECT_GT(info, 0);
    EXPECT_LE(info, this->nev + this->nex);
}

TYPED_TEST(QRfixture, cholQR2)
{
    auto machineEpsilon = MachineEpsilon<TypeParam>::value();

    read_vectors(this->V1.data(), GetFileName<TypeParam>() + "cond_1e4.bin", this->xoff, this->xlen, this->N, this->nev+this->nex, this->column_rank);
    this->DLA->cholQR2(0);
    auto orth = orthogonality<TypeParam>(this->m, this->nev + this->nex, this->V1.data(), this->column_comm);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon + 1e1);
}

TYPED_TEST(QRfixture, cholQR2IllCond)
{
    read_vectors(this->V1.data(), GetFileName<TypeParam>() + "cond_1e8.bin", this->xoff, this->xlen, this->N, this->nev+this->nex, this->column_rank);
    int info = this->DLA->cholQR2(0);
    EXPECT_GT(info, 0);
    EXPECT_LE(info, this->nev + this->nex);
}

TYPED_TEST(QRfixture, scholQR)
{
    auto machineEpsilon = MachineEpsilon<TypeParam>::value();

    read_vectors(this->V1.data(), GetFileName<TypeParam>() + "cond_1e8.bin", this->xoff, this->xlen, this->N, this->nev+this->nex, this->column_rank);
    int info = this->DLA->shiftedcholQR2(0);
    std::cout << "SINFO: " << info << "\n";
    auto orth = orthogonality<TypeParam>(this->m, this->nev + this->nex, this->V1.data(), this->column_comm);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon + 1e1);
}