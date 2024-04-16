#include "QR_test.hpp"

//typedef ::testing::Types<float, double, std::complex<float>, std::complex<double>> MyTypes;
TYPED_TEST_SUITE(QRfixture, MyTypes);

TYPED_TEST(QRfixture, NumberOfProcs)
{
    ASSERT_EQ(this->DLA->get_nprocs(), MPI_NUM_PROCS);    
}

TYPED_TEST(QRfixture, cholQR1)
{
    using T2 = typename TestFixture::T2;
    auto machineEpsilon = MachineEpsilon<T2>::value();

    read_vectors(this->Matrices->C().host(), GetFileName<T2>() + "cond_10.bin", this->xoff, this->xlen, this->N, this->nev + this->nex, this->column_rank);
    this->Matrices->C().syncFromPtr();
    
    int info = this->DLA->cholQR1(0);
    ASSERT_EQ(info, 0);
    this->Matrices->C().sync2Ptr();

    auto orth = orthogonality<T2>(this->m, this->nev + this->nex, this->Matrices->C().host(), this->column_comm);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 10);
}

TYPED_TEST(QRfixture, cholQR1BadlyCond)
{
    using T2 = typename TestFixture::T2;
    auto machineEpsilon = MachineEpsilon<T2>::value();

    read_vectors(this->Matrices->C().host(), GetFileName<T2>() + "cond_1e4.bin", this->xoff, this->xlen, this->N, this->nev+this->nex, this->column_rank);
    this->Matrices->C().syncFromPtr();
    
    int info = this->DLA->cholQR1(0);
    ASSERT_EQ(info, 0);
    this->Matrices->C().sync2Ptr();

    auto orth = orthogonality<T2>(this->m, this->nev+this->nex, this->Matrices->C().host(), this->column_comm);
    EXPECT_GT(orth, machineEpsilon );
    EXPECT_LT(orth, 1.0);
}

TYPED_TEST(QRfixture, cholQR1IllCond)
{
    using T2 = typename TestFixture::T2;
    read_vectors(this->Matrices->C().host(), GetFileName<T2>() + "cond_ill.bin", this->xoff, this->xlen, this->N, this->nev+this->nex, this->column_rank);
    this->Matrices->C().syncFromPtr();
    
    int info = this->DLA->cholQR1(0);
    EXPECT_GT(info, 0);
    EXPECT_LE(info, this->nev + this->nex);
}

TYPED_TEST(QRfixture, cholQR2)
{
    using T2 = typename TestFixture::T2;
    auto machineEpsilon = MachineEpsilon<T2>::value();

    read_vectors(this->Matrices->C().host(), GetFileName<T2>() + "cond_1e4.bin", this->xoff, this->xlen, this->N, this->nev+this->nex, this->column_rank);
    this->Matrices->C().syncFromPtr();
    
    int info = this->DLA->cholQR2(0);
    ASSERT_EQ(info, 0);
    this->Matrices->C().sync2Ptr();

    auto orth = orthogonality<T2>(this->m, this->nev + this->nex, this->Matrices->C().host(), this->column_comm);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon *10);
}

TYPED_TEST(QRfixture, cholQR2IllCond)
{
    using T2 = typename TestFixture::T2;
    read_vectors(this->Matrices->C().host(), GetFileName<T2>() + "cond_ill.bin", this->xoff, this->xlen, this->N, this->nev+this->nex, this->column_rank);
    this->Matrices->C().syncFromPtr();
    
    int info = this->DLA->cholQR2(0);
    EXPECT_GT(info, 0);
    EXPECT_LE(info, this->nev + this->nex);
}

TYPED_TEST(QRfixture, scholQR)
{
    using T2 = typename TestFixture::T2;
    auto machineEpsilon = MachineEpsilon<T2>::value();

    read_vectors(this->Matrices->C().host(), GetFileName<T2>() + "cond_ill.bin", this->xoff, this->xlen, this->N, this->nev+this->nex, this->column_rank);
    this->Matrices->C().syncFromPtr();

    int info = this->DLA->shiftedcholQR2(0);
    ASSERT_EQ(info, 0);
    this->Matrices->C().sync2Ptr();

    auto orth = orthogonality<T2>(this->m, this->nev + this->nex, this->Matrices->C().host(), this->column_comm);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 10);
}
