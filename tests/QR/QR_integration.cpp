#include <limits>
#include <type_traits>

#include <gtest/gtest.h>
#include "gmock/gmock.h"

#include "ChASE-MPI/chase_mpi.hpp"
#include "algorithm/performance.hpp"

#include "ChASE-MPI/impl/chase_mpidla_blaslapack_seq.hpp"
#include "ChASE-MPI/impl/chase_mpidla_blaslapack_seq_inplace.hpp"

#include "ChASE-MPI/impl/chase_mpidla_blaslapack.hpp"

#include "util.h"

using namespace chase::mpi;

template <typename T>
class MockChaseMpiDLA : public chase::mpi::ChaseMpiDLAInterface<T> {
public:
    MockChaseMpiDLA(chase::mpi::ChaseMpiProperties<T>* properties, chase::mpi::ChaseMpiDLAInterface<T>* DLAblaslapack) {}
    //MockChaseMpiDLA(ChaseMpiProperties<T>* matrix_properties, T* H,
    //                      std::size_t ldh, T* V1, chase::Base<T>* ritzv) :
    //                      matrices_(std::move(
    //          matrix_properties->create_matrices(0, H, ldh, V1, ritzv)))
    //{
    //    ON_CALL(*this, getChaseMatrices)
    //    .WillByDefault( [this]{
    //            return &matrices_;
    //    });
    //}
    MOCK_METHOD(void, shiftMatrix, (T, bool), (override));
    MOCK_METHOD(void, preApplication, (T*, std::size_t, std::size_t), (override));
    MOCK_METHOD(void, initVecs, (), (override));
    MOCK_METHOD(void, initRndVecs, (), (override));
    MOCK_METHOD(void, apply, (T, T, std::size_t, std::size_t, std::size_t), (override));
    MOCK_METHOD(void, asynCxHGatherC, (std::size_t, std::size_t, bool), (override));
    MOCK_METHOD(void, Swap, (std::size_t, std::size_t), (override));
    MOCK_METHOD(void, applyVec, (T*, T*, std::size_t), (override));
    MOCK_METHOD(int, get_nprocs, (), (const, override));
    MOCK_METHOD(chase::Base<T>*, get_Resids, (), (override));
    MOCK_METHOD(chase::Base<T>*, get_Ritzv, (), (override));
    MOCK_METHOD(void, Start, (), (override));
    MOCK_METHOD(void, End, (), (override));
    MOCK_METHOD(bool, checkSymmetryEasy, (), (override));
    MOCK_METHOD(void, symOrHermMatrix, (char), (override));
    MOCK_METHOD(void, axpy, (std::size_t, T*, T*, std::size_t, T*, std::size_t), (override));
    MOCK_METHOD(void, scal, (std::size_t, T*, T*, std::size_t), (override));
    MOCK_METHOD(chase::Base<T>, nrm2, (std::size_t, T*, std::size_t), (override));
    MOCK_METHOD(T, dot, (std::size_t, T*, std::size_t, T*, std::size_t), (override));
    MOCK_METHOD(void, RR, (std::size_t, std::size_t, chase::Base<T>*), (override));
    MOCK_METHOD(void, syherk, (char, char, std::size_t, std::size_t, T*, T*, std::size_t, T*, T*, std::size_t, bool), (override));
    MOCK_METHOD(int, potrf, (char, std::size_t, T*, std::size_t, bool), (override));
    MOCK_METHOD(void, trsm, (char, char, char, char, std::size_t, std::size_t, T*, T*, std::size_t, T*, std::size_t, bool), (override));
    MOCK_METHOD(void, heevd, (int, char, char, std::size_t, T*, std::size_t, chase::Base<T>*), (override));
    MOCK_METHOD(void, Resd, (chase::Base<T>*, chase::Base<T>*, std::size_t, std::size_t), (override));
    MOCK_METHOD(void, hhQR, (std::size_t), (override));
    MOCK_METHOD(int, cholQR1, (std::size_t), (override));
    MOCK_METHOD(int, cholQR2, (std::size_t), (override));
    MOCK_METHOD(int, shiftedcholQR2, (std::size_t), (override));
    MOCK_METHOD(void, estimated_cond_evaluator, (std::size_t, chase::Base<T>), (override));
    MOCK_METHOD(void, lockVectorCopyAndOrthoConcatswap, (std::size_t, bool), (override));
    MOCK_METHOD(void, LanczosDos, (std::size_t, std::size_t, T*), (override));
    MOCK_METHOD(void, Lanczos, (std::size_t, chase::Base<T>*), (override));
    MOCK_METHOD(void, mLanczos, (std::size_t, int, chase::Base<T>*, chase::Base<T>*, chase::Base<T>*), (override));
    MOCK_METHOD(void, B2C, (T*, std::size_t, T*, std::size_t, std::size_t), (override));
    MOCK_METHOD(void, lacpy, (char, std::size_t, std::size_t, T*, std::size_t, T*, std::size_t), (override));
    MOCK_METHOD(void, shiftMatrixForQR, (T*, std::size_t, T), (override));
    MOCK_METHOD(chase::mpi::ChaseMpiMatrices<T>*, getChaseMatrices, (), (override));
    MOCK_METHOD(void, computeDiagonalAbsSum, (T *, chase::Base<T> *, std::size_t, std::size_t), (override));
};

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
        mockdla = new MockChaseMpiDLA<T>(properties, DLAblaslapack);
        chasempi = new ChaseMpi<MockChaseMpiDLA, T>(properties, H.data(), m, V1.data(), ritzv.data(), mockdla);
    }

    void TearDown() override {
        delete chasempi;
        delete DLAblaslapack;
    }

    ChaseMpiProperties<T>* properties;
    ChaseMpiDLABlaslapack<T>* DLAblaslapack;
    MockChaseMpiDLA<T>* mockdla;
    ChaseMpi<MockChaseMpiDLA, T>* chasempi;

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
    ASSERT_EQ(this->chasempi->get_nprocs(), MPI_NUM_PROCS);    
}

TYPED_TEST(QRfixture, TestDisableIsOne) {
    setenv("CHASE_DISABLE_CHOLQR", "1", 1);
    EXPECT_CALL(*(this->mockdla), hhQR(::testing::_)).Times(1);  
    this->chasempi->QR(0,0);
}

TYPED_TEST(QRfixture, TestShiftCholQRBasedonCond) {
    setenv("CHASE_DISABLE_CHOLQR", "0", 1);

    EXPECT_CALL(*(this->mockdla), shiftedcholQR2(::testing::_)).Times(1);
    if (std::is_same<TypeParam, double>::value || std::is_same<TypeParam, std::complex<double>>::value) {
        this->chasempi->QR(0, 1e10);
    } else if (std::is_same<TypeParam, float>::value || std::is_same<TypeParam, std::complex<float>>::value) {
        this->chasempi->QR(0, 1e5);
    }
}

TYPED_TEST(QRfixture, TestCholQR2BasedonCond) {
    setenv("CHASE_DISABLE_CHOLQR", "0", 1);

    EXPECT_CALL(*(this->mockdla), cholQR2(::testing::_)).Times(1);
    if (std::is_same<TypeParam, double>::value || std::is_same<TypeParam, std::complex<double>>::value) {
        this->chasempi->QR(0, 1e7);
    } else if (std::is_same<TypeParam, float>::value || std::is_same<TypeParam, std::complex<float>>::value) {
        this->chasempi->QR(0, 1e3);
    }
}

TYPED_TEST(QRfixture, TestCholQR1BasedonCond) {
    setenv("CHASE_DISABLE_CHOLQR", "0", 1);

    EXPECT_CALL(*(this->mockdla), cholQR1(::testing::_)).Times(1);
    if (std::is_same<TypeParam, double>::value || std::is_same<TypeParam, std::complex<double>>::value) {
        this->chasempi->QR(0, 1e1);
    } else if (std::is_same<TypeParam, float>::value || std::is_same<TypeParam, std::complex<float>>::value) {
        this->chasempi->QR(0, 5);
    }
}

TYPED_TEST(QRfixture, TestCholQR1BasedonEnv) {
    setenv("CHASE_CHOLQR1_THLD", "100", 1);

    EXPECT_CALL(*(this->mockdla), cholQR1(::testing::_)).Times(1);
    if (std::is_same<TypeParam, double>::value || std::is_same<TypeParam, std::complex<double>>::value) {
        this->chasempi->QR(0, 50);
    } else if (std::is_same<TypeParam, float>::value || std::is_same<TypeParam, std::complex<float>>::value) {
        this->chasempi->QR(0, 25);
    }
}

TYPED_TEST(QRfixture, TestShiftCholQRFailTohhQR) {
    setenv("CHASE_DISABLE_CHOLQR", "0", 1);

    EXPECT_CALL(*(this->mockdla), shiftedcholQR2(::testing::_)).Times(1).WillOnce(testing::Return(50));
    EXPECT_CALL(*(this->mockdla), hhQR(::testing::_)).Times(1);
    if (std::is_same<TypeParam, double>::value || std::is_same<TypeParam, std::complex<double>>::value) {
        this->chasempi->QR(0, 1e10);
    } else if (std::is_same<TypeParam, float>::value || std::is_same<TypeParam, std::complex<float>>::value) {
        this->chasempi->QR(0, 1e5);
    }
}

TYPED_TEST(QRfixture, TestCholQR2FailTohhQR) {
    setenv("CHASE_DISABLE_CHOLQR", "0", 1);

    EXPECT_CALL(*(this->mockdla), cholQR2(::testing::_)).Times(1).WillOnce(testing::Return(50));
    EXPECT_CALL(*(this->mockdla), hhQR(::testing::_)).Times(1);
    if (std::is_same<TypeParam, double>::value || std::is_same<TypeParam, std::complex<double>>::value) {
        this->chasempi->QR(0, 1e7);
    } else if (std::is_same<TypeParam, float>::value || std::is_same<TypeParam, std::complex<float>>::value) {
        this->chasempi->QR(0, 1e3);
    }
}

TYPED_TEST(QRfixture, TestCholQR1FailTohhQR) {
    setenv("CHASE_DISABLE_CHOLQR", "0", 1);

    EXPECT_CALL(*(this->mockdla), cholQR1(::testing::_)).Times(1).WillOnce(testing::Return(50));
    EXPECT_CALL(*(this->mockdla), hhQR(::testing::_)).Times(1);
    if (std::is_same<TypeParam, double>::value || std::is_same<TypeParam, std::complex<double>>::value) {
        this->chasempi->QR(0, 1e1);
    } else if (std::is_same<TypeParam, float>::value || std::is_same<TypeParam, std::complex<float>>::value) {
        this->chasempi->QR(0, 5);
    }
}

TYPED_TEST(QRfixture, TestEstimatedCondEval) {
    setenv("CHASE_DISABLE_CHOLQR", "0", 1);
    setenv("CHASE_DISPLAY_BOUNDS", "1", 1);

    EXPECT_CALL(*(this->mockdla), estimated_cond_evaluator(::testing::_, ::testing::_)).Times(1);
    this->chasempi->QR(0, 5);
}