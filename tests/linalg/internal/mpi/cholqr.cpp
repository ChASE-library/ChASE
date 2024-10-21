#include <gtest/gtest.h>
#include <complex>
#include <cmath>
#include <cstring>
#include "linalg/internal/mpi/cholqr.hpp"
#include "tests/linalg/internal/mpi/TestConditions.hpp"
#include "tests/linalg/internal/utils.hpp"
#include "grid/mpiGrid2D.hpp"

template <typename T>
class CHOLQRCPUDistTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);             
    }

    void TearDown() override {      }

    int world_rank;
    int world_size;    
    std::size_t N = 100;
    std::size_t n = 50;
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(CHOLQRCPUDistTest, TestTypes);


TYPED_TEST(CHOLQRCPUDistTest, cholQR1) {
    using T = TypeParam;  // Get the current type
    assert(this->world_size == 4);

    std::size_t xlen = this->N / this->world_size;
    std::size_t xoff = this->world_rank * 25;

    std::vector<T> V(xlen * this->n);

    auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(V.data(), GetQRFileName<T>() + "cond_10.bin", xoff, xlen, this->N, this->n, 0);
    int info = chase::linalg::internal::mpi::cholQR1<T>(xlen, this->n, V.data(), xlen, MPI_COMM_WORLD);
    ASSERT_EQ(info, 0);
    auto orth = orthogonality<T>(xlen, this->n, V.data(), xlen, MPI_COMM_WORLD);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 15);

}

TYPED_TEST(CHOLQRCPUDistTest, cholQR1BadlyCond) {
    using T = TypeParam;  // Get the current type
    assert(this->world_size == 4);

    std::size_t xlen = this->N / this->world_size;
    std::size_t xoff = this->world_rank * 25;

    std::vector<T> V(xlen * this->n);
    auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(V.data(), GetQRFileName<T>() + "cond_1e4.bin", xoff, xlen, this->N, this->n, 0);
    int info = chase::linalg::internal::mpi::cholQR1<T>(xlen, this->n, V.data(), xlen, MPI_COMM_WORLD);
    ASSERT_EQ(info, 0);
    auto orth = orthogonality<T>(xlen, this->n, V.data(), xlen, MPI_COMM_WORLD);
    EXPECT_GT(orth, machineEpsilon );
    EXPECT_LT(orth, 1.0);
}

TYPED_TEST(CHOLQRCPUDistTest, cholQR1IllCond) {
    using T = TypeParam;  // Get the current type
    assert(this->world_size == 4);

    std::size_t xlen = this->N / this->world_size;
    std::size_t xoff = this->world_rank * 25;

    std::vector<T> V(xlen * this->n);
    auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(V.data(), GetQRFileName<T>() + "cond_ill.bin", xoff, xlen, this->N, this->n, 0);
    int info = chase::linalg::internal::mpi::cholQR1<T>(xlen, this->n, V.data(), xlen, MPI_COMM_WORLD);
    EXPECT_GT(info, 0);
    EXPECT_LE(info, this->n);
}

TYPED_TEST(CHOLQRCPUDistTest, cholQR2) {
    using T = TypeParam;  // Get the current type
    assert(this->world_size == 4);

    std::size_t xlen = this->N / this->world_size;
    std::size_t xoff = this->world_rank * 25;

    std::vector<T> V(xlen * this->n);
    auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(V.data(), GetQRFileName<T>() + "cond_1e4.bin", xoff, xlen, this->N, this->n, 0);
    int info = chase::linalg::internal::mpi::cholQR2<T>(xlen, this->n, V.data(), xlen, MPI_COMM_WORLD);
    ASSERT_EQ(info, 0);
    auto orth = orthogonality<T>(xlen, this->n, V.data(), xlen, MPI_COMM_WORLD);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 15);
}

TYPED_TEST(CHOLQRCPUDistTest, cholQR2IllCond) {
    using T = TypeParam;  // Get the current type
    assert(this->world_size == 4);

    std::size_t xlen = this->N / this->world_size;
    std::size_t xoff = this->world_rank * 25;

    std::vector<T> V(xlen * this->n);
    auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(V.data(), GetQRFileName<T>() + "cond_ill.bin", xoff, xlen, this->N, this->n, 0);
    int info = chase::linalg::internal::mpi::cholQR2<T>(xlen, this->n, V.data(), xlen, MPI_COMM_WORLD);
    EXPECT_GT(info, 0);
    EXPECT_LE(info, this->n);
}

TYPED_TEST(CHOLQRCPUDistTest, scholQR) {
    using T = TypeParam;  // Get the current type
    assert(this->world_size == 4);

    std::size_t xlen = this->N / this->world_size;
    std::size_t xoff = this->world_rank * 25;

    std::vector<T> V(xlen * this->n);

    auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(V.data(), GetQRFileName<T>() + "cond_ill.bin", xoff, xlen, this->N, this->n, 0);
    int info = chase::linalg::internal::mpi::shiftedcholQR2<T>(this->N, xlen, this->n, V.data(), xlen, MPI_COMM_WORLD);
    ASSERT_EQ(info, 0);
    auto orth = orthogonality<T>(xlen, this->n, V.data(), xlen, MPI_COMM_WORLD);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 10);
}

#ifdef HAS_SCALAPACK 
TYPED_TEST(CHOLQRCPUDistTest, scalapackHHQR) {
    using T = TypeParam;  // Get the current type
    assert(this->world_size == 4);
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(4, 1, MPI_COMM_WORLD);

    auto V_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(this->N, this->n, mpi_grid);

    std::size_t xlen = this->N / this->world_size;
    std::size_t xoff = this->world_rank * 25;

    auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(V_.l_data(), GetQRFileName<T>() + "cond_ill.bin", xoff, xlen, this->N, this->n, 0);
    chase::linalg::internal::mpi::houseHoulderQR(V_);
    auto orth = orthogonality<T>(xlen, this->n, V_.l_data(), xlen, MPI_COMM_WORLD);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 10);
}
#endif