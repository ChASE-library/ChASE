// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <gtest/gtest.h>
#include <complex>
#include <cmath>
#include <cstring>
#include <random>
#include "linalg/internal/mpi/pseudo_hermitian_lanczos.hpp"
#include "tests/linalg/internal/utils.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/matrix/matrix.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"

template <typename T>
class PseudoHermitianLanczosCPUDistTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);  
        
	ritzv_tiny.resize(M_tiny * numvec_tiny);
        ritzV_tiny.resize(M_tiny * M_tiny);
        Tau_tiny.resize(M_tiny * M_tiny * numvec_tiny);       

	ritzv.resize(M * numvec);
        ritzV.resize(M * M);
        Tau.resize(M * M * numvec);        
    }

    void TearDown() override {        
    }

    int world_rank;
    int world_size;

    std::size_t M_tiny = 10;
    std::size_t numvec_tiny = 1;
    std::size_t N_tiny = 10;
    std::vector<chase::Base<T>> ritzv_tiny;
    std::vector<chase::Base<T>> ritzV_tiny;
    std::vector<chase::Base<T>> Tau_tiny;         
    
    std::size_t M = 200;
    std::size_t numvec = 1;
    std::size_t N = 200;
    std::vector<chase::Base<T>> ritzv;
    std::vector<chase::Base<T>> ritzV;
    std::vector<chase::Base<T>> Tau;         
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(PseudoHermitianLanczosCPUDistTest, TestTypes);


TYPED_TEST(PseudoHermitianLanczosCPUDistTest, tinyPseudoHermitianLanczos){
    using T = TypeParam;  // Get the current type

    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    int *coords = mpi_grid.get()->get_coords();

    auto H_ = chase::distMatrix::PseudoHermitianBlockBlockMatrix<T>(this->N_tiny, this->N_tiny, mpi_grid);
    H_.readFromBinaryFile(GetBSE_TinyMatrix<T>());

    chase::matrix::Matrix<T> exact_eigsl_H = chase::matrix::Matrix<T>(this->N_tiny, 1);
    exact_eigsl_H.readFromBinaryFile(GetBSE_TinyEigs<T>());

    auto V_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(this->N_tiny, this->M_tiny, mpi_grid);

    std::mt19937 gen(1337.0 + coords[0]);
    std::normal_distribution<> d;

    for (auto j = 0; j < V_.l_rows() * V_.l_cols(); j++)
    {
        auto rnd = getRandomT<T>([&]() { return d(gen); });
        V_.l_data()[j] = rnd;
    }

    std::size_t xoff, xlen, yoff, ylen;

    chase::Base<T> upperb;

    chase::linalg::internal::cpu_mpi::lanczos_dispatch(this->M_tiny,
                                          this->numvec_tiny,
                                          H_,
                                          V_,
                                          &upperb,
                                          this->ritzv_tiny.data(),
                                          this->Tau_tiny.data(),
                                          this->ritzV_tiny.data());

    chase::Base<T> diff_min = std::norm(this->ritzv_tiny[0]   - std::real(exact_eigsl_H.data()[0]));
    chase::Base<T> diff_max = std::norm(this->ritzv_tiny[this->M_tiny-1] - std::real(exact_eigsl_H.data()[this->N_tiny-1]));

    EXPECT_LT(diff_min,1e3*MachineEpsilon<chase::Base<T>>::value());
    EXPECT_LT(diff_max,1e3*MachineEpsilon<chase::Base<T>>::value());
}

TYPED_TEST(PseudoHermitianLanczosCPUDistTest, tinyPseudoHermitianSimplifiedLanczos) {
    using T = TypeParam;

    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    int *coords = mpi_grid.get()->get_coords();

    auto H_ = chase::distMatrix::PseudoHermitianBlockBlockMatrix<T>(this->N_tiny, this->N_tiny, mpi_grid);
    H_.readFromBinaryFile(GetBSE_TinyMatrix<T>());

    chase::matrix::Matrix<T> exact_eigsl_H = chase::matrix::Matrix<T>(this->N_tiny, 1);
    exact_eigsl_H.readFromBinaryFile(GetBSE_TinyEigs<T>());

    auto V_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(this->N_tiny, this->M_tiny, mpi_grid);

    std::mt19937 gen(1337.0 + coords[0]);
    std::normal_distribution<> d;

    for (auto j = 0; j < V_.l_rows() * V_.l_cols(); j++)
    {
        auto rnd = getRandomT<T>([&]() { return d(gen); });
        V_.l_data()[j] = rnd;
    }

    chase::Base<T> upperb;

    chase::linalg::internal::cpu_mpi::lanczos_dispatch(this->M_tiny, H_, V_, &upperb);
    
    chase::Base<T> actual_upperb = (chase::Base<T>)std::real(exact_eigsl_H.data()[this->N_tiny-1]);
    EXPECT_TRUE(upperb >=  actual_upperb ||
                std::abs(upperb - actual_upperb) / std::abs(actual_upperb) <= 1e-2)
        << "Value " << upperb << " is below the acceptable threshold of "
        << actual_upperb << "\n" ;

    EXPECT_LT(upperb,5*(chase::Base<T>)std::real(exact_eigsl_H.data()[this->N_tiny-1]));

}

TYPED_TEST(PseudoHermitianLanczosCPUDistTest, PseudoHermitianLanczos){
    using T = TypeParam;  // Get the current type

    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    int *coords = mpi_grid.get()->get_coords();

    auto H_ = chase::distMatrix::PseudoHermitianBlockBlockMatrix<T>(this->N, this->N, mpi_grid);
    H_.readFromBinaryFile(GetBSE_Matrix<T>());

    chase::matrix::Matrix<T> exact_eigsl_H = chase::matrix::Matrix<T>(this->N, 1);
    exact_eigsl_H.readFromBinaryFile(GetBSE_Eigs<T>());

    auto V_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(this->N, this->M, mpi_grid);

    std::mt19937 gen(1337.0 + coords[0]);
    std::normal_distribution<> d;

    for (auto j = 0; j < V_.l_rows() * V_.l_cols(); j++)
    {
        auto rnd = getRandomT<T>([&]() { return d(gen); });
        V_.l_data()[j] = rnd;
    }

    std::size_t xoff, xlen, yoff, ylen;

    chase::Base<T> upperb;

    chase::linalg::internal::cpu_mpi::lanczos_dispatch(this->M,
                                          this->numvec,
                                          H_,
                                          V_,
                                          &upperb,
                                          this->ritzv.data(),
                                          this->Tau.data(),
                                          this->ritzV.data());

    chase::Base<T> diff_min = std::norm(this->ritzv[0]   - std::real(exact_eigsl_H.data()[0]));
    chase::Base<T> diff_max = std::norm(this->ritzv[this->M-1] - std::real(exact_eigsl_H.data()[this->N-1]));

    EXPECT_LT(diff_min,1e3*MachineEpsilon<chase::Base<T>>::value());
    EXPECT_LT(diff_max,1e3*MachineEpsilon<chase::Base<T>>::value());
}

TYPED_TEST(PseudoHermitianLanczosCPUDistTest, PseudoHermitianSimplifiedLanczos) {
    using T = TypeParam;

    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    int *coords = mpi_grid.get()->get_coords();

    auto H_ = chase::distMatrix::PseudoHermitianBlockBlockMatrix<T>(this->N, this->N, mpi_grid);
    H_.readFromBinaryFile(GetBSE_Matrix<T>());

    chase::matrix::Matrix<T> exact_eigsl_H = chase::matrix::Matrix<T>(this->N, 1);
    exact_eigsl_H.readFromBinaryFile(GetBSE_Eigs<T>());

    auto V_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(this->N, this->M, mpi_grid);

    std::mt19937 gen(1337.0 + coords[0]);
    std::normal_distribution<> d;

    for (auto j = 0; j < V_.l_rows() * V_.l_cols(); j++)
    {
        auto rnd = getRandomT<T>([&]() { return d(gen); });
        V_.l_data()[j] = rnd;
    }

    chase::Base<T> upperb;

    chase::linalg::internal::cpu_mpi::lanczos_dispatch(this->M, H_, V_, &upperb);
    
    chase::Base<T> actual_upperb = (chase::Base<T>)std::real(exact_eigsl_H.data()[this->N-1]);
    EXPECT_TRUE(upperb >=  actual_upperb ||
                std::abs(upperb - actual_upperb) / std::abs(actual_upperb) <= 1e-2)
        << "Value " << upperb << " is below the acceptable threshold of "
        << actual_upperb << "\n" ;

    EXPECT_LT(upperb,5*(chase::Base<T>)std::real(exact_eigsl_H.data()[this->N-1]));

}

