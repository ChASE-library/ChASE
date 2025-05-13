// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <gtest/gtest.h>
#include <complex>
#include <cmath>
#include <random>
#include <cstring>
#include "linalg/internal/nccl/hemm.hpp"
#include "linalg/internal/nccl/rayleighRitz.hpp"
#include "tests/linalg/internal/utils.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"

namespace {
    bool resources_initialized = false;
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid;
    cublasHandle_t cublasH;
    cusolverDnHandle_t cusolverH;
    cusolverDnParams_t params;
}

template <typename T>
class QuasiRayleighRitzGPUNCCLDistTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        CHECK_CUBLAS_ERROR(cublasCreate(&cublasH));
        CHECK_CUSOLVER_ERROR(cusolverDnCreate(&cusolverH));
	CHECK_CUSOLVER_ERROR(cusolverDnCreateParams(&params));
    }

    void TearDown() override {
    }

    int world_rank;
    int world_size;     
    
    std::size_t N = 200;
    std::size_t N_tiny = 10;

    static cublasHandle_t get_cublas_handle() { return cublasH; }
    static cusolverDnHandle_t get_cusolver_handle() { return cusolverH; }
    static cusolverDnParams_t get_cusolver_params() { return params; }
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(QuasiRayleighRitzGPUNCCLDistTest, TestTypes);


TYPED_TEST(QuasiRayleighRitzGPUNCCLDistTest, TinyQuasiHermitianRRDistGPUCorrectness) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    int *coords = mpi_grid.get()->get_coords();

    auto H_ = chase::distMatrix::QuasiHermitianBlockBlockMatrix<T, chase::platform::GPU>(this->N_tiny, this->N_tiny, mpi_grid);
    H_.allocate_cpu_data();
    H_.readFromBinaryFile(GetBSE_TinyMatrix<T>());
    H_.H2D();
    
    chase::distMatrix::RedundantMatrix<chase::Base<T>, chase::platform::GPU> ritzv_tiny
	    = chase::distMatrix::RedundantMatrix<chase::Base<T>, chase::platform::GPU>(this->N_tiny, 1, mpi_grid);
    ritzv_tiny.allocate_cpu_data();

    chase::matrix::Matrix<T, chase::platform::CPU> exact_eigsl_H = chase::matrix::Matrix<T, chase::platform::CPU>(this->N_tiny, 1);
    exact_eigsl_H.readFromBinaryFile(GetBSE_TinyEigs<T>());

    auto V1_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(this->N_tiny, this->N_tiny, mpi_grid);
    V1_.allocate_cpu_data();

    auto V2_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(this->N_tiny, this->N_tiny, mpi_grid);
    V2_.allocate_cpu_data();
    
    auto W1_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(this->N_tiny, this->N_tiny, mpi_grid);
    auto W2_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(this->N_tiny, this->N_tiny, mpi_grid);

    std::size_t g_off = V1_.g_off();

    for(auto i = 0; i < V1_.l_rows(); i++){
	V1_.cpu_data()[g_off * V1_.cpu_ld() + i * (V1_.cpu_ld() + 1)] = T(1.0);
	V2_.cpu_data()[g_off * V2_.cpu_ld() + i * (V2_.cpu_ld() + 1)] = T(1.0);
    }

    V1_.H2D();
    V2_.H2D();

    std::size_t offset = 0, subSize = this->N_tiny; 

    int* devInfo;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&devInfo, sizeof(int)));

    chase::linalg::internal::cuda_nccl::quasi_hermitian_rayleighRitz(this->get_cublas_handle(), 
		    						     this->get_cusolver_handle(),
								     this->get_cusolver_params(), 
								     H_, 
								     V1_, 
								     V2_, 
								     W1_, 
								     W2_, 
								     ritzv_tiny, 
								     offset, 
								     subSize,
								     devInfo);

    for(auto i = offset; i < offset + subSize; i++)
    {
        EXPECT_NEAR(ritzv_tiny.cpu_data()[i], chase::Base<T>(std::real(exact_eigsl_H.data()[i])), 100 * MachineEpsilon<T>::value());
    }
}

TYPED_TEST(QuasiRayleighRitzGPUNCCLDistTest, QuasiHermitianRRDistGPUCorrectness) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    chase::Base<T> tolerance;
    if constexpr(std::is_same<T,float>::value){
	    tolerance = 1.0e-3;
    }else if constexpr(std::is_same<T,std::complex<float>>::value){
	    tolerance = 1.0e-3;
    }else{
	    tolerance = 1.0e-9;
    }

    int *coords = mpi_grid.get()->get_coords();

    auto H_ = chase::distMatrix::QuasiHermitianBlockBlockMatrix<T, chase::platform::GPU>(this->N, this->N, mpi_grid);
    H_.allocate_cpu_data();
    H_.readFromBinaryFile(GetBSE_Matrix<T>());
    H_.H2D();
    
    chase::distMatrix::RedundantMatrix<chase::Base<T>, chase::platform::GPU> ritzv
	    = chase::distMatrix::RedundantMatrix<chase::Base<T>, chase::platform::GPU>(this->N, 1, mpi_grid);
    ritzv.allocate_cpu_data();

    chase::matrix::Matrix<T, chase::platform::CPU> exact_eigsl_H = chase::matrix::Matrix<T, chase::platform::CPU>(this->N, 1);
    exact_eigsl_H.readFromBinaryFile(GetBSE_Eigs<T>());

    auto V1_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(this->N, this->N, mpi_grid);
    V1_.allocate_cpu_data();

    auto V2_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(this->N, this->N, mpi_grid);
    V2_.allocate_cpu_data();
    
    auto W1_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(this->N, this->N, mpi_grid);
    auto W2_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(this->N, this->N, mpi_grid);

    std::size_t g_off = V1_.g_off();

    for(auto i = 0; i < V1_.l_rows(); i++){
	V1_.cpu_data()[g_off * V1_.cpu_ld() + i * (V1_.cpu_ld() + 1)] = T(1.0);
	V2_.cpu_data()[g_off * V2_.cpu_ld() + i * (V2_.cpu_ld() + 1)] = T(1.0);
    }

    V1_.H2D();
    V2_.H2D();

    std::size_t offset = 0, subSize = this->N; 

    int* devInfo;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&devInfo, sizeof(int)));

    chase::linalg::internal::cuda_nccl::quasi_hermitian_rayleighRitz(this->get_cublas_handle(), 
		    						     this->get_cusolver_handle(), 
								     this->get_cusolver_params(), 
								     H_, 
								     V1_, 
								     V2_, 
								     W1_, 
								     W2_, 
								     ritzv, 
								     offset, 
								     subSize,
								     devInfo);

    for(auto i = offset; i < offset + subSize; i++)
    {
        EXPECT_NEAR(ritzv.cpu_data()[i], chase::Base<T>(std::real(exact_eigsl_H.data()[i])), tolerance);
    }
}
