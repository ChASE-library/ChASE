#include <gtest/gtest.h>
#include <complex>
#include <random>
#include <cmath>
#include <cstring>
#include "linalg/internal/cuda/residuals.hpp"
#include "tests/linalg/internal/utils.hpp"
#include "linalg/matrix/matrix.hpp"
#include "Impl/cuda/cuda_utils.hpp"
#include "linalg/lapackpp/lapackpp.hpp"

template <typename T>
class ResidsGPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        H.resize(N * N);
        evals.resize(N);
        evecs.resize(N * N);
        resids.resize(N);
        //CHECK_CUDA_ERROR(cudaMalloc((void**)&d_abstrace, sizeof(chase::Base<T>)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_resids, N * sizeof(chase::Base<T>)));      
        CHECK_CUDA_ERROR(cudaMalloc(&d_ritzv, N * sizeof(chase::Base<T>)));   
        CHECK_CUBLAS_ERROR(cublasCreate(&cublasH_));   
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
        CHECK_CUBLAS_ERROR(cublasSetStream(cublasH_, stream_));

    }

    void TearDown() override 
    {
        cudaFree(d_resids);
        cudaFree(d_ritzv);
        if (cublasH_)
            cublasDestroy(cublasH_);

    }

    std::size_t N = 64;
    std::vector<T> H;
    std::vector<chase::Base<T>> evals;
    std::vector<T> evecs;
    std::vector<chase::Base<T>> resids;
    chase::Base<T> *d_resids;
    chase::Base<T> *d_ritzv;
    cublasHandle_t cublasH_;
    cudaStream_t stream_;
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(ResidsGPUTest, TestTypes);

TYPED_TEST(ResidsGPUTest, DiagonalMatrix) {
    using T = TypeParam;  // Get the current type
    auto machineEpsilon = MachineEpsilon<T>::value();

    for(auto i = 0; i < this->N; i++)
    {
        this->H[i * this->N + i] = T(i + 1);
    }

    for(auto i = 0; i < this->N; i++)
    {
        this->evals[i] = chase::Base<T>(i + 1);
        this->evecs[i + this->N * i] = T(1.0);
    }

    CHECK_CUDA_ERROR(cudaMemcpy(this->d_ritzv, this->evals.data(), this->N * sizeof(chase::Base<T>), cudaMemcpyHostToDevice));
    chase::matrix::Matrix<T, chase::platform::GPU> Hmat(this->N, this->N, this->N, this->H.data());
    chase::matrix::Matrix<T, chase::platform::GPU> V1(this->N, this->N, this->N, this->evecs.data());
    chase::linalg::internal::cuda::residuals(this->cublasH_, Hmat, V1, this->d_ritzv, this->d_resids, 0, this->N);
    CHECK_CUDA_ERROR(cudaMemcpy(this->resids.data(), this->d_resids, this->N * sizeof(chase::Base<T>), cudaMemcpyDeviceToHost));

    for(auto i = 0; i < this->N; i++)
    {
        EXPECT_NEAR(this->resids[i], machineEpsilon, machineEpsilon * 10);
    }
}

TYPED_TEST(ResidsGPUTest, DenseMatrix) {
    using T = TypeParam;  // Get the current type
    auto machineEpsilon = MachineEpsilon<T>::value();
    T One = T(1.0);
    T Zero = T(0.0);
    std::size_t offset = 2;
    std::size_t subSize = 10;
    for (int i = 0; i < this->N; ++i) {
        this->H[i * this->N + i] = T(0.1 * i + 0.1);
    }

    std::mt19937 gen(1337.0);
    std::normal_distribution<> d;
    std::vector<T> V(this->N * this->N);

    for(auto i = 0; i < this->N * this->N; i++)
    {
        V[i] =  getRandomT<T>([&]() { return d(gen); });
    }

    std::unique_ptr<T[]> tau(new T[this->N]);

    chase::linalg::lapackpp::t_geqrf(LAPACK_COL_MAJOR, this->N, this->N, V.data(), this->N, tau.get());
    chase::linalg::lapackpp::t_gqr(LAPACK_COL_MAJOR, this->N, this->N, this->N, V.data(), this->N, tau.get());

    chase::linalg::blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, this->N, this->N, this->N,
               &One, this->H.data(), this->N, V.data(), this->N, &Zero,
               this->H.data(), this->N);

    chase::linalg::blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasConjTrans, this->N, this->N, this->N,
               &One, V.data(), this->N, this->H.data(), this->N, &Zero,
               this->evecs.data(), this->N);

    chase::linalg::lapackpp::t_lacpy('A', this->N, this->N, this->evecs.data(), this->N, this->H.data(), this->N);

    chase::linalg::lapackpp::t_heevd(CblasColMajor, 'V', 'U', this->N,
                    this->evecs.data(), this->N, this->evals.data());

    CHECK_CUDA_ERROR(cudaMemcpy(this->d_ritzv, this->evals.data(), this->N * sizeof(chase::Base<T>), cudaMemcpyHostToDevice));
    chase::matrix::Matrix<T, chase::platform::GPU> Hmat(this->N, this->N, this->N, this->H.data());
    chase::matrix::Matrix<T, chase::platform::GPU> V1(this->N, this->N, this->N, this->evecs.data());
    chase::linalg::internal::cuda::residuals(this->cublasH_, Hmat, V1, this->d_ritzv, this->d_resids, offset, subSize);
    CHECK_CUDA_ERROR(cudaMemcpy(this->resids.data() + offset, this->d_resids + offset, subSize * sizeof(chase::Base<T>), cudaMemcpyDeviceToHost));

    
    for(auto i = offset; i < offset + subSize; i++)
    {
        EXPECT_NEAR(this->resids[i], machineEpsilon, machineEpsilon * 1e2);
    }
    
}