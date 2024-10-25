#include <gtest/gtest.h>
#include <complex>
#include <random>
#include "linalg/internal/cuda/lanczos.hpp"
#include "tests/linalg/internal/utils.hpp"

template <typename T>
class LaczosGPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        CHECK_CUBLAS_ERROR(cublasCreate(&cublasH_));   
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
        CHECK_CUBLAS_ERROR(cublasSetStream(cublasH_, stream_));

        H.resize(N * N);
        V.resize(N * M);
        ritzv.resize(M * numvec);
        ritzV.resize(M * M);
        Tau.resize(M * M * numvec);

        std::mt19937 gen(1337.0);
        std::normal_distribution<> d;

        for(auto i = 0; i < N * M; i++)
        {
            V[i] =  getRandomT<T>([&]() { return d(gen); });
        }

        //set H to be Clement matrix
        //Clement matrix has eigenvalues -(N-1),-(N-2)...(N-2), (N-1)
        for (auto i = 0; i < N; ++i)
        {
            H[i + N * i] = 0;
            if (i != N - 1)
                H[i + 1 + N * i] = std::sqrt(i * (N + 1 - i));
            if (i != N - 1)
                H[i + N * (i + 1)] = std::sqrt(i * (N + 1 - i));
        }

    }

    void TearDown() override {
        if (cublasH_)
            CHECK_CUBLAS_ERROR(cublasDestroy(cublasH_));        
    }

    std::size_t M = 10;
    std::size_t numvec = 4;
    std::size_t N = 500;
    std::vector<T> H;
    std::size_t ldh = N;
    std::vector<T> V;
    std::size_t ldv = N;
    std::vector<chase::Base<T>> ritzv;
    std::vector<chase::Base<T>> ritzV;
    std::vector<chase::Base<T>> Tau;
    cublasHandle_t cublasH_;
    cudaStream_t stream_;    
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(LaczosGPUTest, TestTypes);

TYPED_TEST(LaczosGPUTest, mlanczos) {
    using T = TypeParam;  // Get the current type
    chase::Base<T> upperb;
    chase::matrix::Matrix<T, chase::platform::GPU> Hmat(this->N, this->N, this->N, this->H.data());
    chase::matrix::Matrix<T, chase::platform::GPU> Vec(this->N, this->M, this->N, this->V.data());
    
    chase::linalg::internal::cuda::lanczos(this->cublasH_,
                                           this->M, 
                                           this->numvec, 
                                           Hmat, 
                                           Vec, 
                                           &upperb, 
                                           this->ritzv.data(), 
                                           this->Tau.data(), 
                                           this->ritzV.data());

    for(auto i = 0; i < this->numvec; i++)
    {
        EXPECT_GT(this->ritzv[i * this->M], 1.0 - chase::Base<T>(this->N));
        EXPECT_LT(this->ritzv[(i + 1) * this->M-1], chase::Base<T>(this->N - 1));
    }
    EXPECT_GT(upperb, chase::Base<T>(this->N - 1) ); //the computed upper bound should larger than the max eigenvalues
    EXPECT_LT(upperb, chase::Base<T>(5 * (this->N - 1) ) );
}

TYPED_TEST(LaczosGPUTest, lanczos) {
    using T = TypeParam;  // Get the current type
    chase::Base<T> upperb;
    chase::matrix::Matrix<T, chase::platform::GPU> Hmat(this->N, this->N, this->N, this->H.data());
    chase::matrix::Matrix<T, chase::platform::GPU> Vec(this->N, 1, this->N, this->V.data());
    
    chase::linalg::internal::cuda::lanczos(this->cublasH_,
                                           this->M, 
                                           Hmat, 
                                           Vec, 
                                           &upperb);

    EXPECT_GT(upperb, chase::Base<T>(this->N - 1) ); //the computed upper bound should larger than the max eigenvalues
    EXPECT_LT(upperb, chase::Base<T>(5 * (this->N - 1) ) );
    
}