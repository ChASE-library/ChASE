// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "algorithm/types.hpp"
#include "linalg/internal/gpu/kernels/cuda/launchers/matrix_inf_norm.cuh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

namespace
{

#define ASSERT_CUDA_SUCCESS(status)                                            \
    ASSERT_EQ((status), cudaSuccess) << cudaGetErrorString(status)

template <typename T>
double abs1(T x)
{
    return std::abs(static_cast<double>(x));
}

template <typename T>
double abs1(std::complex<T> x)
{
    return std::abs(static_cast<double>(x.real())) +
           std::abs(static_cast<double>(x.imag()));
}

template <typename T>
T random_entry(std::mt19937& gen)
{
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    if constexpr (std::is_floating_point_v<T>)
    {
        return static_cast<T>(dist(gen));
    }
    else
    {
        return T(dist(gen), dist(gen));
    }
}

template <typename T>
double reference_inf_norm(const std::vector<T>& A, std::size_t rows,
                          std::size_t cols, std::size_t ld)
{
    double norm = 0.0;
    for (std::size_t i = 0; i < rows; ++i)
    {
        double row_sum = 0.0;
        for (std::size_t j = 0; j < cols; ++j)
        {
            row_sum += abs1(A[i + j * ld]);
        }
        norm = std::max(norm, row_sum);
    }
    return norm;
}

template <typename T>
class MatrixInfNormGPUTest : public ::testing::Test
{
protected:
    using RealT = chase::Base<T>;

    void SetUp() override
    {
        ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream_));
        ASSERT_CUDA_SUCCESS(
            cudaMalloc(reinterpret_cast<void**>(&d_norm_), sizeof(RealT)));
    }

    void TearDown() override
    {
        cudaFree(d_matrix_);
        cudaFree(d_norm_);
        cudaStreamDestroy(stream_);
    }

    void upload(const std::vector<T>& host)
    {
        cudaFree(d_matrix_);
        d_matrix_ = nullptr;
        ASSERT_CUDA_SUCCESS(cudaMalloc(reinterpret_cast<void**>(&d_matrix_),
                                       std::max<std::size_t>(host.size(), 1) *
                                           sizeof(T)));
        ASSERT_CUDA_SUCCESS(cudaMemcpy(d_matrix_, host.data(),
                                       host.size() * sizeof(T),
                                       cudaMemcpyHostToDevice));
    }

    RealT compute(std::size_t rows, std::size_t cols, std::size_t ld)
    {
        chase::linalg::internal::cuda::matrixInfNorm_gpu(d_matrix_, rows, cols,
                                                         ld, d_norm_, stream_);
        RealT norm = RealT(-1);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
        EXPECT_EQ(cudaMemcpyAsync(&norm, d_norm_, sizeof(RealT),
                                  cudaMemcpyDeviceToHost, stream_),
                  cudaSuccess);
        EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        return norm;
    }

    cudaStream_t stream_ = nullptr;
    T* d_matrix_ = nullptr;
    RealT* d_norm_ = nullptr;
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(MatrixInfNormGPUTest, TestTypes);

struct Shape
{
    std::size_t rows;
    std::size_t cols;
    std::size_t ld;
};

} // namespace

TYPED_TEST(MatrixInfNormGPUTest, MatchesReferenceAcrossShapes)
{
    using T = TypeParam;
    using RealT = chase::Base<T>;

    // Covers single-block, partial row tiles, lda > rows, very wide matrices
    // (column-split path) and tall-skinny matrices (fused path).
    const std::vector<Shape> shapes = {
        {1, 1, 1},       {7, 3, 9},      {31, 1000, 40},  {33, 5, 33},
        {100, 20000, 128}, {1000, 1000, 1001}, {50000, 17, 50000},
        {3000, 3000, 3000},
    };

    for (const Shape& s : shapes)
    {
        SCOPED_TRACE(::testing::Message() << s.rows << "x" << s.cols
                                          << " ld=" << s.ld);
        std::mt19937 gen(1234 + s.rows * 7 + s.cols);
        std::vector<T> host(s.ld * s.cols);
        for (auto& a : host)
        {
            a = random_entry<T>(gen);
        }
        this->upload(host);

        const double expected = reference_inf_norm(host, s.rows, s.cols, s.ld);
        const double got = static_cast<double>(this->compute(s.rows, s.cols, s.ld));
        const double tol = 4.0 * std::numeric_limits<RealT>::epsilon() *
                           static_cast<double>(s.cols) * expected;
        EXPECT_NEAR(got, expected, tol);
    }
}

TYPED_TEST(MatrixInfNormGPUTest, IgnoresPaddingRows)
{
    using T = TypeParam;
    using RealT = chase::Base<T>;

    constexpr std::size_t rows = 3;
    constexpr std::size_t cols = 2;
    constexpr std::size_t ld = 5;
    std::vector<T> host(ld * cols, T(RealT(1000)));
    // Column-major: rows 0..2 are data, rows 3..4 are padding.
    host[0] = T(RealT(1));
    host[1] = T(RealT(-2));
    host[2] = T(RealT(3));
    host[5] = T(RealT(-4));
    host[6] = T(RealT(5));
    host[7] = T(RealT(-6));
    this->upload(host);

    EXPECT_EQ(this->compute(rows, cols, ld), RealT(9));
}

TYPED_TEST(MatrixInfNormGPUTest, EmptyMatrixYieldsZero)
{
    using T = TypeParam;
    using RealT = chase::Base<T>;

    this->upload(std::vector<T>(4, T(RealT(1))));
    const RealT stale = RealT(42);
    ASSERT_CUDA_SUCCESS(
        cudaMemcpy(this->d_norm_, &stale, sizeof(RealT), cudaMemcpyHostToDevice));
    EXPECT_EQ(this->compute(0, 4, 1), RealT(0));
    EXPECT_EQ(this->compute(4, 0, 4), RealT(0));
}

TYPED_TEST(MatrixInfNormGPUTest, IsBitwiseDeterministic)
{
    using T = TypeParam;
    using RealT = chase::Base<T>;

    constexpr std::size_t rows = 500;
    constexpr std::size_t cols = 20000;
    std::mt19937 gen(7);
    std::vector<T> host(rows * cols);
    for (auto& a : host)
    {
        a = random_entry<T>(gen);
    }
    this->upload(host);

    const RealT first = this->compute(rows, cols, rows);
    for (int rep = 0; rep < 5; ++rep)
    {
        const RealT again = this->compute(rows, cols, rows);
        EXPECT_EQ(std::memcmp(&first, &again, sizeof(RealT)), 0);
    }
}

TYPED_TEST(MatrixInfNormGPUTest, PropagatesNaN)
{
    using T = TypeParam;
    using RealT = chase::Base<T>;

    for (const std::size_t cols : {std::size_t(8), std::size_t(20000)})
    {
        SCOPED_TRACE(::testing::Message() << "cols=" << cols);
        constexpr std::size_t rows = 64;
        std::vector<T> host(rows * cols, T(RealT(1)));
        host[rows * (cols / 2) + 40] = T(std::numeric_limits<RealT>::quiet_NaN());
        this->upload(host);
        EXPECT_TRUE(std::isnan(this->compute(rows, cols, rows)));
    }
}

TEST(MatrixInfNormGPUComplexTest, UsesAbsRealPlusAbsImag)
{
    // BLAS asum convention: |3+4i| contributes 7, not 5.
    std::vector<std::complex<double>> host = {{3.0, 4.0}, {0.0, 1.0},
                                              {-1.0, 0.0}, {0.0, -2.0}};
    std::complex<double>* d_matrix = nullptr;
    double* d_norm = nullptr;
    ASSERT_CUDA_SUCCESS(cudaMalloc(reinterpret_cast<void**>(&d_matrix),
                                   host.size() * sizeof(host[0])));
    ASSERT_CUDA_SUCCESS(cudaMalloc(reinterpret_cast<void**>(&d_norm), sizeof(double)));
    ASSERT_CUDA_SUCCESS(cudaMemcpy(d_matrix, host.data(),
                                   host.size() * sizeof(host[0]),
                                   cudaMemcpyHostToDevice));

    chase::linalg::internal::cuda::matrixInfNorm_gpu(d_matrix, 2, 2, 2, d_norm,
                                                     nullptr);
    double norm = -1.0;
    ASSERT_CUDA_SUCCESS(
        cudaMemcpy(&norm, d_norm, sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_matrix);
    cudaFree(d_norm);

    // Row 0: |3|+|4| + |-1| = 8; row 1: |1| + |-2| = 3.
    EXPECT_EQ(norm, 8.0);
}
