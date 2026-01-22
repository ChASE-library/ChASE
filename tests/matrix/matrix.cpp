// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/matrix/matrix.hpp"
#include <cmath>
#include <complex>
#include <cstring>
#include <gtest/gtest.h>
// #include "linalg/cublaspp/cublaspp.hpp"

// Helper function to create a temporary file with matrix data
template <typename T>
std::string createTempFile(const std::vector<T>& data, std::size_t size)
{
    std::string filename = "temp_matrix_file.bin";
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()), size);
    file.close();
    return filename;
}

template <typename T>
class MatrixCPUTest : public ::testing::Test
{
protected:
    void SetUp() override {}

    void TearDown() override {}
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(MatrixCPUTest, TestTypes);

TYPED_TEST(MatrixCPUTest, InternalMemoryAllocation)
{
    using T = TypeParam; // Get the current type
    std::size_t rows = 3, cols = 4;
    chase::matrix::Matrix<T, chase::platform::CPU> matrix(rows, cols);
    EXPECT_EQ(matrix.rows(), rows);
    EXPECT_EQ(matrix.cols(), cols);
    EXPECT_EQ(matrix.ld(), rows);
    EXPECT_NE(matrix.data(), nullptr);
}

TYPED_TEST(MatrixCPUTest, ExternalMemoryUsage)
{
    using T = TypeParam; // Get the current type
    std::size_t rows = 3, cols = 2;
    T external_data[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    chase::matrix::Matrix<T> matrix(rows, cols, rows, external_data);
    EXPECT_EQ(matrix.rows(), rows);
    EXPECT_EQ(matrix.cols(), cols);
    EXPECT_EQ(matrix.ld(), rows);
    EXPECT_EQ(matrix.data(), external_data);

    for (auto i = 0; i < rows * cols; i++)
    {
        EXPECT_EQ(matrix.data()[i], external_data[i]);
    }
}

TYPED_TEST(MatrixCPUTest, DataAccessAndCorrectness)
{
    using T = TypeParam; // Get the current type
    std::size_t rows = 3, cols = 2;
    chase::matrix::Matrix<T, chase::platform::CPU> matrix(rows, cols);
    T* data = matrix.data();
    for (std::size_t i = 0; i < rows * cols; ++i)
    {
        data[i] = T(i + 1);
    }

    for (auto i = 0; i < rows * cols; i++)
    {
        EXPECT_EQ(matrix.data()[i], T(i + 1));
    }
}

TYPED_TEST(MatrixCPUTest, SWAP)
{
    using T = TypeParam; // Get the current type

    T array1[4] = {0, 1, 2, 3};
    T array2[6] = {-1, -2, -3, -4, -5, -6};
    chase::matrix::Matrix<T, chase::platform::CPU> matrix1(2, 2, 2, array1);
    chase::matrix::Matrix<T, chase::platform::CPU> matrix2(3, 2, 3, array2);

    matrix1.swap(matrix2);
    EXPECT_EQ(matrix1.rows(), 3);
    EXPECT_EQ(matrix1.cols(), 2);
    EXPECT_EQ(matrix1.ld(), 3);
    EXPECT_EQ(matrix2.rows(), 2);
    EXPECT_EQ(matrix2.cols(), 2);
    EXPECT_EQ(matrix2.ld(), 2);

    EXPECT_NE(matrix2.data(), nullptr);
    EXPECT_NE(matrix1.data(), nullptr);
    EXPECT_EQ(matrix2.data(), array1);
    EXPECT_EQ(matrix1.data(), array2);
}

TYPED_TEST(MatrixCPUTest, PseudoHermitianMatrix)
{

    std::size_t rows = 4, cols = 4;

    std::complex<double> external_data[16] = {
        std::complex<double>(1.010, 0), std::complex<double>(0, -0.20),
        std::complex<double>(0.010, 0), std::complex<double>(0, 0.010),
        std::complex<double>(0, 0.200), std::complex<double>(1.010, 0),
        std::complex<double>(0, 0.100), std::complex<double>(0.100, 0),
        std::complex<double>(-0.10, 0), std::complex<double>(0, 0.100),
        std::complex<double>(-1.01, 0), std::complex<double>(0, -0.20),
        std::complex<double>(0, 0.100), std::complex<double>(-0.10, 0),
        std::complex<double>(0, 0.200), std::complex<double>(-1.01, 0)};

    chase::matrix::PseudoHermitianMatrix<std::complex<double>> matrix(
        rows, cols, rows, external_data);
    EXPECT_EQ(matrix.rows(), rows);
    EXPECT_EQ(matrix.cols(), cols);
    EXPECT_EQ(matrix.ld(), rows);
    EXPECT_EQ(matrix.data(), external_data);

    for (auto i = 0; i < rows * cols; i++)
    {
        EXPECT_EQ(matrix.data()[i], external_data[i]);
    }
}

#ifdef ENABLE_MIXED_PRECISION
TYPED_TEST(MatrixCPUTest, EnableSinglePrecision)
{
    using T = TypeParam; // Get the current type
    using singlePrecisionT = typename chase::ToSinglePrecisionTrait<T>::Type;
    if constexpr (std::is_same<T, double>::value ||
                  std::is_same<T, std::complex<double>>::value)
    {
        std::size_t rows = 3, cols = 3;
        chase::matrix::Matrix<T, chase::platform::CPU> matrix(rows, cols);

        T* data = matrix.data();
        for (std::size_t i = 0; i < rows * cols; ++i)
        {
            data[i] = T(i + 1);
        }

        // Enable single precision
        matrix.enableSinglePrecision();
        EXPECT_TRUE(matrix.isSinglePrecisionEnabled());
        auto sp_data = matrix.matrix_sp();

        // Check if the single precision data is correctly converted
        for (std::size_t i = 0; i < rows * cols; ++i)
        {
            EXPECT_EQ(static_cast<singlePrecisionT>(matrix.data()[i]),
                      sp_data->data()[i]);
        }
    }
}

TYPED_TEST(MatrixCPUTest, DisableSinglePrecisionWithCopyback)
{
    using T = TypeParam; // Get the current type
    if constexpr (std::is_same<T, double>::value ||
                  std::is_same<T, std::complex<double>>::value)
    {
        std::size_t rows = 3, cols = 3;
        chase::matrix::Matrix<T, chase::platform::CPU> matrix(rows, cols);

        T* data = matrix.data();
        for (std::size_t i = 0; i < rows * cols; ++i)
        {
            data[i] = T(i + 1);
        }

        // Enable and then disable single precision with copyback
        matrix.enableSinglePrecision();
        matrix.disableSinglePrecision(true);
        EXPECT_FALSE(matrix.isSinglePrecisionEnabled());

        // Check if the data was restored correctly
        for (std::size_t i = 0; i < rows * cols; ++i)
        {
            EXPECT_EQ(matrix.data()[i], T(i + 1));
        }
    }
}

TYPED_TEST(MatrixCPUTest, UnsupportedSinglePrecisionOperation)
{
    using T = TypeParam; // Get the current type
    if constexpr (std::is_same<T, float>::value ||
                  std::is_same<T, std::complex<float>>::value)
    {
        std::size_t rows = 3, cols = 3;
        chase::matrix::Matrix<T, chase::platform::CPU> matrix(rows, cols);

        EXPECT_THROW(matrix.enableSinglePrecision(), std::runtime_error);
        EXPECT_THROW(matrix.disableSinglePrecision(), std::runtime_error);
    }
}

TYPED_TEST(MatrixCPUTest, SinglePrecisionMemoryDeallocation)
{
    using T = TypeParam; // Get the current type
    if constexpr (std::is_same<T, double>::value ||
                  std::is_same<T, std::complex<double>>::value)
    {
        std::size_t rows = 3, cols = 3;
        chase::matrix::Matrix<T, chase::platform::CPU> matrix(rows, cols);

        matrix.enableSinglePrecision();
        EXPECT_NE(matrix.matrix_sp()->data(), nullptr);

        matrix.disableSinglePrecision();
        EXPECT_THROW(matrix.matrix_sp(), std::runtime_error);
    }
}
#endif

#ifdef HAS_CUDA
template <typename T>
class MatrixGPUTest : public ::testing::Test
{
protected:
    void SetUp() override {}

    void TearDown() override {}
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(MatrixGPUTest, TestTypes);

TYPED_TEST(MatrixGPUTest, InternalMemoryAllocation)
{
    using T = TypeParam; // Get the current type
    std::size_t rows = 3, cols = 4;
    chase::matrix::Matrix<T, chase::platform::GPU> matrix(rows, cols);
    EXPECT_EQ(matrix.rows(), rows);
    EXPECT_EQ(matrix.cols(), cols);
    EXPECT_EQ(matrix.ld(), rows);
    EXPECT_NE(matrix.data(), nullptr);
    EXPECT_EQ(matrix.cpu_ld(), 0);
    EXPECT_THROW({ matrix.cpu_data(); }, std::runtime_error);
    EXPECT_EQ(matrix.cpu_ld(), 0);
}

TYPED_TEST(MatrixGPUTest, ExternalCPUMemoryUsage)
{
    using T = TypeParam; // Get the current type
    std::size_t rows = 3, cols = 2;
    T external_data[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    chase::matrix::Matrix<T, chase::platform::GPU> matrix(rows, cols, rows,
                                                          external_data);
    EXPECT_EQ(matrix.rows(), rows);
    EXPECT_EQ(matrix.cols(), cols);
    EXPECT_EQ(matrix.cpu_ld(), rows);
    EXPECT_EQ(matrix.cpu_data(), external_data);
    EXPECT_NE(matrix.data(), nullptr);
    EXPECT_EQ(matrix.ld(), rows);

    for (auto i = 0; i < rows * cols; i++)
    {
        EXPECT_EQ(matrix.cpu_data()[i], external_data[i]);
    }
}

TYPED_TEST(MatrixGPUTest, ExternalGPUMemoryUsage)
{
    using T = TypeParam; // Get the current type
    std::size_t rows = 3, cols = 2;
    T* external_data;
    CHECK_CUDA_ERROR(
        cudaMalloc((void**)&external_data, rows * cols * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMemset(external_data, 0, rows * cols * sizeof(T)));
    chase::matrix::Matrix<T, chase::platform::GPU> matrix(
        rows, cols, rows, external_data, chase::matrix::BufferType::GPU);
    EXPECT_EQ(matrix.rows(), rows);
    EXPECT_EQ(matrix.cols(), cols);
    EXPECT_EQ(matrix.cpu_ld(), 0);
    EXPECT_THROW({ matrix.cpu_data(); }, std::runtime_error);
    EXPECT_EQ(matrix.data(), external_data);
    EXPECT_EQ(matrix.ld(), rows);

    matrix.D2H();

    for (auto i = 0; i < rows * cols; i++)
    {
        EXPECT_EQ(matrix.cpu_data()[i], T(0.0));
    }
}

TYPED_TEST(MatrixGPUTest, DataAccessAndCorrectness)
{
    using T = TypeParam; // Get the current type
    std::size_t rows = 3, cols = 2;
    std::vector<T> data(rows * cols);
    T* cpu_data = data.data();
    for (std::size_t i = 0; i < rows * cols; ++i)
    {
        cpu_data[i] = T(i + 1);
    }
    chase::matrix::Matrix<T, chase::platform::GPU> matrix(rows, cols, rows,
                                                          cpu_data);

    std::vector<T> gpu_copied(matrix.rows() * matrix.cols());
    CHECK_CUBLAS_ERROR(cublasGetMatrix(matrix.rows(), matrix.cols(), sizeof(T),
                                       matrix.data(), matrix.ld(),
                                       gpu_copied.data(), matrix.rows()));

    for (auto i = 0; i < rows * cols; i++)
    {
        EXPECT_EQ(gpu_copied.data()[i], T(i + 1));
    }
}

TYPED_TEST(MatrixGPUTest, SWAP)
{
    using T = TypeParam; // Get the current type

    T array1[4] = {0, 1, 2, 3};
    T array2[6] = {-1, -2, -3, -4, -5, -6};
    chase::matrix::Matrix<T, chase::platform::GPU> matrix1(2, 2, 2, array1);
    chase::matrix::Matrix<T, chase::platform::GPU> matrix2(3, 2, 3, array2);

    matrix1.swap(matrix2);
    EXPECT_EQ(matrix1.rows(), 3);
    EXPECT_EQ(matrix1.cols(), 2);
    EXPECT_EQ(matrix1.cpu_ld(), 3);
    EXPECT_EQ(matrix1.ld(), 3);
    EXPECT_EQ(matrix2.rows(), 2);
    EXPECT_EQ(matrix2.cols(), 2);
    EXPECT_EQ(matrix2.cpu_ld(), 2);
    EXPECT_EQ(matrix2.ld(), 2);

    EXPECT_NE(matrix2.cpu_data(), nullptr);
    EXPECT_NE(matrix1.cpu_data(), nullptr);
    EXPECT_NE(matrix2.data(), nullptr);
    EXPECT_NE(matrix1.data(), nullptr);

    EXPECT_EQ(matrix2.cpu_data(), array1);
    EXPECT_EQ(matrix1.cpu_data(), array2);

    std::vector<T> gpu_copied_1(matrix1.rows() * matrix1.cols());
    std::vector<T> gpu_copied_2(matrix2.rows() * matrix2.cols());

    CHECK_CUBLAS_ERROR(cublasGetMatrix(matrix1.rows(), matrix1.cols(),
                                       sizeof(T), matrix1.data(), matrix1.ld(),
                                       gpu_copied_1.data(), matrix1.rows()));

    CHECK_CUBLAS_ERROR(cublasGetMatrix(matrix2.rows(), matrix2.cols(),
                                       sizeof(T), matrix2.data(), matrix2.ld(),
                                       gpu_copied_2.data(), matrix2.rows()));

    for (auto i = 0; i < matrix1.rows() * matrix1.cols(); i++)
    {
        gpu_copied_1.data()[i] = array2[i];
    }

    for (auto i = 0; i < matrix2.rows() * matrix2.cols(); i++)
    {
        gpu_copied_2.data()[i] = array1[i];
    }
}

#ifdef ENABLE_MIXED_PRECISION
TYPED_TEST(MatrixGPUTest, EnableSinglePrecision)
{
    using T = TypeParam; // Get the current type
    using singlePrecisionT = typename chase::ToSinglePrecisionTrait<T>::Type;
    if constexpr (std::is_same<T, double>::value ||
                  std::is_same<T, std::complex<double>>::value)
    {
        std::size_t rows = 3, cols = 3;
        std::vector<T> data(rows * cols);

        for (std::size_t i = 0; i < rows * cols; ++i)
        {
            data[i] = T(i + 1);
        }

        chase::matrix::Matrix<T, chase::platform::GPU> matrix(rows, cols, rows,
                                                              data.data());

        // Enable single precision
        matrix.enableSinglePrecision();
        EXPECT_TRUE(matrix.isSinglePrecisionEnabled());
        auto sp_data = matrix.matrix_sp();
        sp_data->allocate_cpu_data();
        sp_data->D2H();
        // Check if the single precision data is correctly converted
        for (std::size_t i = 0; i < rows * cols; ++i)
        {
            EXPECT_EQ(static_cast<singlePrecisionT>(matrix.cpu_data()[i]),
                      sp_data->cpu_data()[i]);
        }
    }
}

TYPED_TEST(MatrixGPUTest, DisableSinglePrecisionWithCopyback)
{
    using T = TypeParam; // Get the current type
    if constexpr (std::is_same<T, double>::value ||
                  std::is_same<T, std::complex<double>>::value)
    {
        std::size_t rows = 3, cols = 3;
        std::vector<T> data(rows * cols);
        for (std::size_t i = 0; i < rows * cols; ++i)
        {
            data[i] = T(i + 1);
        }
        chase::matrix::Matrix<T, chase::platform::GPU> matrix(rows, cols, rows,
                                                              data.data());

        // Enable and then disable single precision with copyback
        matrix.enableSinglePrecision();
        CHECK_CUDA_ERROR(cudaMemset(matrix.data(), 0, rows * cols * sizeof(T)));
        memset(matrix.cpu_data(), 0, rows * cols * sizeof(T));

        matrix.disableSinglePrecision(true);
        EXPECT_FALSE(matrix.isSinglePrecisionEnabled());
        matrix.D2H();
        // Check if the data was restored correctly
        for (std::size_t i = 0; i < rows * cols; ++i)
        {
            EXPECT_EQ(matrix.cpu_data()[i], T(i + 1));
        }
    }
}

TYPED_TEST(MatrixGPUTest, UnsupportedSinglePrecisionOperation)
{
    using T = TypeParam; // Get the current type
    if constexpr (std::is_same<T, float>::value ||
                  std::is_same<T, std::complex<float>>::value)
    {
        std::size_t rows = 3, cols = 3;
        chase::matrix::Matrix<T, chase::platform::GPU> matrix(rows, cols);

        EXPECT_THROW(matrix.enableSinglePrecision(), std::runtime_error);
        EXPECT_THROW(matrix.disableSinglePrecision(), std::runtime_error);
    }
}

TYPED_TEST(MatrixGPUTest, SinglePrecisionMemoryDeallocation)
{
    using T = TypeParam; // Get the current type
    if constexpr (std::is_same<T, double>::value ||
                  std::is_same<T, std::complex<double>>::value)
    {
        std::size_t rows = 3, cols = 3;
        chase::matrix::Matrix<T, chase::platform::GPU> matrix(rows, cols);

        matrix.enableSinglePrecision();
        EXPECT_NE(matrix.matrix_sp()->data(), nullptr);

        matrix.disableSinglePrecision();
        EXPECT_THROW(matrix.matrix_sp(), std::runtime_error);
    }
}

#endif
#endif
