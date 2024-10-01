#include <gtest/gtest.h>
#include <complex>
#include <random>
#include <cmath>
#include <cstring>
#include "linalg/internal/cuda/lacpy.hpp"
#include "linalg/matrix/matrix.hpp"

template <typename T>
class lacpyGPUTest : public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(lacpyGPUTest, TestTypes);

TYPED_TEST(lacpyGPUTest, fullCopy) {
    using T = TypeParam;

    std::size_t rows = 3;
    std::size_t cols = 3;
    std::size_t ld = 4;
    std::vector<T> buffer(ld * cols);

    for(auto i = 0; i < ld * cols; i++){
        buffer[i] = i + 1;
    }

    chase::matrix::MatrixGPU<T> src_matrix(rows, cols, ld, buffer.data());
    chase::matrix::MatrixGPU<T> target_matrix(rows, cols);
    chase::linalg::internal::cuda::t_lacpy('A', 
                                           rows, 
                                           cols, 
                                           src_matrix.gpu_data(), 
                                           src_matrix.gpu_ld(), 
                                           target_matrix.gpu_data(), 
                                           target_matrix.gpu_ld());
    target_matrix.D2H();

    for(auto j = 0; j < cols; j++)
    {
        for(auto i = 0; i < rows; i++)
        {
            EXPECT_EQ(target_matrix.cpu_data()[j * rows + i], buffer.data()[j * ld + i]);
        }
    }
}

TYPED_TEST(lacpyGPUTest, upperCopy) {
    using T = TypeParam;

    std::size_t rows = 3;
    std::size_t cols = 3;
    std::vector<T> buffer(rows * cols);

    for(auto i = 0; i < rows * cols; i++){
        buffer[i] = i + 1;
    }

    std::vector<T> buffer2(rows * cols);
    for(auto i = 0; i < rows * cols; i++){
        buffer2[i] = -(i + 1);
    }

    chase::matrix::MatrixGPU<T> src_matrix(rows, cols, rows, buffer.data());
    chase::matrix::MatrixGPU<T> target_matrix(rows, cols, rows, buffer2.data());
    chase::linalg::internal::cuda::t_lacpy('U', 
                                           rows, 
                                           cols, 
                                           src_matrix.gpu_data(), 
                                           src_matrix.gpu_ld(), 
                                           target_matrix.gpu_data(), 
                                           target_matrix.gpu_ld());
    target_matrix.D2H();

    for(auto j = 0; j < cols; j++)
    {
        for(auto i = 0; i < rows; i++)
        {
            if(j >= i){
                EXPECT_EQ(target_matrix.cpu_data()[j * rows + i], buffer.data()[j * rows + i]);
            }
            else
            {
                EXPECT_EQ(target_matrix.cpu_data()[j * rows + i],  T(0.0));
            }
        }
    }
}

TYPED_TEST(lacpyGPUTest, lowerCopy) {
    using T = TypeParam;

    std::size_t rows = 3;
    std::size_t cols = 3;
    std::vector<T> buffer(rows * cols);

    for(auto i = 0; i < rows * cols; i++){
        buffer[i] = i + 1;
    }

    std::vector<T> buffer2(rows * cols);
    for(auto i = 0; i < rows * cols; i++){
        buffer2[i] = -(i + 1);
    }

    chase::matrix::MatrixGPU<T> src_matrix(rows, cols, rows, buffer.data());
    chase::matrix::MatrixGPU<T> target_matrix(rows, cols, rows, buffer2.data());
    chase::linalg::internal::cuda::t_lacpy('L', 
                                           rows, 
                                           cols, 
                                           src_matrix.gpu_data(), 
                                           src_matrix.gpu_ld(), 
                                           target_matrix.gpu_data(), 
                                           target_matrix.gpu_ld());
    target_matrix.D2H();

    for(auto j = 0; j < cols; j++)
    {
        for(auto i = 0; i < rows; i++)
        {
            if(j <= i){
                EXPECT_EQ(target_matrix.cpu_data()[j * rows + i], buffer.data()[j * rows + i]);
            }
            else
            {
                EXPECT_EQ(target_matrix.cpu_data()[j * rows + i],  T(0.0));
            }
        }
    }
}
