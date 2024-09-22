#include <gtest/gtest.h>
#include <complex>
#include <cmath>
#include <cstring>
#include "linalg/matrix/matrix.hpp"

// Helper function to create a temporary file with matrix data
template<typename T>
std::string createTempFile(const std::vector<T>& data, std::size_t size) {
    std::string filename = "temp_matrix_file.bin";
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()), size);
    file.close();
    return filename;
}

template <typename T>
class MatrixCPUTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {}
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(MatrixCPUTest, TestTypes);

TYPED_TEST(MatrixCPUTest, InternalMemoryAllocation) {
    using T = TypeParam;  // Get the current type
    std::size_t rows = 3, cols = 4;
    chase::matrix::MatrixCPU<T> matrix(rows, cols);
    EXPECT_EQ(matrix.rows(), rows);
    EXPECT_EQ(matrix.cols(), cols);
    EXPECT_EQ(matrix.ld(), rows);
    EXPECT_NE(matrix.data(), nullptr);
}

TYPED_TEST(MatrixCPUTest, ExternalMemoryUsage) {
    using T = TypeParam;  // Get the current type
    std::size_t rows = 3, cols = 2;
    T external_data[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    chase::matrix::MatrixCPU<T> matrix(rows, cols, rows, external_data);
    EXPECT_EQ(matrix.rows(), rows);
    EXPECT_EQ(matrix.cols(), cols);
    EXPECT_EQ(matrix.ld(), rows);
    EXPECT_EQ(matrix.data(), external_data);

    for(auto i = 0; i < rows * cols; i++)
    {
        EXPECT_EQ(matrix.data()[i], external_data[i]);
    }
}

TYPED_TEST(MatrixCPUTest, DataAccessAndCorrectness) {
    using T = TypeParam;  // Get the current type
    std::size_t rows = 3, cols = 2;
    chase::matrix::MatrixCPU<T> matrix(rows, cols);
    T *data = matrix.data();
    for (std::size_t i = 0; i < rows * cols; ++i) {
        data[i] = T(i + 1);
    }

    for(auto i = 0; i < rows * cols; i++)
    {
        EXPECT_EQ(matrix.data()[i], T(i + 1));
    }
}

TYPED_TEST(MatrixCPUTest, SWAP) {
    using T = TypeParam;  // Get the current type
    
    T array1[4] = {0, 1, 2, 3};
    T array2[6] = {-1, -2, -3, -4, -5, -6};
    chase::matrix::MatrixCPU<T> matrix1(2, 2, 2, array1);
    chase::matrix::MatrixCPU<T> matrix2(3, 2, 3, array2);
    
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

// Test for saveToBinaryFile
TYPED_TEST(MatrixCPUTest, SaveToBinaryFile) {
    using T = TypeParam;
    
    // Sample matrix data
    std::size_t rows = 3;
    std::size_t cols = 3;
    std::size_t ld = 4;
    std::vector<T> sampleData = {1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0}; // 3x3 matrix data

    // Create and save matrix
    chase::matrix::MatrixCPU<T> matrix(rows, cols, ld, sampleData.data());

    std::string filename = "test_save.bin";
    matrix.saveToBinaryFile(filename);

    // Read the file content
    std::ifstream file(filename, std::ios::binary | std::ios::in);
    ASSERT_TRUE(file.is_open());

    // Check if file has the expected size
    std::size_t expectedSize = rows * cols * sizeof(T);
    file.seekg(0, std::ios::end);
    std::size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    ASSERT_EQ(fileSize, expectedSize);

    // Check if file content is as expected
    std::vector<T> buffer(expectedSize / sizeof(T));
    file.read(reinterpret_cast<char*>(buffer.data()), expectedSize);
    file.close();

    for(auto j = 0; j < cols; j++)
    {
        for(auto i = 0; i < rows; i++)
        {
             EXPECT_EQ(buffer[i + j * rows], sampleData[i + j * ld]);
        }
    }

    // Clean up
    std::remove(filename.c_str());
}

// Test for readFromBinaryFile
TYPED_TEST(MatrixCPUTest, ReadFromBinaryFile) {
    using T = TypeParam;
    
    // Sample matrix data
    std::size_t rows = 3;
    std::size_t cols = 3;
    std::size_t ld = 4;
    std::vector<T> buffer(ld * cols);
    std::vector<T> sampleData = {1, 2, 3, 4, 5, 6, 7, 8, 9}; // 3x3 matrix data

    // Create a temporary file with sample data
    std::string filename = createTempFile(sampleData, sampleData.size() * sizeof(T));

    // Create matrix and read from file
    chase::matrix::MatrixCPU<T> matrix(rows, cols, ld, buffer.data());
    matrix.readFromBinaryFile(filename);

    for(auto j = 0; j < cols; j++)
    {
        for(auto i = 0; i < rows; i++)
        {
             EXPECT_EQ(buffer[i + j * ld], sampleData[i + j * cols]);
        }
    }

    std::remove(filename.c_str());
}

#ifdef ENABLE_MIXED_PRECISION
TYPED_TEST(MatrixCPUTest, EnableSinglePrecision) {
    using T = TypeParam;  // Get the current type
    using singlePrecisionT = typename chase::ToSinglePrecisionTrait<T>::Type;
    if constexpr (std::is_same<T, double>::value || std::is_same<T, std::complex<double>>::value) {
        std::size_t rows = 3, cols = 3;
        chase::matrix::MatrixCPU<T> matrix(rows, cols);

        T *data = matrix.data();
        for (std::size_t i = 0; i < rows * cols; ++i) {
            data[i] = T(i + 1);
        }

        // Enable single precision
        matrix.enableSinglePrecision();
        EXPECT_TRUE(matrix.isSinglePrecisionEnabled());
        auto sp_data = matrix.matrix_sp();

        // Check if the single precision data is correctly converted
        for (std::size_t i = 0; i < rows * cols; ++i) {
            EXPECT_EQ(static_cast<singlePrecisionT>(matrix.data()[i]), sp_data->data()[i]);
        }
    }
}

TYPED_TEST(MatrixCPUTest, DisableSinglePrecisionWithCopyback) {
    using T = TypeParam;  // Get the current type
    if constexpr (std::is_same<T, double>::value || std::is_same<T, std::complex<double>>::value) {
        std::size_t rows = 3, cols = 3;
        chase::matrix::MatrixCPU<T> matrix(rows, cols);

        T *data = matrix.data();
        for (std::size_t i = 0; i < rows * cols; ++i) {
            data[i] = T(i + 1);
        }

        // Enable and then disable single precision with copyback
        matrix.enableSinglePrecision();
        matrix.disableSinglePrecision(true);
        EXPECT_FALSE(matrix.isSinglePrecisionEnabled());

        // Check if the data was restored correctly
        for (std::size_t i = 0; i < rows * cols; ++i) {
            EXPECT_EQ(matrix.data()[i], T(i + 1));
        }
    }
}

TYPED_TEST(MatrixCPUTest, UnsupportedSinglePrecisionOperation) {
    using T = TypeParam;  // Get the current type
    if constexpr (std::is_same<T, float>::value || std::is_same<T, std::complex<float>>::value) {
        std::size_t rows = 3, cols = 3;
        chase::matrix::MatrixCPU<T> matrix(rows, cols);

        EXPECT_THROW(matrix.enableSinglePrecision(), std::runtime_error);
        EXPECT_THROW(matrix.disableSinglePrecision(), std::runtime_error);
    }
}

TYPED_TEST(MatrixCPUTest, SinglePrecisionMemoryDeallocation) {
    using T = TypeParam;  // Get the current type
    if constexpr (std::is_same<T, double>::value || std::is_same<T, std::complex<double>>::value) {
        std::size_t rows = 3, cols = 3;
        chase::matrix::MatrixCPU<T> matrix(rows, cols);

        matrix.enableSinglePrecision();
        EXPECT_NE(matrix.matrix_sp()->data(), nullptr);

        matrix.disableSinglePrecision();
        EXPECT_EQ(matrix.matrix_sp(), nullptr);  // Check if the memory is deallocated
    }
}
#endif