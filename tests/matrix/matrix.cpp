#include <gtest/gtest.h>
#include <complex>
#include <cmath>
#include <cstring>
#include "linalg/matrix/matrix.hpp"
//#include "linalg/cublaspp/cublaspp.hpp"

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

#ifdef HAS_CUDA
template <typename T>
class MatrixGPUTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {}
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(MatrixGPUTest, TestTypes);

TYPED_TEST(MatrixGPUTest, InternalMemoryAllocation) {
    using T = TypeParam;  // Get the current type
    std::size_t rows = 3, cols = 4;
    chase::matrix::MatrixGPU<T> matrix(rows, cols);
    EXPECT_EQ(matrix.rows(), rows);
    EXPECT_EQ(matrix.cols(), cols);
    EXPECT_EQ(matrix.gpu_ld(), rows);
    EXPECT_NE(matrix.gpu_data(), nullptr);
    EXPECT_EQ(matrix.cpu_ld(), 0);
    EXPECT_EQ(matrix.cpu_data(), nullptr);
    EXPECT_EQ(matrix.cpu_ld(), 0);
}

TYPED_TEST(MatrixGPUTest, ExternalMemoryUsage) {
    using T = TypeParam;  // Get the current type
    std::size_t rows = 3, cols = 2;
    T external_data[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    chase::matrix::MatrixGPU<T> matrix(rows, cols, rows, external_data);
    EXPECT_EQ(matrix.rows(), rows);
    EXPECT_EQ(matrix.cols(), cols);
    EXPECT_EQ(matrix.cpu_ld(), rows);
    EXPECT_EQ(matrix.cpu_data(), external_data);
    EXPECT_NE(matrix.gpu_data(), nullptr);

    for(auto i = 0; i < rows * cols; i++)
    {
        EXPECT_EQ(matrix.cpu_data()[i], external_data[i]);
    }
}

TYPED_TEST(MatrixGPUTest, DataAccessAndCorrectness) {
    using T = TypeParam;  // Get the current type
    std::size_t rows = 3, cols = 2;
    std::vector<T> data(rows * cols);
    T *cpu_data = data.data();
    for (std::size_t i = 0; i < rows * cols; ++i) {
        cpu_data[i] = T(i + 1);
    }
    chase::matrix::MatrixGPU<T> matrix(rows, cols, rows, cpu_data);

    std::vector<T> gpu_copied(matrix.rows() * matrix.cols());
    CHECK_CUBLAS_ERROR(cublasGetMatrix(matrix.rows(), matrix.cols(), sizeof(T),
                        matrix.gpu_data(), matrix.gpu_ld(), gpu_copied.data(), matrix.rows()));
    
    for(auto i = 0; i < rows * cols; i++)
    {
        EXPECT_EQ(gpu_copied.data()[i], T(i + 1));
    }
}

TYPED_TEST(MatrixGPUTest, SWAP) {
    using T = TypeParam;  // Get the current type
    
    T array1[4] = {0, 1, 2, 3};
    T array2[6] = {-1, -2, -3, -4, -5, -6};
    chase::matrix::MatrixGPU<T> matrix1(2, 2, 2, array1);
    chase::matrix::MatrixGPU<T> matrix2(3, 2, 3, array2);
    
    matrix1.swap(matrix2);
    EXPECT_EQ(matrix1.rows(), 3);
    EXPECT_EQ(matrix1.cols(), 2);
    EXPECT_EQ(matrix1.cpu_ld(), 3);
    EXPECT_EQ(matrix1.gpu_ld(), 3);
    EXPECT_EQ(matrix2.rows(), 2);
    EXPECT_EQ(matrix2.cols(), 2);
    EXPECT_EQ(matrix2.cpu_ld(), 2);
    EXPECT_EQ(matrix2.gpu_ld(), 2);

    EXPECT_NE(matrix2.cpu_data(), nullptr);
    EXPECT_NE(matrix1.cpu_data(), nullptr);
    EXPECT_NE(matrix2.gpu_data(), nullptr);
    EXPECT_NE(matrix1.gpu_data(), nullptr); 

    EXPECT_EQ(matrix2.cpu_data(), array1);
    EXPECT_EQ(matrix1.cpu_data(), array2);

    std::vector<T> gpu_copied_1(matrix1.rows() * matrix1.cols());
    std::vector<T> gpu_copied_2(matrix2.rows() * matrix2.cols());

    CHECK_CUBLAS_ERROR(cublasGetMatrix(matrix1.rows(), matrix1.cols(), sizeof(T),
                        matrix1.gpu_data(), matrix1.gpu_ld(), gpu_copied_1.data(), matrix1.rows()));

    CHECK_CUBLAS_ERROR(cublasGetMatrix(matrix2.rows(), matrix2.cols(), sizeof(T),
                        matrix2.gpu_data(), matrix2.gpu_ld(), gpu_copied_2.data(), matrix2.rows()));

    for(auto i = 0; i < matrix1.rows() * matrix1.cols(); i++)
    {
        gpu_copied_1.data()[i] = array2[i];
    }

    for(auto i = 0; i < matrix2.rows() * matrix2.cols(); i++)
    {
        gpu_copied_2.data()[i] = array1[i];
    }
}

// Test for saveToBinaryFile
TYPED_TEST(MatrixGPUTest, SaveToBinaryFile) {
    using T = TypeParam;
    
    // Sample matrix data
    std::size_t rows = 3;
    std::size_t cols = 3;
    std::size_t ld = 4;
    std::vector<T> sampleData = {1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0}; // 3x3 matrix data

    // Create and save matrix
    chase::matrix::MatrixGPU<T> matrix(rows, cols, ld, sampleData.data());

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
TYPED_TEST(MatrixGPUTest, ReadFromBinaryFile) {
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
    chase::matrix::MatrixGPU<T> matrix(rows, cols, ld, buffer.data());
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
TYPED_TEST(MatrixGPUTest, EnableSinglePrecision) {
    using T = TypeParam;  // Get the current type
    using singlePrecisionT = typename chase::ToSinglePrecisionTrait<T>::Type;
    if constexpr (std::is_same<T, double>::value || std::is_same<T, std::complex<double>>::value) {
        std::size_t rows = 3, cols = 3;
        std::vector<T> data(rows * cols);

        for (std::size_t i = 0; i < rows * cols; ++i) {
            data[i] = T(i + 1);
        }

        chase::matrix::MatrixGPU<T> matrix(rows, cols, rows, data.data());

        // Enable single precision
        matrix.enableSinglePrecision();
        EXPECT_TRUE(matrix.isSinglePrecisionEnabled());
        auto sp_data = matrix.matrix_sp();
        sp_data->allocate_cpu_data();
        sp_data->D2H();
        // Check if the single precision data is correctly converted
        for (std::size_t i = 0; i < rows * cols; ++i) {
            EXPECT_EQ(static_cast<singlePrecisionT>(matrix.cpu_data()[i]), sp_data->cpu_data()[i]);
        }
    }
}

TYPED_TEST(MatrixGPUTest, DisableSinglePrecisionWithCopyback) {
    using T = TypeParam;  // Get the current type
    if constexpr (std::is_same<T, double>::value || std::is_same<T, std::complex<double>>::value) {
        std::size_t rows = 3, cols = 3;
        std::vector<T> data(rows * cols);
        for (std::size_t i = 0; i < rows * cols; ++i) {
            data[i] = T(i + 1);
        }        
        chase::matrix::MatrixGPU<T> matrix(rows, cols, rows, data.data());

        // Enable and then disable single precision with copyback
        matrix.enableSinglePrecision();
        CHECK_CUDA_ERROR(cudaMemset(matrix.gpu_data(), 0, rows * cols * sizeof(T)));
        memset(matrix.cpu_data(), 0, rows * cols * sizeof(T));

        matrix.disableSinglePrecision(true);
        EXPECT_FALSE(matrix.isSinglePrecisionEnabled());
        matrix.D2H();
        // Check if the data was restored correctly
        for (std::size_t i = 0; i < rows * cols; ++i) {
            EXPECT_EQ(matrix.cpu_data()[i], T(i + 1));
        }
    }
}

TYPED_TEST(MatrixGPUTest, UnsupportedSinglePrecisionOperation) {
    using T = TypeParam;  // Get the current type
    if constexpr (std::is_same<T, float>::value || std::is_same<T, std::complex<float>>::value) {
        std::size_t rows = 3, cols = 3;
        chase::matrix::MatrixGPU<T> matrix(rows, cols);

        EXPECT_THROW(matrix.enableSinglePrecision(), std::runtime_error);
        EXPECT_THROW(matrix.disableSinglePrecision(), std::runtime_error);
    }
}

TYPED_TEST(MatrixGPUTest, SinglePrecisionMemoryDeallocation) {
    using T = TypeParam;  // Get the current type
    if constexpr (std::is_same<T, double>::value || std::is_same<T, std::complex<double>>::value) {
        std::size_t rows = 3, cols = 3;
        chase::matrix::MatrixGPU<T> matrix(rows, cols);

        matrix.enableSinglePrecision();
        EXPECT_NE(matrix.matrix_sp()->gpu_data(), nullptr);

        matrix.disableSinglePrecision();
        EXPECT_EQ(matrix.matrix_sp(), nullptr);  // Check if the memory is deallocated
    }
}

#endif
#endif
