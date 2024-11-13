#pragma once

#include <vector>
#include <random>
#include "algorithm/types.hpp"
#include "external/blaspp/blaspp.hpp"
#include "external/lapackpp/lapackpp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/mpi/hemm.hpp"
#include "../typeTraits.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace mpi
{
    /**
    * @brief Checks if a matrix is symmetric using a randomized approach.
    *
    * This function checks the symmetry of a square matrix \( H \) by performing two matrix-vector multiplications:
    * 1. It computes \( u = H \cdot v \), where \( v \) is a random vector.
    * 2. It computes \( uT = H^T \cdot v \), where \( H^T \) is the transpose of \( H \).
    * The matrix is considered symmetric if the vectors \( u \) and \( uT \) are the same, i.e., \( u = uT \).
    *
    * This method is computationally efficient and uses random vectors to test symmetry with high probability.
    * However, it is not a guarantee for exact symmetry due to numerical errors, but it can be a quick heuristic check.
    *
    * @tparam T Data type for the matrix (e.g., float, double).
    * @param[in] N The size of the matrix (N x N).
    * @param[in] H The matrix to be checked for symmetry (of size N x N).
    * @param[in] ldh The leading dimension of the matrix H.
    * @return `true` if the matrix is symmetric, `false` otherwise.
    */ 
    template<typename MatrixType>
    bool checkSymmetryEasy(MatrixType& H) 
    {

        using T = typename MatrixType::value_type;
        using ColumnMultiVectorType = typename ColumnMultiVectorType<MatrixType>::type;
        using RowMultiVectorType = typename RowMultiVectorType<MatrixType>::type;

        T One = T(1.0);
        T Zero = T(0.0);

        std::size_t N = H.g_rows();
        auto v = H.template cloneMultiVector<ColumnMultiVectorType>(N, 1);
        auto u = H.template cloneMultiVector<RowMultiVectorType>(N, 1);
        auto uT = H.template cloneMultiVector<ColumnMultiVectorType>(N, 1);
        auto v_2 = H.template cloneMultiVector<RowMultiVectorType>(N, 1);

        int *coords = H.getMpiGrid()->get_coords();

        std::mt19937 gen(1337.0 + coords[0]);
        std::normal_distribution<> d;
        
        for (auto i = 0; i < v.l_ld() * v.l_cols(); i++)
        {
            v.l_data()[i] = getRandomT<T>([&]() { return d(gen); });
        }

        v.redistributeImpl(&v_2);

        MatrixMultiplyMultiVectors(&One, 
                                       H,
                                       v,
                                       &Zero,
                                       u);

        MatrixMultiplyMultiVectors(&One, 
                                       H,
                                       v_2,
                                       &Zero,
                                       uT);

        u.redistributeImpl(&v);

        bool is_sym = true;

        for(auto i = 0; i < v.l_rows(); i++)
        {
            if(std::abs(v.l_data()[i] - uT.l_data()[i]) > 1e-10)
            {
                is_sym = false;
                break;
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, 
                      &is_sym, 
                      1, 
                      MPI_CXX_BOOL, 
                      MPI_LAND, 
                      H.getMpiGrid()->get_col_comm());

        return is_sym;

    }
    /**
    * @brief Converts a BlockBlock matrix to its Hermitian or symmetric form based on the given `uplo` argument.
    *
    * This function modifies the matrix \( H \) in-place such that it becomes symmetric or Hermitian, depending on the value of the `uplo` parameter.
    * - If `uplo` is `'U'`, the function converts the upper triangular part of the matrix to the Hermitian form, by setting the lower triangular part to the conjugate transpose of the upper part.
    * - If `uplo` is `'L'`, the function converts the lower triangular part of the matrix to the Hermitian form, by setting the upper triangular part to the conjugate transpose of the lower part.
    *
    * The function assumes that the matrix is square (N x N) and modifies the elements of the matrix in-place. The conjugation is done using the `conjugate` function.
    *
    * @tparam T Data type for the matrix (e.g., float, double, std::complex).
    * @param[in] uplo A character indicating which part of the matrix to modify:
    * - `'U'` for the upper triangular part.
    * - `'L'` for the lower triangular part.
    * @param[in,out] N The size of the matrix (N x N). The matrix is modified in-place.
    * @param[in,out] H The matrix to be modified. It is transformed into a symmetric or Hermitian matrix.
    * @param[in] ldh The leading dimension of the matrix H.
    */
    template<typename T>
    void symOrHermMatrix(char uplo, chase::distMatrix::BlockBlockMatrix<T>& H) 
    {
#ifdef HAS_SCALAPACK
        std::size_t *desc = H.scalapack_descriptor_init();

        std::size_t xlen = H.l_rows();
        std::size_t ylen = H.l_cols();
        std::size_t *g_offs = H.g_offs();
        std::size_t h_ld = H.l_ld();

        if(uplo == 'U')
        {
            #pragma omp parallel for
            for(auto j = 0; j < ylen; j++)
            {
                for(auto i = 0; i < xlen; i++)
                {
                    if(g_offs[0] + i == g_offs[1] + j)
                    {
                        H.l_data()[i + j * h_ld ] *= T(0.5);
                    }
                    else if(g_offs[0] + i > g_offs[1] + j)
                    {
                        H.l_data()[i + j * h_ld ] = T(0.0);
                    }
                }
            }
        }
        else
        {
            #pragma omp parallel for
            for(auto j = 0; j < ylen; j++)
            {
                for(auto i = 0; i < xlen; i++)
                {
                    if(g_offs[0] + i == g_offs[1] + j)
                    {
                        H.l_data()[i + j * h_ld ] *= T(0.5);
                    }
                    else if(g_offs[0] + i < g_offs[1] + j)
                    {
                        H.l_data()[i + j * h_ld ] = T(0.0);
                    }
                }
            }
        }

        T One = T(1.0);
        T Zero = T(0.0);
        int zero = 0;
        int one = 1;
        std::vector<T> tmp(H.l_rows() * (H.l_cols() + 2*H.l_cols()));
        chase::linalg::scalapackpp::t_ptranc(H.g_rows(), 
                                             H.g_cols(), 
                                             One, 
                                             H.l_data(), 
                                             one, one, desc, Zero, tmp.data(), one, one, desc);
        #pragma omp parallel for
        for(auto j = 0; j < H.l_cols(); j++)
        {
            for(auto i = 0; i < H.l_rows(); i++)
            {
                H.l_data()[i + j * H.l_ld()] += tmp[i + j * H.l_rows()];
            }
        }      
#else
        std::runtime_error("For ChASE-MPI, symOrHermMatrix requires ScaLAPACK, which is not detected\n");
#endif
    }

    /**
    * @brief Converts a BlockCyclic matrix to its Hermitian or symmetric form based on the given `uplo` argument.
    *
    * This function modifies the matrix \( H \) in-place such that it becomes symmetric or Hermitian, depending on the value of the `uplo` parameter.
    * - If `uplo` is `'U'`, the function converts the upper triangular part of the matrix to the Hermitian form, by setting the lower triangular part to the conjugate transpose of the upper part.
    * - If `uplo` is `'L'`, the function converts the lower triangular part of the matrix to the Hermitian form, by setting the upper triangular part to the conjugate transpose of the lower part.
    *
    * The function assumes that the matrix is square (N x N) and modifies the elements of the matrix in-place. The conjugation is done using the `conjugate` function.
    *
    * @tparam T Data type for the matrix (e.g., float, double, std::complex).
    * @param[in] uplo A character indicating which part of the matrix to modify:
    * - `'U'` for the upper triangular part.
    * - `'L'` for the lower triangular part.
    * @param[in,out] N The size of the matrix (N x N). The matrix is modified in-place.
    * @param[in,out] H The matrix to be modified. It is transformed into a symmetric or Hermitian matrix.
    * @param[in] ldh The leading dimension of the matrix H.
    */
    template<typename T>
    void symOrHermMatrix(char uplo, chase::distMatrix::BlockCyclicMatrix<T>& H) 
    {
#ifdef HAS_SCALAPACK
        std::size_t *desc = H.scalapack_descriptor_init();

        auto m_contiguous_global_offs = H.m_contiguous_global_offs();
        auto n_contiguous_global_offs = H.n_contiguous_global_offs();
        auto m_contiguous_local_offs = H.m_contiguous_local_offs();
        auto n_contiguous_local_offs = H.n_contiguous_local_offs();
        auto m_contiguous_lens = H.m_contiguous_lens();
        auto n_contiguous_lens = H.n_contiguous_lens();
        auto mblocks = H.mblocks();
        auto nblocks = H.nblocks();

        if(uplo == 'U')
        {
            for(std::size_t j = 0; j < nblocks; j++){
                for(std::size_t i = 0; i < mblocks; i++){
                    for(std::size_t q = 0; q < n_contiguous_lens[j]; q++){
                        for(std::size_t p = 0; p < m_contiguous_lens[i]; p++){
                            if(q + n_contiguous_global_offs[j] == p + m_contiguous_global_offs[i]){
                                H.l_data()[(q + n_contiguous_local_offs[j]) * H.l_ld() + p + m_contiguous_local_offs[i]] *= T(0.5);
                            }
                            else if(q + n_contiguous_global_offs[j] < p + m_contiguous_global_offs[i])
                            {
                                H.l_data()[(q + n_contiguous_local_offs[j]) * H.l_ld()  + p + m_contiguous_local_offs[i]] = T(0.0);
                            }
                        }
                    }
                }
            }

        }else
        {
            for(std::size_t j = 0; j < nblocks; j++){
                for(std::size_t i = 0; i < mblocks; i++){
                    for(std::size_t q = 0; q < n_contiguous_lens[j]; q++){
                        for(std::size_t p = 0; p < m_contiguous_lens[i]; p++){
                            if(q + n_contiguous_global_offs[j] == p + m_contiguous_global_offs[i]){
                                H.l_data()[(q + n_contiguous_local_offs[j]) * H.l_ld()  + p + m_contiguous_global_offs[i]] *= T(0.5);
                            }
                            else if(q + n_contiguous_global_offs[j] > p + m_contiguous_global_offs[i])
                            {
                                H.l_data()[(q + n_contiguous_local_offs[j]) * H.l_ld()  + p + m_contiguous_local_offs[i]] = T(0.0);
                            }
                        }
                    }
                }
            }
        }

        
        T One = T(1.0);
        T Zero = T(0.0);
        int zero = 0;
        int one = 1;
        std::vector<T> tmp(H.l_rows() * (H.l_cols() + 2*H.nb()));
        chase::linalg::scalapackpp::t_ptranc(H.g_rows(), 
                                             H.g_cols(), 
                                             One, 
                                             H.l_data(), 
                                             one, one, desc, Zero, tmp.data(), one, one, desc);
        #pragma omp parallel for
        for(auto j = 0; j < H.l_cols(); j++)
        {
            for(auto i = 0; i < H.l_rows(); i++)
            {
                H.l_data()[i + j * H.l_ld()] += tmp[i + j * H.l_rows()];
            }
        }
              
#else
        std::runtime_error("For ChASE-MPI, symOrHermMatrix requires ScaLAPACK, which is not detected\n");
#endif
    }


}
}
}
}