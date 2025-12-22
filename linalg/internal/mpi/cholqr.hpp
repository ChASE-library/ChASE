// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <limits>
#include <iomanip>
#include "mpi.h"
#include "grid/mpiTypes.hpp"
#include "external/blaspp/blaspp.hpp"
#include "external/lapackpp/lapackpp.hpp"
#include "linalg/internal/cpu/utils.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/mpi/mpi_kernels.hpp"

using namespace chase::linalg::blaspp;
using namespace chase::linalg::lapackpp;

namespace chase
{
namespace linalg
{
namespace internal
{
    /**
     * @brief Performs a one-step Cholesky-based QR decomposition.
     * 
     * Computes a Cholesky factorization-based QR decomposition for the input matrix `V` in an MPI-distributed setting. 
     * If `A` is not provided, it is allocated internally.
     * 
     * @tparam T The data type of the matrix elements.
     * 
     * @param m The number of rows of matrix `V`.
     * @param n The number of columns of matrix `V`.
     * @param V Pointer to the matrix data in column-major order.
     * @param ldv The leading dimension of matrix `V`.
     * @param comm The MPI communicator for distributed operations.
     * @param A Optional pointer to an allocated `n x n` matrix used in the Cholesky factorization.
     * 
     * @return int Returns 0 on success; non-zero on failure.
     */    
    template<typename T>
    int cpu_mpi::cholQR1(std::size_t m, std::size_t n, T *V, int ldv, MPI_Comm comm, T *A)
    {
        T one = T(1.0);
        T zero = T(0.0);
        int info = 1;
        
        std::unique_ptr<T[]> ptr;

        if (A == nullptr)
        {
            ptr = std::unique_ptr<T[]>{new T[n * n]};
            A = ptr.get();
        }

        blaspp::t_syherk('U', 
                         'C', 
                         n, 
                         m, 
                         &one, 
                         V, 
                         ldv, 
                         &zero, 
                         A, 
                         n);

        MPI_Allreduce(MPI_IN_PLACE, 
                      A, 
                      n * n, 
                      chase::mpi::getMPI_Type<T>(),
                      MPI_SUM,
                      comm);

        info = lapackpp::t_potrf('U', 
                                 n, 
                                 A, 
                                 n); 

        if(info != 0)
        {
            return info;
        }
        else
        {
            blaspp::t_trsm('R', 
                           'U', 
                           'N', 
                           'N', 
                           m, 
                           n, 
                           &one, 
                           A, 
                           n, 
                           V, 
                           ldv); 
#ifdef CHASE_OUTPUT
            int grank;
            MPI_Comm_rank(MPI_COMM_WORLD, &grank);
            if(grank == 0)
            {
                std::cout << "choldegree: 1" << std::endl;
            }
#endif      
            return info;        
        }
    }

    /**
     * @brief Performs a two-step Cholesky-based QR decomposition.
     * 
     * Computes a Cholesky factorization-based QR decomposition with two Cholesky steps 
     * for the input matrix `V` in an MPI-distributed setting. If `A` is not provided, it is allocated internally.
     * 
     * @tparam T The data type of the matrix elements.
     * 
     * @param m The number of rows of matrix `V`.
     * @param n The number of columns of matrix `V`.
     * @param V Pointer to the matrix data in column-major order.
     * @param ldv The leading dimension of matrix `V`.
     * @param comm The MPI communicator for distributed operations.
     * @param A Optional pointer to an allocated `n x n` matrix used in the Cholesky factorization.
     * 
     * @return int Returns 0 on success; non-zero on failure.
     */
    template<typename T>
    int cpu_mpi::cholQR2(std::size_t m, std::size_t n, T *V, int ldv, MPI_Comm comm, T *A)
    {
        T one = T(1.0);
        T zero = T(0.0);
        int info = 1;
        
        std::unique_ptr<T[]> ptr;

        if (A == nullptr)
        {
            ptr = std::unique_ptr<T[]>{new T[n * n]};
            A = ptr.get();
        }

        blaspp::t_syherk('U', 
                         'C', 
                         n, 
                         m, 
                         &one, 
                         V, 
                         ldv, 
                         &zero, 
                         A, 
                         n);

        MPI_Allreduce(MPI_IN_PLACE, 
                      A, 
                      n * n, 
                      chase::mpi::getMPI_Type<T>(),
                      MPI_SUM,
                      comm);        
        info = lapackpp::t_potrf('U', 
                                 n, 
                                 A, 
                                 n); 
        if(info != 0)
        {
            return info;
        }
        else
        {
            blaspp::t_trsm('R', 
                           'U', 
                           'N', 
                           'N', 
                           m, 
                           n, 
                           &one, 
                           A, 
                           n, 
                           V, 
                           ldv);

            blaspp::t_syherk('U', 
                             'C', 
                             n, 
                             m, 
                             &one, 
                             V, 
                             ldv, 
                             &zero, 
                             A, 
                             n);

            MPI_Allreduce(MPI_IN_PLACE, 
                        A, 
                        n * n, 
                        chase::mpi::getMPI_Type<T>(),
                        MPI_SUM,
                        comm);            
            info = lapackpp::t_potrf('U', 
                                     n, 
                                     A, 
                                     n); 
            blaspp::t_trsm('R', 
                           'U', 
                           'N', 
                           'N', 
                           m, 
                           n, 
                           &one, 
                           A, 
                           n, 
                           V, 
                           ldv); 
#ifdef CHASE_OUTPUT
            int grank;
            MPI_Comm_rank(MPI_COMM_WORLD, &grank);
            if(grank == 0)
            {
                std::cout << "choldegree: 2" << std::endl;
            }
#endif                    
            return info;              
        }
    }

    /**
     * @brief Variant of cholQR1 for InputMultiVectorType.
     * 
     * This variant works with InputMultiVectorType and performs the Cholesky QR decomposition in the same
     * manner as cholQR1, but with different input data type handling. The use of MPI ensures that the 
     * computation is parallelized across multiple processes.
     * 
     * @tparam InputMultiVectorType The type of the input multi-vector (e.g., a distributed matrix type).
     * 
     * @param V The input matrix to decompose.
     * @param A Optional pointer to an allocated matrix used in the Cholesky factorization.
     * 
     * @return int Status code: 0 for success, non-zero on failure.
     */
    template<typename InputMultiVectorType>
    int cpu_mpi::cholQR1(InputMultiVectorType& V, 
                        typename InputMultiVectorType::value_type *A)
    {
        using T = typename InputMultiVectorType::value_type;

        T one = T(1.0);
        T zero = T(0.0);
        int info = 1;
        
        std::unique_ptr<T[]> ptr;

        if (A == nullptr)
        {
            ptr = std::unique_ptr<T[]>{new T[V.l_cols() * V.l_cols()]};
            A = ptr.get();
        }

        blaspp::t_syherk('U', 
                         'C', 
                         V.l_cols(), 
                         V.l_rows(), 
                         &one, 
                         V.l_data(), 
                         V.l_ld(), 
                         &zero, 
                         A, 
                         V.l_cols());

        MPI_Allreduce(MPI_IN_PLACE, 
                      A, 
                      V.l_cols() * V.l_cols(), 
                      chase::mpi::getMPI_Type<T>(),
                      MPI_SUM,
                      V.getMpiGrid()->get_col_comm());

        info = lapackpp::t_potrf('U', 
                                 V.l_cols(), 
                                 A, 
                                 V.l_cols()); 
   
        if(info != 0)
        {
            return info;
        }
        else
        {
            blaspp::t_trsm('R', 
                           'U', 
                           'N', 
                           'N', 
                           V.l_rows(), 
                           V.l_cols(), 
                           &one, 
                           A, 
                           V.l_cols(), 
                           V.l_data(), 
                           V.l_ld()); 
#ifdef CHASE_OUTPUT
            int grank;
            MPI_Comm_rank(MPI_COMM_WORLD, &grank);
            if(grank == 0)
            {
                std::cout << "choldegree: 1" << std::endl;
            }
#endif      
            return info;              
        }
    }

    /**
     * @brief Performs a Cholesky QR decomposition on an input multi-vector type.
     * 
     * This function computes the Cholesky QR decomposition of the input matrix or multi-vector `V`
     * using BLAS and LAPACK on the CPU. It is designed to work with multi-vector input types 
     * where `V` can be a matrix or a vector, and the decomposition is performed in parallel on multiple CPUs.
     * The function also allows for memory optimization through an optional workspace buffer for 
     * intermediate calculations.
     * 
     * @tparam InputMultiVectorType The type of the input multi-vector (e.g., a matrix or vector type).
     * 
     * @param V The input multi-vector (matrix or vector) to decompose. It will be modified during the process.
     * @param A Optional matrix to store the result of the factorization. If not provided, one will be 
     *        allocated internally.
     * 
     * @return int Status code indicating the success or failure of the computation.
     *         - 0 for success.
     *         - Non-zero value indicates failure.
     * 
     * @note This function assumes the input multi-vector `V` is stored in a format compatible with
     *       BLAS and LAPACK. The input matrix must be stored in column-major format.
     * 
     * @note This function supports mixed precision when ENABLE_MIXED_PRECISION is defined. For double
     *       precision types, it can use single precision for the first Cholesky QR step to improve
     *       performance while maintaining accuracy.
     */
    template<typename InputMultiVectorType>
    int cpu_mpi::cholQR2(InputMultiVectorType& V, 
                        typename InputMultiVectorType::value_type *A)
    {
        using T = typename InputMultiVectorType::value_type;

        T one = T(1.0);
        T zero = T(0.0);
        int info = 0;
        
        std::unique_ptr<T[]> ptr;

        if (A == nullptr)
        {
            ptr = std::unique_ptr<T[]>{new T[V.l_cols() * V.l_cols()]};
            A = ptr.get();
        }

#ifdef ENABLE_MIXED_PRECISION
        if constexpr (std::is_same<T, double>::value || std::is_same<T, std::complex<double>>::value)
        {
            std::cout << "In cholqr2, the first cholqr using Single Precision" << std::endl;
            V.enableSinglePrecision();
            auto V_sp = V.getSinglePrecisionMatrix();
            info = cholQR1(*V_sp);
            V.disableSinglePrecision(true);
        }else
        {
            info = cholQR1(V, A);
        }      
#else
        info = cholQR1(V, A);
#endif
        if(info != 0)
        {
            return info;
        }

        info = cholQR1(V, A);

        return info;       
    }

    /**
     * @brief Performs a two-step shifted Cholesky-based QR decomposition.
     * 
     * Computes a shifted Cholesky factorization-based QR decomposition with two Cholesky steps 
     * for the input matrix `V` in an MPI-distributed setting. If `A` is not provided, it is allocated internally.
     * 
     * @tparam T The data type of the matrix elements.
     * 
     * @param N The size parameter used for calculating the diagonal shift.
     * @param m The number of rows of matrix `V`.
     * @param n The number of columns of matrix `V`.
     * @param V Pointer to the matrix data in column-major order.
     * @param ldv The leading dimension of matrix `V`.
     * @param comm The MPI communicator for distributed operations.
     * @param A Optional pointer to an allocated `n x n` matrix used in the Cholesky factorization.
     * 
     * @return int Returns 0 on success; non-zero on failure.
     */
    template<typename T>
    int cpu_mpi::shiftedcholQR2(std::size_t N, std::size_t m, std::size_t n, T *V, int ldv, MPI_Comm comm, T *A)
    { 
        Base<T> shift;
        T one = T(1.0);
        T zero = T(0.0);
        int info = 1;
        
        std::unique_ptr<T[]> ptr;

        if (A == nullptr)
        {
            ptr = std::unique_ptr<T[]>{new T[n * n]};
            A = ptr.get();
        }

        blaspp::t_syherk('U', 
                         'C', 
                         n, 
                         m, 
                         &one, 
                         V, 
                         ldv, 
                         &zero, 
                         A, 
                         n);
        MPI_Allreduce(MPI_IN_PLACE, 
                      A, 
                      n * n, 
                      chase::mpi::getMPI_Type<T>(),
                      MPI_SUM,
                      comm);           
        Base<T> nrmf = 0.0;
        chase::linalg::internal::cpu::computeDiagonalAbsSum(n, n, A, n, &nrmf);
        shift = std::sqrt(N) * nrmf * std::numeric_limits<Base<T>>::epsilon();
        chase::linalg::internal::cpu::shiftMatrixDiagonal(n, n, A, n, (T)shift);
        info = lapackpp::t_potrf('U', 
                                 n, 
                                 A, 
                                 n); 
        if(info != 0)
        {
            return info;
        }
        else
        {
            blaspp::t_trsm('R', 
                           'U', 
                           'N', 
                           'N', 
                           m, 
                           n, 
                           &one, 
                           A, 
                           n, 
                           V, 
                           ldv); 
            blaspp::t_syherk('U', 
                             'C', 
                             n, 
                             m, 
                             &one, 
                             V, 
                             ldv, 
                             &zero, 
                             A, 
                             n);
            MPI_Allreduce(MPI_IN_PLACE, 
                        A, 
                        n * n, 
                        chase::mpi::getMPI_Type<T>(),
                        MPI_SUM,
                        comm);               
            info = lapackpp::t_potrf('U', 
                                     n, 
                                     A, 
                                     n); 
            blaspp::t_trsm('R', 
                           'U', 
                           'N', 
                           'N', 
                           m, 
                           n, 
                           &one, 
                           A, 
                           n, 
                           V, 
                           ldv); 
            blaspp::t_syherk('U', 
                             'C', 
                             n, 
                             m, 
                             &one, 
                             V, 
                             ldv, 
                             &zero, 
                             A, 
                             n);
            MPI_Allreduce(MPI_IN_PLACE, 
                        A, 
                        n * n, 
                        chase::mpi::getMPI_Type<T>(),
                        MPI_SUM,
                        comm);               
            info = lapackpp::t_potrf('U', 
                                     n, 
                                     A, 
                                     n); 
            blaspp::t_trsm('R', 
                           'U', 
                           'N', 
                           'N', 
                           m, 
                           n, 
                           &one, 
                           A, 
                           n, 
                           V, 
                           ldv);  

#ifdef CHASE_OUTPUT
            int grank;
            MPI_Comm_rank(MPI_COMM_WORLD, &grank);
            if(grank == 0)
            {
                std::cout << "choldegree: 2, shift = " << shift << std::endl;
            }
#endif
            return info;                        
        }

    }

    /**
     * @brief Computes the Householder QR decomposition for a distributed multi-vector.
     * 
     * This function performs a QR factorization using Householder transformations on the input multi-vector `V`.
     * Requires ScaLAPACK to be available for distributed QR computations.
     * 
     * @tparam InputMultiVectorType The type of the distributed multi-vector, containing matrix data.
     * 
     * @param V The distributed multi-vector on which QR decomposition is performed.
     * 
     * @throws std::runtime_error If ScaLAPACK is not available.
     */
    template<typename InputMultiVectorType>
    void cpu_mpi::houseHoulderQR(InputMultiVectorType& V)
    {
        using T = typename InputMultiVectorType::value_type;

#ifdef HAS_SCALAPACK
        std::size_t *desc = V.scalapack_descriptor_init();
        int one = 1;
        std::vector<T> tau(V.l_cols());

        chase::linalg::scalapackpp::t_pgeqrf(V.g_rows(), 
                                             V.g_cols(), 
                                             V.l_data(), 
                                             one, 
                                             one, 
                                             desc,
                                             tau.data());

        chase::linalg::scalapackpp::t_pgqr(V.g_rows(), 
                                           V.g_cols(), 
                                           V.g_cols(), 
                                           V.l_data(), 
                                           one, 
                                           one, 
                                           desc, 
                                           tau.data());

#else
        std::runtime_error("For ChASE-MPI, distributed Householder QR requires ScaLAPACK, which is not detected\n");
#endif
    }

    /**
     * @brief Computes the condition number of a distributed matrix using ScaLAPACK SVD.
     *
     * This function computes the condition number of a distributed matrix \( V \) using the singular value decomposition (SVD).
     * The condition number is defined as the ratio of the largest to the smallest singular value: cond(V) = σ_max / σ_min.
     * It first allocates and transfers the data to the CPU if not already allocated, performs the SVD using ScaLAPACK routines,
     * and then computes the condition number from the singular values.
     * 
     * @tparam InputMultiVectorType The type of the input multi-vector (e.g., a distributed matrix).
     * 
     * @param[in] V The input distributed matrix. The matrix should be in column-major order. Note: the matrix content will be destroyed during SVD computation.
     * 
     * @return chase::Base<typename InputMultiVectorType::value_type> The condition number of the matrix.
     * 
     * @throws std::runtime_error If ScaLAPACK is not available (i.e., for ChASE-MPI builds without ScaLAPACK support).
     * 
     * @note This function requires ScaLAPACK for distributed SVD. If ScaLAPACK is not available, a runtime error will be thrown.
     * @note The input matrix V will be destroyed during the SVD computation. Make a copy if the original matrix needs to be preserved.
     * @note If the matrix is rank-deficient (smallest singular value is zero or near machine precision), the condition number will be very large or infinite.
     * 
     * @par ScaLAPACK Functions Used:
     *   - `t_pgesvd`: Computes the singular value decomposition of the matrix.
     */
    template<typename InputMultiVectorType>
    chase::Base<typename InputMultiVectorType::value_type> cpu_mpi::computeConditionNumber(InputMultiVectorType& V)
    {
        using T = typename InputMultiVectorType::value_type;
        using BaseT = chase::Base<T>;

#ifdef HAS_SCALAPACK
        std::size_t *desc_a = V.scalapack_descriptor_init();
        int one = 1;
        int info = 0;
        
        std::size_t m = V.g_rows();
        std::size_t n = V.g_cols();
        std::size_t min_mn = std::min(m, n);
        
        // Get MPI rank for debug output
        int grank;
        MPI_Comm_rank(MPI_COMM_WORLD, &grank);
        
        // Allocate arrays for singular values
        std::vector<BaseT> s(min_mn);
        
        // Compute only singular values to avoid descriptor issues
        char jobu = 'N';   // Don't compute U
        char jobvt = 'N';  // Don't compute VT
        
        // Create minimal dummy arrays (not used but required by template)
        T dummy_u = T(0);
        T dummy_vt = T(0);
        
        // Create minimal dummy descriptors
        std::size_t desc_u[9];
        std::size_t desc_vt[9];
        for (int i = 0; i < 9; i++) {
            desc_u[i] = desc_a[i];   // Copy input descriptor
            desc_vt[i] = desc_a[i];  // Copy input descriptor
        }
        
        // Basic parameter validation
        if (m == 0 || n == 0) {
            if (grank == 0) {
                std::cout << "Error: Invalid matrix dimensions m=" << m << ", n=" << n << std::endl;
            }
            return std::numeric_limits<BaseT>::infinity();
        }
        
        // Call ScaLAPACK SVD function
        chase::linalg::scalapackpp::t_pgesvd(jobu, jobvt, m, n, 
                                             V.l_data(), 
                                             one, one, desc_a,
                                             s.data(),
                                             &dummy_u, one, one, desc_u,        // U not computed
                                             &dummy_vt, one, one, desc_vt,      // VT not computed
                                             &info);
        
        if (info != 0) {
            if (grank == 0) {
                std::cout << "SVD computation failed with info = " << info << std::endl;
                std::cout << "SVD Error Details:" << std::endl;
                std::cout << "  Matrix dimensions: " << m << "x" << n << std::endl;
                std::cout << "  jobu = '" << jobu << "', jobvt = '" << jobvt << "'" << std::endl;
                std::cout << "  Descriptor A: [";
                for (int i = 0; i < 9; i++) {
                    std::cout << desc_a[i];
                    if (i < 8) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
                
                // Print common ScaLAPACK error meanings
                if (info < 0) {
                    std::cout << "  Error: Parameter " << -info << " had an illegal value" << std::endl;
                } else if (info > 0) {
                    std::cout << "  Error: " << info << " superdiagonals of an intermediate bidiagonal form did not converge" << std::endl;
                }
            }
            // Return infinity for failed SVD computation
            return std::numeric_limits<BaseT>::infinity();
        }
#ifdef CHASE_OUTPUT        
        // Debug: Print singular values to understand what's happening
        if (grank == 0 && min_mn > 0) {
            std::cout << "SVD Debug: Matrix size " << m << "x" << n << ", min_mn=" << min_mn << std::endl;
            std::cout << "First 5 singular values: ";
            for (int i = 0; i < std::min(5, (int)min_mn); i++) {
                std::cout << s[i] << " ";
            }
            std::cout << std::endl;
            if (min_mn > 5) {
                std::cout << "Last 5 singular values: ";
                for (int i = std::max(0, (int)min_mn - 5); i < (int)min_mn; i++) {
                    std::cout << s[i] << " ";
                }
                std::cout << std::endl;
            }
        }
#endif
        // Compute condition number as ratio of largest to smallest singular value
        BaseT cond_num = std::numeric_limits<BaseT>::infinity();
        
        if (min_mn > 0) {
            BaseT sigma_max = s[0];  // Singular values are sorted in descending order
            BaseT sigma_min = s[min_mn - 1];
            
            // Check for rank deficiency
            const BaseT eps = std::numeric_limits<BaseT>::epsilon();
            const BaseT tolerance = std::max(m, n) * sigma_max * eps;
#ifdef CHASE_OUTPUT            
            if (grank == 0) {
                std::cout << "Condition number debug: sigma_max=" << sigma_max 
                          << ", sigma_min=" << sigma_min 
                          << ", tolerance=" << tolerance 
                          << ", eps=" << eps << std::endl;
            }
#endif
            if (sigma_min > tolerance && sigma_min > 0) {
                cond_num = sigma_max / sigma_min;
            }
            // If sigma_min is too small, condition number remains infinity
        }
        
        return cond_num;

#else
        throw std::runtime_error("For ChASE-MPI, computing condition number requires ScaLAPACK, which is not detected\n");
#endif
    }

}
}
}