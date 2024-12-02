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

}
}
}