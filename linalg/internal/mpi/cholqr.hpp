#pragma once

#include "mpi.h"
#include "Impl/mpi/mpiTypes.hpp"
#include "linalg/blaspp/blaspp.hpp"
#include "linalg/lapackpp/lapackpp.hpp"
#include "linalg/internal/cpu/utils.hpp"
#include "linalg/matrix/distMatrix.hpp"
#include "linalg/matrix/distMultiVector.hpp"

using namespace chase::linalg::blaspp;
using namespace chase::linalg::lapackpp;

namespace chase
{
namespace linalg
{
namespace internal
{
namespace mpi
{
    template<typename T>
    int cholQR1(std::size_t m, std::size_t n, T *V, int ldv, MPI_Comm comm, T *A = nullptr)
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


    template<typename T>
    int cholQR2(std::size_t m, std::size_t n, T *V, int ldv, MPI_Comm comm, T *A = nullptr)
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

    template<typename T>
    int shiftedcholQR2(std::size_t N, std::size_t m, std::size_t n, T *V, int ldv, MPI_Comm comm, T *A = nullptr)
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

    template<typename T, chase::distMultiVector::CommunicatorType InputCommType>
    void houseHoulderQR(chase::distMultiVector::DistMultiVector1D<T, InputCommType>& V)
    {
#ifdef HAS_SCALAPACK
        std::size_t *desc = V.get_scalapack_desc();
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
}