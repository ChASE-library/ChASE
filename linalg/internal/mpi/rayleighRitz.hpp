#pragma once

#include <cassert>  // For assertions
#include "mpi.h"
#include "Impl/mpi/mpiTypes.hpp"
#include "linalg/blaspp/blaspp.hpp"
#include "linalg/lapackpp/lapackpp.hpp"
#include "linalg/matrix/distMatrix.hpp"
#include "linalg/matrix/distMultiVector.hpp"
#include "linalg/internal/mpi/hemm.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace mpi
{
    template<typename T>
    void rayleighRitz(chase::distMatrix::BlockBlockMatrix<T>& H,
                    chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>& V1,
                    chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>& V2,
                    chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>& W1,
                    chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>& W2,
                    chase::Base<T>* ritzv,
                    std::size_t offset,
                    std::size_t subSize,
                    chase::distMatrix::RedundantMatrix<T>* A = nullptr)
    {
        if (ritzv == nullptr) {
            throw std::invalid_argument("ritzv cannot be a nullptr.");
        }

        std::unique_ptr<chase::distMatrix::RedundantMatrix<T>> A_ptr;

        if (A == nullptr) {
            // Allocate A if not provided
            A_ptr = std::make_unique<chase::distMatrix::RedundantMatrix<T>>(subSize, subSize, V1.getMpiGrid_shared_ptr());
            A = A_ptr.get();
        }

        // Perform the distributed matrix-matrix multiplication
        chase::linalg::internal::mpi::BlockBlockMultiplyMultiVectorsAndRedistributeAsync<T>(
                        H, 
                        V1, 
                        W1, 
                        V2, 
                        W2,
                        offset,
                        subSize);

        T One = T(1.0);
        T Zero = T(0.0);

        chase::linalg::blaspp::t_gemm(CblasColMajor, 
                                     CblasConjTrans, 
                                     CblasNoTrans, 
                                     subSize, 
                                     subSize, 
                                     W2.l_rows(),
                                     &One, 
                                     W2.l_data() + offset * W2.l_ld(),
                                     W2.l_ld(), 
                                     W1.l_data() + offset * W1.l_ld(), 
                                     W1.l_ld(), 
                                     &Zero, 
                                     A->l_data(),  
                                     subSize);

        // Perform the MPI_Allreduce to sum up the results
        MPI_Allreduce(MPI_IN_PLACE, 
                    A->l_data(), 
                    subSize * subSize, 
                    chase::mpi::getMPI_Type<T>(), 
                    MPI_SUM, 
                    A->getMpiGrid()->get_row_comm());

        chase::linalg::lapackpp::t_heevd(LAPACK_COL_MAJOR, 
                                        'V', 
                                        'L', 
                                        subSize, 
                                        A->l_data(), 
                                        subSize, 
                                        ritzv + offset);

        // GEMM for applying eigenvectors back to V1 from V2 * A
        chase::linalg::blaspp::t_gemm(CblasColMajor,
                                     CblasNoTrans,
                                     CblasNoTrans,
                                     V2.l_rows(),
                                     subSize,
                                     subSize,
                                     &One,
                                     V2.l_data() + offset * V2.l_ld(),
                                     V2.l_ld(),
                                     A->l_data(),
                                     subSize,
                                     &Zero,
                                     V1.l_data() + offset * V1.l_ld(),
                                     V1.l_ld());
    }

}
}
}
}