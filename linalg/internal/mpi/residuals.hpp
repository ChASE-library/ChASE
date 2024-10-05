#pragma once

#include "mpi.h"
#include "Impl/grid/mpiTypes.hpp"
#include "linalg/blaspp/blaspp.hpp"
#include "linalg/lapackpp/lapackpp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
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
    void residuals(chase::distMatrix::BlockBlockMatrix<T, chase::platform::CPU>& H,
                   chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::CPU>& V1,
                   chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::CPU>& V2,
                   chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::CPU>& W1,
                   chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::CPU>& W2,
                   chase::Base<T>* ritzv,
                   chase::Base<T>* resids,
                   std::size_t offset,
                   std::size_t subSize)
    {
        if (ritzv == nullptr) {
            throw std::invalid_argument("ritzv cannot be a nullptr.");
        }

        if (resids == nullptr) {
            throw std::invalid_argument("resids cannot be a nullptr.");
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

        std::size_t i_one = 1;
        for (auto i = 0; i < subSize; i++)
        {
            T alpha = -ritzv[i + offset];
            chase::linalg::blaspp::t_axpy(W2.l_rows(), 
                                          &alpha, 
                                          W2.l_data() + offset * W2.l_ld() + i * W2.l_ld(),
                                          i_one,
                                          W1.l_data() + offset * W1.l_ld() + i * W1.l_ld(),
                                          i_one);

            resids[i + offset] = chase::linalg::blaspp::t_norm_p2(W2.l_rows(), 
                                                                 W1.l_data() + offset * W1.l_ld() + i * W1.l_ld());
        }

        MPI_Allreduce(MPI_IN_PLACE,
                      resids + offset,
                      subSize,
                      chase::mpi::getMPI_Type<chase::Base<T>>(),
                      MPI_SUM,
                      V1.getMpiGrid()->get_row_comm());

        for (auto i = 0; i < subSize; ++i)
        {
            resids[i + offset] = std::sqrt(resids[i + offset]);
        }
    }

}
}
}
}