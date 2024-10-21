#pragma once

#include "mpi.h"
#include "grid/mpiTypes.hpp"
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
    template <typename MatrixType, typename InputMultiVectorType>
    void residuals(MatrixType& H,
                   InputMultiVectorType& V1,
                   InputMultiVectorType& V2,
                   typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W1,
                   typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W2,
                   chase::Base<typename MatrixType::value_type>* ritzv,
                   chase::Base<typename MatrixType::value_type>* resids,
                   std::size_t offset,
                   std::size_t subSize)           
    {
        using T = typename MatrixType::value_type;

        if (ritzv == nullptr) {
            throw std::invalid_argument("ritzv cannot be a nullptr.");
        }

        if (resids == nullptr) {
            throw std::invalid_argument("resids cannot be a nullptr.");
        }

        // Perform the distributed matrix-matrix multiplication
        chase::linalg::internal::mpi::MatrixMultiplyMultiVectorsAndRedistributeAsync(
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