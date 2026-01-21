// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "../typeTraits.hpp"
#include "external/blaspp/blaspp.hpp"
#include "external/lapackpp/lapackpp.hpp"
#include "grid/mpiTypes.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/mpi/mpi_kernels.hpp"
#include "mpi.h"

namespace chase
{
namespace linalg
{
namespace internal
{
/**
 * @brief Computes the residuals of the Ritz values using distributed matrix
 * operations.
 *
 * This function calculates the residuals associated with the Ritz values, a key
 * step in the Rayleigh-Ritz procedure for eigenvalue problems. It performs a
 * distributed matrix-matrix multiplication and adjusts the results by the Ritz
 * values. The final residuals are computed and gathered across all MPI
 * processes.
 *
 * @tparam MatrixType Type of the input matrix, must support distributed matrix
 * operations.
 * @tparam InputMultiVectorType Type of the input multivector, used for
 * eigenvector approximations.
 *
 * @param H Distributed matrix representing matrix in the eigenvalue problem.
 * @param V1 The input multivector representing the orthonormal basis.
 * @param V2 duplication of V1.
 * @param W1 Multivector sotring H*V1.
 * @param W2 Resulting multivector by redistribute V2 from column communicator
 * based to row communicator based.
 * @param ritzv Pointer to an array of Ritz values, used to adjust the residual
 * computation.
 * @param resids Pointer to an array to store the computed residuals.
 * @param offset Starting index within the vector of residuals to be computed.
 * @param subSize Number of residuals to compute starting from the offset.
 *
 * @throws std::invalid_argument if either `ritzv` or `resids` is a nullptr.
 *
 * @details
 * The function first performs a distributed matrix-matrix multiplication and
 * then iterates over each residual vector to apply the Ritz value shifts. The
 * final residuals are normalized and gathered using MPI all-reduce operations.
 * Each residual norm is calculated as the Euclidean norm of the adjusted
 * vector.
 */
template <typename MatrixType, typename InputMultiVectorType>
void cpu_mpi::residuals(
    MatrixType& H, InputMultiVectorType& V1, InputMultiVectorType& V2,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W1,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W2,
    chase::Base<typename MatrixType::value_type>* ritzv,
    chase::Base<typename MatrixType::value_type>* resids, std::size_t offset,
    std::size_t subSize)
{
    using T = typename MatrixType::value_type;

    if (ritzv == nullptr)
    {
        throw std::invalid_argument("ritzv cannot be a nullptr.");
    }

    if (resids == nullptr)
    {
        throw std::invalid_argument("resids cannot be a nullptr.");
    }

    // Perform the distributed matrix-matrix multiplication
    chase::linalg::internal::cpu_mpi::
        MatrixMultiplyMultiVectorsAndRedistributeAsync(H, V1, W1, V2, W2,
                                                       offset, subSize);

    std::size_t i_one = 1;
    for (auto i = 0; i < subSize; i++)
    {
        T alpha = -ritzv[i + offset];
        chase::linalg::blaspp::t_axpy(
            W2.l_rows(), &alpha,
            W2.l_data() + offset * W2.l_ld() + i * W2.l_ld(), i_one,
            W1.l_data() + offset * W1.l_ld() + i * W1.l_ld(), i_one);

        resids[i + offset] = chase::linalg::blaspp::t_norm_p2(
            W2.l_rows(), W1.l_data() + offset * W1.l_ld() + i * W1.l_ld());
    }

    MPI_Allreduce(MPI_IN_PLACE, resids + offset, subSize,
                  chase::mpi::getMPI_Type<chase::Base<T>>(), MPI_SUM,
                  V1.getMpiGrid()->get_row_comm());

    for (auto i = 0; i < subSize; ++i)
    {
        resids[i + offset] = std::sqrt(resids[i + offset]);
    }
}
} // namespace internal
} // namespace linalg
} // namespace chase