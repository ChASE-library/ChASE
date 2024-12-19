// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "mpi.h"
#include "grid/mpiTypes.hpp"
#include "external/blaspp/blaspp.hpp"
#include "external/lapackpp/lapackpp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/mpi/mpi_kernels.hpp"
#include "external/scalapackpp/scalapackpp.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{        
    /**
     * @brief Performs the Rayleigh-Ritz procedure, reducing the matrix H using the basis vectors
     * in V1 and V2, and storing the results in W1, W2, and the eigenvalues in ritzv.
     * 
     * This function computes the distributed matrix multiplication and reduction required
     * for the Rayleigh-Ritz procedure. If matrix A is provided, it uses that as a workspace.
     * Otherwise, a temporary matrix is created.
     *
     * @tparam MatrixType Type of the input matrix H.
     * @tparam InputMultiVectorType Type of the input multivector V1 and V2.
     * @param H The input matrix representing the system to be reduced.
     * @param V1 The input multivector representing the orthonormal basis.
     * @param V2 duplication of V1.
     * @param W1 Multivector sotring H*V1.
     * @param W2 Resulting multivector by redistribute V2 from column communicator based to row communicator based.
     * @param ritzv Pointer to an array where computed eigenvalues of the reduced matrix will be stored.
     * @param offset The starting offset in V1, V2, W1, and W2 for submatrix processing.
     * @param subSize Size of the submatrix used in reduction.
     * @param A Optional workspace matrix for storing intermediate results; if nullptr, a temporary matrix is created.
     * 
     * @throws std::invalid_argument if ritzv is a nullptr.
     */                  
    template <typename MatrixType, typename InputMultiVectorType>
     void cpu_mpi::rayleighRitz(MatrixType& H,
                       InputMultiVectorType& V1,
                       InputMultiVectorType& V2,
                       typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W1,
                       typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W2,
                       chase::Base<typename MatrixType::value_type>* ritzv,
                       std::size_t offset,
                       std::size_t subSize,
                       chase::distMatrix::RedundantMatrix<typename MatrixType::value_type, chase::platform::CPU>* A)                
    {
        using T = typename MatrixType::value_type;
        
        if (ritzv == nullptr) {
            throw std::invalid_argument("ritzv cannot be a nullptr.");
        }

        std::unique_ptr<chase::distMatrix::RedundantMatrix<T, chase::platform::CPU>> A_ptr;

        if (A == nullptr) {
            // Allocate A if not provided
            A_ptr = std::make_unique<chase::distMatrix::RedundantMatrix<T, chase::platform::CPU>>(subSize, subSize, V1.getMpiGrid_shared_ptr());
            A = A_ptr.get();
        }

        // Perform the distributed matrix-matrix multiplication
        chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectorsAndRedistributeAsync(
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