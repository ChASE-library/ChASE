// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "residuals.cuh"
#include "external/cublaspp/cublaspp.hpp"
#include "linalg/matrix/matrix.hpp"
#include "Impl/chase_gpu/cuda_utils.hpp"
#include "Impl/chase_gpu/nvtx.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
    /**
    * @brief Compute the residuals of eigenvectors for a given matrix and eigenvalues.
    *
    * This function computes the residuals of the eigenvectors, which measure how well
    * the eigenvectors satisfy the eigenvalue equation \( H \mathbf{v}_i = \lambda_i \mathbf{v}_i \).
    * The residual for each eigenvector \( \mathbf{v}_i \) is defined as \( ||H \mathbf{v}_i - \lambda_i \mathbf{v}_i|| \),
    * where \( \lambda_i \) is the corresponding eigenvalue. The computed residuals are stored in the `resids` array.
    *
    * @tparam T Data type for the matrix and vectors (e.g., float, double, etc.).
    * @param[in] N The number of rows and columns of the matrix H.
    * @param[in] H The input matrix (N x N).
    * @param[in] ldh The leading dimension of the matrix H.
    * @param[in] eigen_nb The number of eigenvalues and eigenvectors.
    * @param[in] evals The array of eigenvalues.
    * @param[in] evecs The matrix of eigenvectors (N x eigen_nb), where each column is an eigenvector.
    * @param[in] ldv The leading dimension of the matrix evecs.
    * @param[out] resids The array that will store the computed residuals for each eigenvector.
    * @param[in] V A temporary matrix used in intermediate calculations. If not provided, it is allocated internally.
    *
    * The function performs the following steps:
    * 1. Computes the matrix-vector multiplication \( V = H \cdot E \), where E are the eigenvectors.
    * 2. Subtracts the eigenvalue \( \lambda_i \) times the eigenvector from the result.
    * 3. Computes the 2-norm of the residual for each eigenvector and stores it in the `resids` array.
    */    
    template<typename T>
    void residuals(cublasHandle_t cublas_handle, 
                   chase::matrix::Matrix<T, chase::platform::GPU>& H,
                   chase::matrix::Matrix<T, chase::platform::GPU>& V1,
                   chase::Base<T> *d_ritzv, 
                   chase::Base<T> *d_resids, 
                   std::size_t offset,
                   std::size_t subSize,                   
                   chase::matrix::Matrix<T, chase::platform::GPU>* V2 = nullptr)
    {
        SCOPED_NVTX_RANGE();

        if (V2 == nullptr)
        {
            V2 = new chase::matrix::Matrix<T, chase::platform::GPU>(V1.rows(), V1.cols());
        }

        T alpha = T(1.0);
        T beta = T(0.0);

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                                                                cublas_handle, 
                                                                CUBLAS_OP_C, 
                                                                CUBLAS_OP_N, 
                                                                H.rows(), 
                                                                subSize, 
                                                                H.cols(), 
                                                                &alpha,
                                                                H.data(), 
                                                                H.ld(), 
                                                                V1.data() + offset * V1.ld(), 
                                                                V1.ld(), 
                                                                &beta, 
                                                                V2->data() + offset * V2->ld(),
                                                                V2->ld()));

        cudaStream_t usedStream;
        CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &usedStream));

        residual_gpu(V2->rows(), 
                     subSize, 
                     V2->data() +  offset * V2->ld(), 
                     V2->ld(),
                     V1.data() + offset * V1.ld(),
                     V1.ld(), 
                     d_ritzv + offset,
                     d_resids + offset, 
                     true, usedStream);        
    }

}
}
}
}