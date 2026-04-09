// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "Impl/chase_gpu/cuda_utils.hpp"
#include "Impl/chase_gpu/nvtx.hpp"
#include "external/blaspp/blaspp.hpp"
#include "external/cublaspp/cublaspp.hpp"
#include "external/cusolverpp/cusolverpp.hpp"
#include "external/lapackpp/lapackpp.hpp"
#include "grid/mpiTypes.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/nccl/hemm.hpp"
#include "linalg/internal/nccl/nccl_kernels.hpp"
#include "mpi.h"

namespace chase
{
namespace linalg
{
namespace internal
{

template <typename MatrixType, typename InputMultiVectorType>
void cuda_nccl::rayleighRitz(
    cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
    MatrixType& H, InputMultiVectorType& V1, InputMultiVectorType& V2,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W1,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W2,
    chase::distMatrix::RedundantMatrix<
        chase::Base<typename MatrixType::value_type>, chase::platform::GPU>&
        ritzv,
    std::size_t offset, std::size_t subSize, int* devInfo,
    typename MatrixType::value_type* workspace, int lwork_heevd,
    chase::distMatrix::RedundantMatrix<typename MatrixType::value_type,
                                       chase::platform::GPU>* A)
{
    using T = typename MatrixType::value_type;
    using RitzReal = chase::Base<T>;
    SCOPED_NVTX_RANGE();

    std::unique_ptr<chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>
        A_ptr;
    std::size_t upperTriangularSize = std::size_t(subSize * (subSize + 1) / 2);

    cudaStream_t stream_orig_cublas = nullptr;
    cudaStream_t stream_orig_cusolver = nullptr;
    cudaStream_t compute_stream = nullptr;
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t evt_begin = nullptr;
    cudaEvent_t evt_ritz_ready = nullptr;
    cudaEvent_t evt_end_compute = nullptr;
    bool owns_workspace = false;
    int info = 0;

    if (A == nullptr)
    {
        A_ptr = std::make_unique<
            chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>(
            subSize, subSize, V1.getMpiGrid_shared_ptr());
        A = A_ptr.get();
    }

    CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &stream_orig_cublas));
    CHECK_CUSOLVER_ERROR(
        cusolverDnGetStream(cusolver_handle, &stream_orig_cusolver));
    CHECK_CUDA_ERROR(
        cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    CHECK_CUDA_ERROR(
        cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
    CHECK_CUDA_ERROR(cudaEventCreate(&evt_begin));
    CHECK_CUDA_ERROR(cudaEventCreate(&evt_ritz_ready));
    CHECK_CUDA_ERROR(cudaEventCreate(&evt_end_compute));
    CHECK_CUDA_ERROR(cudaEventRecord(evt_begin, stream_orig_cublas));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(compute_stream, evt_begin, 0));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, compute_stream));
    CHECK_CUSOLVER_ERROR(cusolverDnSetStream(cusolver_handle, compute_stream));

    cublasPointerMode_t cublas_pointer_mode_prev = CUBLAS_POINTER_MODE_HOST;
    CHECK_CUBLAS_ERROR(cublasGetPointerMode(cublas_handle, &cublas_pointer_mode_prev));
    CHECK_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));

    std::unique_ptr<T, chase::cuda::utils::CudaDeleter> work_ptr = nullptr;
    if (workspace == nullptr)
    {
        lwork_heevd = 0;
        CHECK_CUSOLVER_ERROR(
            chase::linalg::cusolverpp::cusolverDnTheevd_bufferSize(
                cusolver_handle, CUSOLVER_EIG_MODE_VECTOR,
                CUBLAS_FILL_MODE_UPPER, subSize, A->l_data(), subSize,
                ritzv.l_data() + offset, &lwork_heevd));

        if (upperTriangularSize > lwork_heevd)
        {
            lwork_heevd = upperTriangularSize;
        }

        CHECK_CUDA_ERROR(cudaMallocAsync((void**)&workspace,
                                         sizeof(T) * static_cast<std::size_t>(lwork_heevd),
                                         compute_stream));
        work_ptr.reset(workspace);
        workspace = work_ptr.get();
        owns_workspace = true;
    }

    chase::linalg::internal::cuda_nccl::
        MatrixMultiplyMultiVectorsAndRedistributeAsync(cublas_handle, H, V1, W1,
                                                       V2, W2, offset, subSize);

    T One = T(1.0);
    T Zero = T(0.0);

    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
        cublas_handle, CUBLAS_OP_C, CUBLAS_OP_N, subSize, subSize, W2.l_rows(),
        &One, W2.l_data() + offset * W2.l_ld(), W2.l_ld(),
        W1.l_data() + offset * W1.l_ld(), W1.l_ld(), &Zero, A->l_data(),
        subSize));

    chase::linalg::internal::cuda::extractUpperTriangular(A->l_data(), subSize,
                                                          workspace, subSize,
                                                          &compute_stream);
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(
        workspace, workspace, upperTriangularSize, ncclSum,
        A->getMpiGrid()->get_nccl_row_comm(), &compute_stream));
    chase::linalg::internal::cuda::unpackUpperTriangular(workspace, subSize,
                                                         A->l_data(), subSize,
                                                         &compute_stream);

#ifdef RR_DOUBLE_PRECISION
    if constexpr (std::is_same<T, std::complex<float>>::value ||
                  std::is_same<T, float>::value)
    {
        if (A->isDoublePrecisionEnabled())
        {
            A->copyToSubBlockAsync(0, subSize * subSize, &compute_stream);
        }
        else
        {
            A->enableDoublePrecisionAsync(&compute_stream);
        }

        if (!ritzv.isDoublePrecisionEnabled())
        {
            ritzv.enableDoublePrecisionAsync(&compute_stream);
        }

        auto A_d = A->getDoublePrecisionMatrix();
        auto ritzv_d = ritzv.getDoublePrecisionMatrix();
        using Td = typename chase::ToDoublePrecisionTrait<T>::Type;
        Td* workspace_d = nullptr;
        CHECK_CUDA_ERROR(cudaMallocAsync(
            (void**)&workspace_d,
            sizeof(Td) * static_cast<std::size_t>(lwork_heevd), compute_stream));
        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTheevd(
            cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
            subSize, A_d->l_data(), subSize,
            ritzv_d->l_data() + offset, workspace_d, lwork_heevd, devInfo));
        ritzv.disableDoublePrecisionAsync(true, &compute_stream);
        A->copyBackSubBlockAsync(0, subSize * subSize, &compute_stream);
        CHECK_CUDA_ERROR(cudaFreeAsync(workspace_d, compute_stream));
    }
    else
    {
#endif
        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTheevd(
            cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
            subSize, A->l_data(), subSize, ritzv.l_data() + offset, workspace,
            lwork_heevd, devInfo));
#ifdef RR_DOUBLE_PRECISION
    }
#endif

    CHECK_CUDA_ERROR(cudaMemcpyAsync(&info, devInfo, sizeof(int),
                                     cudaMemcpyDeviceToHost, compute_stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(compute_stream));

    if (info != 0)
    {
        CHECK_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, cublas_pointer_mode_prev));
        CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream_orig_cublas));
        CHECK_CUSOLVER_ERROR(
            cusolverDnSetStream(cusolver_handle, stream_orig_cusolver));
        CHECK_CUDA_ERROR(cudaEventDestroy(evt_begin));
        CHECK_CUDA_ERROR(cudaEventDestroy(evt_ritz_ready));
        CHECK_CUDA_ERROR(cudaEventDestroy(evt_end_compute));
        CHECK_CUDA_ERROR(cudaStreamDestroy(copy_stream));
        if (owns_workspace)
            CHECK_CUDA_ERROR(cudaFreeAsync(workspace, compute_stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(compute_stream));
        CHECK_CUDA_ERROR(cudaStreamDestroy(compute_stream));
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        throw std::runtime_error("cusolver HEEVD failed in RayleighRitz");
    }

    CHECK_CUDA_ERROR(cudaEventRecord(evt_ritz_ready, compute_stream));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(copy_stream, evt_ritz_ready, 0));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        ritzv.cpu_data() + offset, ritzv.l_data() + offset,
        subSize * sizeof(RitzReal), cudaMemcpyDeviceToHost, copy_stream));

    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, V2.l_rows(), subSize, subSize,
        &One, V2.l_data() + offset * V2.l_ld(), V2.l_ld(), A->l_data(), subSize,
        &Zero, V1.l_data() + offset * V1.l_ld(), V1.l_ld()));

    CHECK_CUDA_ERROR(cudaEventRecord(evt_end_compute, compute_stream));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_orig_cublas, evt_end_compute, 0));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(copy_stream));

    CHECK_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, cublas_pointer_mode_prev));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream_orig_cublas));
    CHECK_CUSOLVER_ERROR(cusolverDnSetStream(cusolver_handle, stream_orig_cusolver));
    CHECK_CUDA_ERROR(cudaEventDestroy(evt_begin));
    CHECK_CUDA_ERROR(cudaEventDestroy(evt_ritz_ready));
    CHECK_CUDA_ERROR(cudaEventDestroy(evt_end_compute));
    CHECK_CUDA_ERROR(cudaStreamDestroy(copy_stream));
    if (owns_workspace)
        CHECK_CUDA_ERROR(cudaFreeAsync(workspace, compute_stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(compute_stream));
    CHECK_CUDA_ERROR(cudaStreamDestroy(compute_stream));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace internal
} // namespace linalg
} // namespace chase
