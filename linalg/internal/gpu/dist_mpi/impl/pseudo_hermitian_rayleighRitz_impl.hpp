// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "external/cpu/lapackpp/lapackpp.hpp"
#include "grid/mpiTypes.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/gpu/backend/handle_utils.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"
#include "linalg/internal/gpu/backend/blas.hpp"
#include "linalg/internal/gpu/backend/solver.hpp"
#include "linalg/internal/gpu/dist_mpi/impl/rayleighRitz_impl.hpp"
#include "linalg/internal/gpu/dist_mpi/impl/mpi_collectives.hpp"
#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"
#include "linalg/internal/typeTraits.hpp"
#include <algorithm>
#include <complex>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace gpu_mpi
{
template <typename Backend, typename BlasHandle, typename SolverHandle,
          typename SolverParams, typename MpiKernelNamespace, typename MatrixType,
          typename InputMultiVectorType>
void pseudo_hermitian_rayleighRitz_impl(
    BlasHandle blas_handle, SolverHandle solver_handle, SolverParams params,
    MatrixType& H, InputMultiVectorType& V1, InputMultiVectorType& V2,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W1,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W2,
    chase::distMatrix::RedundantMatrix<
        chase::Base<typename MatrixType::value_type>, chase::platform::GPU>&
        ritzv,
    std::size_t offset, std::size_t subSize, int* devInfo,
    typename MatrixType::value_type* d_workspace, int d_lwork,
    typename MatrixType::value_type* h_workspace, int h_lwork,
    chase::distMatrix::RedundantMatrix<typename MatrixType::value_type,
                                       chase::platform::GPU>* A)
{
    using T = typename MatrixType::value_type;
    using blas_dispatch = gpu_backend::blas<Backend>;
    using util = gpu_backend::handle_utils<Backend>;
    using solver = gpu_backend::solver<Backend>;
    using kernels = gpu_device::kernels<Backend>;

    auto usedStream = util::get_blas_stream(blas_handle);

    std::unique_ptr<chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>
        A_ptr;

    if (A == nullptr)
    {
        // Allocate A if not provided
        A_ptr = std::make_unique<
            chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>(
            3 * subSize, subSize, V1.getMpiGrid_shared_ptr());
        A = A_ptr.get();
    }

    T One = T(1.0);
    T Zero = T(0.0);
    T NegOne = -T(1.0);
    chase::Base<T> real_Zero = chase::Base<T>(0.0);

    T* M = A->l_data() + subSize * subSize;
    T* W = A->l_data() + 2 * subSize * subSize;
    T* ritzv_complex = A->l_data() + subSize * subSize;

    std::vector<T> ritzvs_cmplex_cpu(subSize);

#ifdef XGEEV_EXISTS
    // Allocating workspace memory for Xgeev
    if (d_workspace == nullptr ||
        h_workspace == nullptr) // To update once Xgeev is plugged in
    {
        std::size_t temp_d_lwork = 0;
        std::size_t temp_h_lwork = 0;

        util::check_solver(solver::geev_buffer_size(
            solver_handle, params, solver::eig_mode_no_vector,
            solver::eig_mode_vector, subSize, A->l_data(), subSize,
            ritzv_complex, nullptr, 1, V1.l_data() + offset * V1.l_ld(),
            static_cast<int>(V1.l_ld()), &temp_d_lwork, &temp_h_lwork));
        if (d_workspace == nullptr)
        {

            d_lwork = temp_d_lwork;

            gpu_backend::device_malloc<Backend>(&d_workspace, static_cast<std::size_t>(d_lwork));
        }

        if (h_workspace == nullptr)
        {

            h_lwork = temp_h_lwork;

            h_workspace = new T[h_lwork]();
        }
    }
#else
    A->allocate_cpu_data();
#endif

    // Allocate the space for scaling weights. Can be Base<T> since reals?
    chase::distMatrix::RedundantMatrix<
        chase::Base<typename MatrixType::value_type>, chase::platform::GPU>
        diag(1, subSize, V1.getMpiGrid_shared_ptr());

    // Allocate the space for the imaginary parts of ritz values
    std::vector<chase::Base<T>> ritzvi(subSize, chase::Base<T>(0.0));

    // Perform the distributed matrix-matrix multiplication
    MpiKernelNamespace::
        MatrixMultiplyMultiVectorsAndRedistribute(blas_handle, H, V1, W1, V2,
                                                  W2, offset, subSize);

    util::check_blas(blas_dispatch::gemm(
        blas_handle, util::op_c(), util::op_n(), subSize, subSize, W2.l_rows(),
        &One, W2.l_data() + offset * W2.l_ld(), W2.l_ld(),
        W1.l_data() + offset * W1.l_ld(), W1.l_ld(), &Zero, W, subSize));

    gpu_mpi::mpiAllreduceSum(W, subSize * subSize,
                                   A->getMpiGrid()->get_row_comm());

    MpiKernelNamespace::flipLowerHalfMatrixSign(V2, offset,
                                                               subSize);
    gpu_backend::device_synchronize<Backend>();

    util::check_blas(blas_dispatch::gemm(
        blas_handle, util::op_c(), util::op_n(), subSize, subSize, V2.l_rows(),
        &One, V2.l_data() + offset * V2.l_ld(), V2.l_ld(),
        V1.l_data() + offset * V1.l_ld(), V1.l_ld(), &Zero, M, subSize));

    gpu_mpi::mpiAllreduceSum(M, subSize * subSize,
                                   A->getMpiGrid()->get_col_comm());

    kernels::plus_inverse_diagonal(
        M, subSize, subSize, real_Zero, diag.l_data(), usedStream);

    kernels::set_diagonal(M, subSize, subSize, Zero,
                                                      usedStream);

    util::check_blas(blas_dispatch::gemm(
        blas_handle, util::op_n(), util::op_n(), subSize, subSize, subSize,
        &NegOne, M, subSize, W, subSize, &Zero, A->l_data(),
        subSize)); // A = (Diag(M) - M) * A

    MpiKernelNamespace::flipLowerHalfMatrixSign(W1, offset,
                                                               subSize);
    gpu_backend::device_synchronize<Backend>();

    // Last GEMM for the construction of the rayleigh Quotient : (H' * Ql)' * Qr
    util::check_blas(blas_dispatch::gemm(
        blas_handle, util::op_c(), util::op_n(), subSize, subSize, W2.l_rows(),
        &One, W2.l_data() + offset * W2.l_ld(), W2.l_ld(),
        W1.l_data() + offset * W1.l_ld(), W1.l_ld(), &Zero, M, subSize));

    gpu_mpi::mpiAllreduceSum(M, subSize * subSize,
                                   A->getMpiGrid()->get_row_comm());

    util::check_blas(blas_dispatch::axpy(
        blas_handle, subSize * subSize, &One, M, 1, A->l_data(), 1));

    kernels::scale_rows_matrix(
        A->l_data(), subSize, subSize, subSize, diag.l_data(), usedStream);
    gpu_backend::stream_synchronize<Backend>(usedStream);

    chase::Base<T>* ptx = ritzv.cpu_data() + offset;

#ifdef XGEEV_EXISTS

#ifdef CHASE_OUTPUT
    if (H.grank() == 0)
    {
        std::cout
            << "Compute the eigenpairs of the non-hermitian rayleigh quotient"
            << std::endl;
    }
#endif
    if constexpr (0 && std::is_same<T, std::complex<float>>::value)
    {
        if (A->isDoublePrecisionEnabled())
        {
            A->copyToSubBlock(0, subSize * subSize);
        }
        else
        {
            A->enableDoublePrecision();
        }
        auto A_d = A->getDoublePrecisionMatrix();
        std::complex<double>* W_d = A_d->l_data() + 2 * subSize * subSize;
        std::complex<double>* ritzv_complex_d =
            A_d->l_data() + 1 * subSize * subSize;

        util::check_solver(solver::geev(
            solver_handle, params, solver::eig_mode_no_vector,
            solver::eig_mode_vector, subSize, A_d->l_data(), subSize,
            ritzv_complex_d, nullptr, 1, W_d, static_cast<int>(subSize),
            reinterpret_cast<std::complex<double>*>(d_workspace), d_lwork,
            reinterpret_cast<std::complex<double>*>(h_workspace), h_lwork,
            reinterpret_cast<typename util::info_type*>(devInfo)));

        int info;
        gpu_backend::copy_to_host<Backend>(&info, devInfo, 1);

        if (info != 0)
        {
            throw std::runtime_error("cusolver HEEVD failed in RayleighRitz");
        }

        A->copyBackSubBlock(subSize * subSize, 2 * subSize * subSize);
        // A->copyBack();
        if constexpr (std::is_same<T, std::complex<chase::Base<T>>>::value)
        {
            gpu_backend::copy_to_host<Backend>(ritzvs_cmplex_cpu.data(), ritzv_complex, subSize);
            for (auto i = 0; i < subSize; i++)
            {
                ptx[i] = std::real(ritzvs_cmplex_cpu[i]);
            }
        }
        else
        {
            gpu_backend::copy_to_host<Backend>(ptx, ritzv_complex, subSize);
        }
    }
    else
    {
        util::check_solver(solver::geev(
            solver_handle, params, solver::eig_mode_no_vector,
            solver::eig_mode_vector, subSize, A->l_data(), subSize,
            ritzv_complex, nullptr, 1, W, static_cast<int>(subSize),
            d_workspace, d_lwork, h_workspace, h_lwork,
            reinterpret_cast<typename util::info_type*>(devInfo)));

        int info;
        gpu_backend::copy_to_host<Backend>(&info, devInfo, 1);

        if (info != 0)
        {
            throw std::runtime_error("cusolver HEEVD failed in RayleighRitz");
        }

#ifdef CHASE_OUTPUT
        if (H.grank() == 0)
        {
            std::cout << "Eigenvalues of the non-Hermitian rayleigh-quotient "
                         "computed with GEEV."
                      << std::endl;
        }
#endif

        // std::cout << "Copying the complex ritz values back to cpu" <<
        // std::endl; thrust::device_vector<int> indices(n); //Does not compile,
        // returns unimplemented on this system...
        if constexpr (std::is_same<T, std::complex<chase::Base<T>>>::value)
        {
            gpu_backend::copy_to_host<Backend>(ritzvs_cmplex_cpu.data(), ritzv_complex, subSize);
            for (auto i = 0; i < subSize; i++)
            {
                ptx[i] = std::real(ritzvs_cmplex_cpu[i]);
            }
        }
        else
        {
            gpu_backend::copy_to_host<Backend>(ptx, ritzv_complex, subSize);
        }
    }
#else
#ifdef CHASE_OUTPUT
    if (H.grank() == 0)
    {
        std::cout << "WARNING! XGeev not found in cuda. Compute Geev on CPU "
                     "with Lapack..."
                  << std::endl;
    }
#endif

    if constexpr (0 && std::is_same<T, std::complex<float>>::value)
    {
        // Initialize vectors with proper size and type
        std::vector<chase::Base<T>> ptx_imag(subSize, chase::Base<T>(0.0));
        std::vector<double> ptx_imag_double(subSize, 0.0);
        std::vector<double> ptx_double(subSize, 0.0);
        std::vector<std::complex<double>> W_cpu_double(subSize * subSize, 0.0);
        std::vector<std::complex<double>> A_double(subSize * subSize, 0.0);

        // Transfer data from device to host
        A->D2H();
        T* W_cpu = A->cpu_data() + 2 * subSize * subSize;

        // Convert single precision to double precision
        std::transform(A->cpu_data(), A->cpu_data() + subSize * subSize,
                       A_double.begin(), [](const T& val)
                       { return static_cast<std::complex<double>>(val); });

        // Compute eigenvalues and eigenvectors in double precision
        lapackpp::t_geev(LAPACK_COL_MAJOR, 'V', subSize, A_double.data(),
                         subSize, ptx_double.data(), ptx_imag_double.data(),
                         W_cpu_double.data(), subSize);

        // Convert results back to single precision
        std::transform(ptx_double.begin(), ptx_double.end(), ptx,
                       [](double val) { return static_cast<float>(val); });

        // Convert W and A back to single precision
        std::transform(W_cpu_double.begin(), W_cpu_double.end(), W_cpu,
                       [](const std::complex<double>& val)
                       { return static_cast<std::complex<float>>(val); });

        std::transform(A_double.begin(), A_double.end(), A->cpu_data(),
                       [](const std::complex<double>& val)
                       { return static_cast<std::complex<float>>(val); });

        // Transfer data back to device
        A->H2D();
    }

    else
    {
        std::vector<chase::Base<T>> ptx_imag =
            std::vector<chase::Base<T>>(subSize, chase::Base<T>(0.0));

        A->D2H();

        T* W_cpu = A->cpu_data() + 2 * subSize * subSize;

        // Compute the eigenpairs of the non-hermitian rayleigh quotient on the
        // CPU
        lapackpp::t_geev(LAPACK_COL_MAJOR, 'V', subSize, A->cpu_data(), subSize,
                         ptx, ptx_imag.data(), W_cpu, subSize);

        A->H2D();
    }
#endif

    std::vector<std::size_t> indices(subSize);
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., n-1
    std::sort(indices.begin(), indices.end(),
              [&ptx](std::size_t i1, std::size_t i2)
              { return ptx[i1] < ptx[i2]; });

    std::vector<chase::Base<T>> sorted_ritzv(subSize);
    T* d_sorted_W = A->l_data();
    // Reorder eigenvalues and eigenvectors

    for (std::size_t i = 0; i < subSize; ++i)
    {
        sorted_ritzv[i] = ptx[indices[i]];
        gpu_backend::memcpy_d2d<Backend>(d_sorted_W + i * subSize, W + indices[i] * subSize, subSize);
    }

    std::copy(sorted_ritzv.begin(), sorted_ritzv.end(), ptx);

    ritzv.H2D();

    MpiKernelNamespace::flipLowerHalfMatrixSign(V2, offset,
                                                               subSize);
    gpu_backend::device_synchronize<Backend>();

    util::check_blas(blas_dispatch::gemm(
        blas_handle, util::op_n(), util::op_n(), V2.l_rows(), subSize, subSize,
        &One, V2.l_data() + offset * V2.l_ld(), V2.l_ld(), d_sorted_W, subSize,
        &Zero, V1.l_data() + offset * V1.l_ld(), V1.l_ld()));
}

template <typename Backend, typename BlasHandle, typename SolverHandle,
          typename SolverParams, typename MpiKernelNamespace, typename MatrixType,
          typename InputMultiVectorType>
void pseudo_hermitian_rayleighRitz_v2_impl(
    BlasHandle blas_handle, SolverHandle solver_handle, SolverParams params,
    MatrixType& H, InputMultiVectorType& V1, InputMultiVectorType& V2,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W1,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W2,
    chase::distMatrix::RedundantMatrix<
        chase::Base<typename MatrixType::value_type>, chase::platform::GPU>&
        ritzv,
    std::size_t offset, std::size_t subSize, int* devInfo,
    typename MatrixType::value_type* workspace, int lwork,
    chase::distMatrix::RedundantMatrix<typename MatrixType::value_type,
                                       chase::platform::GPU>* A)
{
    using T = typename MatrixType::value_type;
    using blas_dispatch = gpu_backend::blas<Backend>;
    using util = gpu_backend::handle_utils<Backend>;
    using solver = gpu_backend::solver<Backend>;
    using kernels = gpu_device::kernels<Backend>;

    (void)params;

    auto usedStream = util::get_blas_stream(blas_handle);

    T One = T(1.0);
    T Zero = T(0.0);

    std::unique_ptr<chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>
        A_ptr;

    if (A == nullptr)
    {
        // Allocate A if not provided
        A_ptr = std::make_unique<
            chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>(
            2 * subSize, subSize, V1.getMpiGrid_shared_ptr());
        A = A_ptr.get();
    }

    T* M = A->l_data() + subSize * subSize;

    std::unique_ptr<T, typename util::device_deleter> work_ptr = nullptr;
    std::size_t upperTriangularSize = std::size_t(subSize * (subSize + 1) / 2);

    if (workspace == nullptr)
    {
        lwork = 0;
        detail::query_heevd_lower_buffer_size<Backend, T>(
            solver_handle, subSize, A->l_data(), subSize,
            ritzv.l_data() + offset, &lwork);

        int lwork_potrf = 0;
        detail::query_potrf_lower_buffer_size<Backend, T>(
            solver_handle, subSize, A->l_data(), subSize, &lwork_potrf);

        if (lwork < lwork_potrf)
            lwork = lwork_potrf;

        if (upperTriangularSize > lwork)
        {
            lwork = upperTriangularSize;
        }

        gpu_backend::device_malloc<Backend>(&workspace, static_cast<std::size_t>(lwork));

        work_ptr.reset(workspace);
        workspace = work_ptr.get();
    }

    // Perform the distributed matrix-matrix multiplication
    MpiKernelNamespace::
        MatrixMultiplyMultiVectorsAndRedistribute(blas_handle, H, V1, W1, V2,
                                                  W2, offset, subSize);

    MpiKernelNamespace::flipLowerHalfMatrixSign(W1, offset,
                                                               subSize);
    gpu_backend::device_synchronize<Backend>();

    util::check_blas(blas_dispatch::gemm(
        blas_handle, util::op_c(), util::op_n(), subSize, subSize, W2.l_rows(),
        &One, W2.l_data() + offset * W2.l_ld(), W2.l_ld(),
        W1.l_data() + offset * W1.l_ld(), W1.l_ld(), &Zero, A->l_data(),
        subSize));

    gpu_mpi::mpiAllreduceSum(A->l_data(), subSize * subSize,
                                   A->getMpiGrid()->get_row_comm());

    detail::run_potrf_lower<Backend, T>(
        solver_handle, subSize, A->l_data(), subSize,
        reinterpret_cast<typename util::info_type*>(devInfo), workspace, lwork);

    MpiKernelNamespace::flipLowerHalfMatrixSign(V1, offset,
                                                               subSize);
    gpu_backend::device_synchronize<Backend>();

    util::check_blas(blas_dispatch::gemm(
        blas_handle, util::op_c(), util::op_n(), subSize, subSize, V2.l_rows(),
        &One, V2.l_data() + offset * V2.l_ld(), V2.l_ld(),
        V1.l_data() + offset * V1.l_ld(), V1.l_ld(), &Zero, M, subSize));

    gpu_mpi::mpiAllreduceSum(M, subSize * subSize,
                                   A->getMpiGrid()->get_col_comm());

    util::check_blas(blas_dispatch::trsm(
        blas_handle, util::side_left(), util::fill_lower(), util::op_n(),
        util::diag_non_unit(), subSize, subSize, &One, A->l_data(), subSize, M,
        subSize));

    util::check_blas(blas_dispatch::trsm(
        blas_handle, util::side_right(), util::fill_lower(), util::op_c(),
        util::diag_non_unit(), subSize, subSize, &One, A->l_data(), subSize, M,
        subSize));

    T NegOne = T(-1.0);
    util::check_blas(blas_dispatch::scal(
        blas_handle, subSize * subSize, &NegOne, M, 1));
#ifdef RR_DOUBLE_PRECISION
    if constexpr (std::is_same<T, std::complex<float>>::value)
    {
        if (A->isDoublePrecisionEnabled())
        {
            A->copyToSubBlock(subSize * subSize, subSize * subSize);
        }
        else
        {
            A->enableDoublePrecision();
        }

        if (!ritzv.isDoublePrecisionEnabled())
        {
            ritzv.enableDoublePrecision();
        }
        auto A_d = A->getDoublePrecisionMatrix();
        auto ritzv_d = ritzv.getDoublePrecisionMatrix();
        std::complex<double>* workspace_d = nullptr;
        gpu_backend::device_malloc<Backend>(&workspace_d, static_cast<std::size_t>(lwork));
        std::unique_ptr<std::complex<double>, typename util::device_deleter>
            workspace_d_ptr(workspace_d);

        detail::run_heevd_lower<Backend, std::complex<double>>(
            solver_handle, subSize, A_d->l_data() + subSize * subSize, subSize,
            ritzv_d->l_data() + offset, workspace_d, lwork,
            reinterpret_cast<typename util::info_type*>(devInfo));

        ritzv.disableDoublePrecision(true);
        A->copyBackSubBlock(subSize * subSize, subSize * subSize);
        gpu_backend::device_synchronize<Backend>();
    }
    else
    {
#endif
        detail::run_heevd_lower<Backend, T>(
            solver_handle, subSize, M, subSize, ritzv.l_data() + offset,
            workspace, lwork,
            reinterpret_cast<typename util::info_type*>(devInfo));
#ifdef RR_DOUBLE_PRECISION
    }
#endif

    chase::Base<T> ritz_neg_one = chase::Base<T>(-1.0);
    util::check_blas(blas_dispatch::scal(
        blas_handle, subSize, &ritz_neg_one, ritzv.l_data() + offset, 1));

    int info;
    gpu_backend::copy_to_host<Backend>(&info, devInfo, 1);

    if (info != 0)
    {
        throw std::runtime_error("cusolver HEEVD failed in Pseudo-Hermitian "
                                 "RayleighRitz, return value: " +
                                 std::to_string(info));
    }

    util::check_blas(blas_dispatch::trsm(
        blas_handle, util::side_left(), util::fill_lower(), util::op_c(),
        util::diag_non_unit(), subSize, subSize, &One, A->l_data(), subSize, M,
        subSize));

    kernels::template inverse_entries<chase::Base<T>>(
        ritzv.l_data() + offset, subSize, usedStream);

    gpu_backend::stream_synchronize<Backend>(usedStream);
    gpu_backend::copy_to_host<Backend>(ritzv.cpu_data() + offset, ritzv.l_data() + offset, subSize);

    std::vector<chase::Base<T>> vectorNorms(subSize);
    std::vector<T> norm_divider(subSize);

    for (auto idx = 0; idx < subSize; idx++)
    {
        util::check_blas(blas_dispatch::nrm2(
            blas_handle, subSize, M + idx * subSize, 1, &vectorNorms[idx]));
    }

    for (auto idx = 0; idx < subSize; idx++)
    {
        norm_divider[idx] = T(1 / vectorNorms[idx]);
    }

    for (auto idx = 0; idx < subSize; idx++)
    {
        util::check_blas(blas_dispatch::scal(
            blas_handle, subSize, &norm_divider[idx], M + idx * subSize, 1));
    }

    util::check_blas(blas_dispatch::gemm(
        blas_handle, util::op_n(), util::op_n(), V2.l_rows(), subSize, subSize,
        &One, V2.l_data() + offset * V2.l_ld(), V2.l_ld(), M, subSize, &Zero,
        V1.l_data() + offset * V1.l_ld(), V1.l_ld()));
}

} // namespace gpu_mpi
} // namespace internal
} // namespace linalg
} // namespace chase
