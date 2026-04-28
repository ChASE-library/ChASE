// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

/**
 * \defgroup cuda_aware_mpi_kernels chase::linalg::internal::cuda_aware_mpi
 * Namespace
 * \brief The `chase::linalg::internal::cuda_aware_mpi` namespace contains
 * kernels required by ChASE for the distributed-memory GPU using cuda-aware-mpi
 * for communications.
 * @{
 */
#include "../typeTraits.hpp"
#include "external/cublaspp/cublaspp.hpp"
#include "external/cusolverpp/cusolverpp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
/** @} */

namespace chase
{
namespace linalg
{
namespace internal
{
struct cuda_mpi
{
    template <typename T, typename MatrixType, typename InputMultiVectorType>
    static void MatrixMultiplyMultiVectors(
        cublasHandle_t cublas_handle, T* alpha, MatrixType& blockMatrix,
        InputMultiVectorType& input_multiVector, T* beta,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            result_multiVector,
        std::size_t offset, std::size_t subSize);

    template <typename T, typename MatrixType, typename InputMultiVectorType>
    static void MatrixMultiplyMultiVectors(
        cublasHandle_t cublas_handle, T* alpha, MatrixType& blockMatrix,
        InputMultiVectorType& input_multiVector, T* beta,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            result_multiVector);

    // this operation do: W1<-1.0 * H * V1, while redistribute V2 to W2
    template <typename MatrixType, typename InputMultiVectorType>
    static void MatrixMultiplyMultiVectorsAndRedistribute(
        cublasHandle_t cublas_handle, MatrixType& blockMatrix,
        InputMultiVectorType& input_multiVector,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            result_multiVector,
        InputMultiVectorType& src_multiVector,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            target_multiVector,
        std::size_t offset, std::size_t subSize);

    template <typename MatrixType, typename InputMultiVectorType>
    static void MatrixMultiplyMultiVectorsAndRedistribute(
        cublasHandle_t cublas_handle, MatrixType& blockMatrix,
        InputMultiVectorType& input_multiVector,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            result_multiVector,
        InputMultiVectorType& src_multiVector,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            target_multiVector);

    template <typename MatrixType, typename InputMultiVectorType>
    static void
    lanczos_dispatch(cublasHandle_t cublas_handle, std::size_t M,
                     std::size_t numvec, MatrixType& H, InputMultiVectorType& V,
                     chase::Base<typename MatrixType::value_type>* upperb,
                     chase::Base<typename MatrixType::value_type>* ritzv,
                     chase::Base<typename MatrixType::value_type>* Tau,
                     chase::Base<typename MatrixType::value_type>* ritzV);

    template <typename MatrixType, typename InputMultiVectorType>
    static void
    lanczos_dispatch(cublasHandle_t cublas_handle, std::size_t M, MatrixType& H,
                     InputMultiVectorType& V,
                     chase::Base<typename MatrixType::value_type>* upperb);

    template <typename MatrixType, typename InputMultiVectorType>
    static void lanczos(cublasHandle_t cublas_handle, std::size_t M,
                        std::size_t numvec, MatrixType& H,
                        InputMultiVectorType& V,
                        chase::Base<typename MatrixType::value_type>* upperb,
                        chase::Base<typename MatrixType::value_type>* ritzv,
                        chase::Base<typename MatrixType::value_type>* Tau,
                        chase::Base<typename MatrixType::value_type>* ritzV);

    template <typename MatrixType, typename InputMultiVectorType>
    static void lanczos(cublasHandle_t cublas_handle, std::size_t M,
                        MatrixType& H, InputMultiVectorType& V,
                        chase::Base<typename MatrixType::value_type>* upperb);

    template <typename MatrixType, typename InputMultiVectorType>
    static void pseudo_hermitian_lanczos(
        cublasHandle_t cublas_handle, std::size_t M, std::size_t numvec,
        MatrixType& H, InputMultiVectorType& V,
        chase::Base<typename MatrixType::value_type>* upperb,
        chase::Base<typename MatrixType::value_type>* ritzv,
        chase::Base<typename MatrixType::value_type>* Tau,
        chase::Base<typename MatrixType::value_type>* ritzV);

    template <typename MatrixType, typename InputMultiVectorType>
    static void pseudo_hermitian_lanczos(
        cublasHandle_t cublas_handle, std::size_t M, MatrixType& H,
        InputMultiVectorType& V,
        chase::Base<typename MatrixType::value_type>* upperb);

    template <typename T>
    static int cholQR1(cublasHandle_t cublas_handle,
                       cusolverDnHandle_t cusolver_handle, std::size_t m,
                       std::size_t n, T* V, int ldv, MPI_Comm comm,
                       T* workspace = nullptr, int lwork = 0, T* A = nullptr);

    template <typename InputMultiVectorType>
    static int
    cholQR1(cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
            InputMultiVectorType& V,
            typename InputMultiVectorType::value_type* workspace = nullptr,
            int lwork = 0,
            typename InputMultiVectorType::value_type* A = nullptr);

    template <typename T>
    static int cholQR2(cublasHandle_t cublas_handle,
                       cusolverDnHandle_t cusolver_handle, std::size_t m,
                       std::size_t n, T* V, int ldv, MPI_Comm comm,
                       T* workspace = nullptr, int lwork = 0, T* A = nullptr);

    template <typename InputMultiVectorType>
    static int
    cholQR2(cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
            InputMultiVectorType& V,
            typename InputMultiVectorType::value_type* workspace = nullptr,
            int lwork = 0,
            typename InputMultiVectorType::value_type* A = nullptr);

    template <typename T>
    static int shiftedcholQR2(cublasHandle_t cublas_handle,
                              cusolverDnHandle_t cusolver_handle, std::size_t N,
                              std::size_t m, std::size_t n, T* V, int ldv,
                              MPI_Comm comm, T* workspace = nullptr,
                              int lwork = 0, T* A = nullptr);

    template <typename T>
    static int modifiedGramSchmidtCholQR(
        cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
        std::size_t m, std::size_t n, std::size_t locked, T* V, std::size_t ldv,
        MPI_Comm comm, T* workspace = nullptr, int lwork = 0, T* A = nullptr);

    template <typename InputMultiVectorType>
    static void houseHoulderQR(InputMultiVectorType& V);

    /** Runtime tuning knobs for Householder QR (performance/numerical only).
     *  Defaults match the current non-env fallback behavior. */
    struct HouseQRTuning
    {
        std::size_t outer_block_nb = 32;   // level-1: blocked QR panel width (API nb)
        int panel_sub_nb = 8;      // level-3: inner unblocked panel micro-block size
        int formq_chunks = 1;      // backward formQ chunk count
        int timing_blocking = 0;   // per-phase timing host sync: 0 off, 1 on
        int panel_hiprec = 0;      // (unused for CUDA-aware MPI; kept for API parity)
    };

    /** Lightweight timing container for Householder panel factorization. */
    struct HouseholderPanelTiming
    {
        float norm_ms          = 0.f;
        float scalar_kernel_ms = 0.f;
        float allreduce_tau_ms = 0.f;
        float scal_ms          = 0.f;
        float trail_ms         = 0.f;
        /** Full panel: cudaEvent elapsed over entire jj loop (ms), debug/tuning only. */
        float panel_total_ms   = 0.f;
    };

    /** Panel factorization for columns [k, k+jb) in regular 1D block-row layout. */
    template <typename T>
    static void distributed_houseQR_panel_factor(
        std::size_t n, std::size_t l_rows, std::size_t g_off,
        std::size_t ldv, T* V, std::size_t k, std::size_t jb, T* d_tau,
        cublasHandle_t cublas_handle,
        chase::Base<T>* d_real_scalar, T* d_T_scalar,
        T* d_one, T* d_zero, T* d_minus_one, T* d_panel_scalars, T* d_w,
        MPI_Comm mpi_col_comm,
        HouseholderPanelTiming* panel_timing = nullptr);

    /** Panel factorization for 1-D block-cyclic rows (segment metadata). */
    template <typename T>
    static void distributed_houseQR_panel_factor_block_cyclic_1d(
        std::size_t n, std::size_t m_global,
        const std::vector<std::size_t>& seg_global_offs,
        const std::vector<std::size_t>& seg_local_offs,
        const std::vector<std::size_t>& seg_lens,
        std::size_t ldv, T* V, std::size_t k, std::size_t jb, std::size_t nb_dist,
        T* d_tau,
        cublasHandle_t cublas_handle,
        chase::Base<T>* d_real_scalar, T* d_T_scalar,
        T* d_one, T* d_zero, T* d_minus_one, T* d_panel_scalars, T* d_w,
        MPI_Comm mpi_col_comm,
        std::size_t l_rows, const std::uint64_t* d_row_global, T* d_r_diag,
        const HouseQRTuning* tuning = nullptr,
        HouseholderPanelTiming* panel_timing = nullptr);

    /** Column sub-range of block-cyclic panel [jj_begin, jj_end). */
    template <typename T>
    static void distributed_houseQR_panel_factor_block_cyclic_1d_columns(
        std::size_t n, std::size_t m_global,
        const std::vector<std::size_t>& seg_global_offs,
        const std::vector<std::size_t>& seg_local_offs,
        const std::vector<std::size_t>& seg_lens,
        std::size_t ldv, T* V, std::size_t k, std::size_t jj_begin,
        std::size_t jj_end, std::size_t jb_panel_total, std::size_t nb_dist,
        T* d_tau, cublasHandle_t cublas_handle,
        chase::Base<T>* d_real_scalar, T* d_T_scalar, T* d_one, T* d_zero,
        T* d_minus_one, T* d_panel_scalars, T* d_w, MPI_Comm mpi_col_comm,
        int col_rank, int col_size, std::size_t l_rows,
        const std::uint64_t* d_row_global, T* d_r_diag,
        const HouseQRTuning* tuning, HouseholderPanelTiming* panel_timing,
        T* d_sub_workspace, std::size_t d_sub_workspace_elems);

    /** Distributed Householder QR + form Q (unblocked). V on device; CUDA-aware MPI. */
    template <typename T>
    static void distributed_houseQR_formQ(
        std::size_t m_global, std::size_t n, std::size_t l_rows,
        std::size_t g_off, std::size_t ldv, T* V, MPI_Comm mpi_comm,
        cublasHandle_t cublas_handle, T* d_workspace,
        std::size_t lwork_elems, MPI_Comm mpi_col_comm,
        const HouseQRTuning* tuning = nullptr);

    /** Blocked distributed Householder QR + form Q (1D block). V on device; CUDA-aware MPI. */
    template <typename T>
    static void distributed_blocked_houseQR_formQ(
        std::size_t m_global, std::size_t n, std::size_t l_rows,
        std::size_t g_off, std::size_t ldv, T* V, MPI_Comm mpi_comm,
        cublasHandle_t cublas_handle, T* d_workspace,
        std::size_t lwork_elems, MPI_Comm mpi_col_comm,
        const HouseQRTuning* tuning = nullptr);

    /** Unblocked distributed Householder QR + form Q for block-cyclic rows. */
    template <typename T>
    static void distributed_houseQR_formQ_block_cyclic_1d(
        std::size_t m_global, std::size_t n,
        const std::vector<std::size_t>& seg_global_offs,
        const std::vector<std::size_t>& seg_local_offs,
        const std::vector<std::size_t>& seg_lens,
        std::size_t ldv, T* V, std::size_t nb_dist, MPI_Comm mpi_comm,
        cublasHandle_t cublas_handle, T* d_workspace, std::size_t lwork_elems,
        MPI_Comm mpi_col_comm,
        const HouseQRTuning* tuning = nullptr);

    /** Blocked distributed Householder QR + form Q for block-cyclic rows. */
    template <typename T>
    static void distributed_blocked_houseQR_formQ_block_cyclic_1d(
        std::size_t m_global, std::size_t n,
        const std::vector<std::size_t>& seg_global_offs,
        const std::vector<std::size_t>& seg_local_offs,
        const std::vector<std::size_t>& seg_lens,
        std::size_t ldv, T* V, MPI_Comm mpi_comm,
        std::size_t nb_dist,
        cublasHandle_t cublas_handle, T* d_workspace, std::size_t lwork_elems,
        MPI_Comm mpi_col_comm,
        const HouseQRTuning* tuning = nullptr);

    /** GPU-facing entry point for Householder QR (geqrf + form Q) using CUDA-aware MPI. */
    template <typename InputMultiVectorType>
    static void houseQR1_formQ(cublasHandle_t cublas_handle,
                               InputMultiVectorType& V1,
                               typename InputMultiVectorType::value_type* workspace,
                               int lwork,
                               const HouseQRTuning* tuning = nullptr);

    template <typename MatrixType, typename InputMultiVectorType>
    static void rayleighRitz(
        cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
        MatrixType& H, InputMultiVectorType& V1, InputMultiVectorType& V2,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W1,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W2,
        chase::distMatrix::RedundantMatrix<
            chase::Base<typename MatrixType::value_type>, chase::platform::GPU>&
            ritzv,
        std::size_t offset, std::size_t subSize, int* devInfo,
        typename MatrixType::value_type* workspace = nullptr,
        int lwork_heevd = 0,
        chase::distMatrix::RedundantMatrix<typename MatrixType::value_type,
                                           chase::platform::GPU>* A = nullptr);

    template <typename MatrixType, typename InputMultiVectorType>
    static void pseudo_hermitian_rayleighRitz(
        cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
        cusolverDnParams_t params, MatrixType& H, InputMultiVectorType& V1,
        InputMultiVectorType& V2,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W1,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W2,
        chase::distMatrix::RedundantMatrix<
            chase::Base<typename MatrixType::value_type>, chase::platform::GPU>&
            ritzv,
        std::size_t offset, std::size_t subSize, int* devInfo,
        typename MatrixType::value_type* d_workspace = nullptr, int d_lwork = 0,
        typename MatrixType::value_type* h_workspace = nullptr, int h_lwork = 0,
        chase::distMatrix::RedundantMatrix<typename MatrixType::value_type,
                                           chase::platform::GPU>* A = nullptr);

    template <typename MatrixType, typename InputMultiVectorType>
    static void pseudo_hermitian_rayleighRitz_v2(
        cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
        cusolverDnParams_t params, MatrixType& H, InputMultiVectorType& V1,
        InputMultiVectorType& V2,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W1,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W2,
        chase::distMatrix::RedundantMatrix<
            chase::Base<typename MatrixType::value_type>, chase::platform::GPU>&
            ritzv,
        std::size_t offset, std::size_t subSize, int* devInfo,
        typename MatrixType::value_type* d_workspace = nullptr, int d_lwork = 0,
        // typename MatrixType::value_type *h_workspace = nullptr,
        // int h_lwork = 0,
        chase::distMatrix::RedundantMatrix<typename MatrixType::value_type,
                                           chase::platform::GPU>* A = nullptr);

    template <typename MatrixType, typename InputMultiVectorType>
    static void residuals(
        cublasHandle_t cublas_handle, MatrixType& H, InputMultiVectorType& V1,
        InputMultiVectorType& V2,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W1,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W2,
        chase::matrix::Matrix<chase::Base<typename MatrixType::value_type>,
                              typename MatrixType::platform_type>& ritzv,
        chase::matrix::Matrix<chase::Base<typename MatrixType::value_type>,
                              typename MatrixType::platform_type>& resids,
        std::size_t offset, std::size_t subSize);

    template <typename MatrixType>
    static bool checkSymmetryEasy(cublasHandle_t cublas_handle, MatrixType& H);

    template <typename T>
    static void symOrHermMatrix(
        char uplo,
        chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>& H);

    template <typename T>
    static void symOrHermMatrix(
        char uplo,
        chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>& H);

    template <typename MatrixType>
    static void
    shiftDiagonal(MatrixType& H, std::size_t* d_off_m, std::size_t* d_off_n,
                  std::size_t offsize,
                  chase::Base<typename MatrixType::value_type> shift);

    template <typename T>
    static void flipLowerHalfMatrixSign(
        chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>& H);

    template <typename T>
    static void flipLowerHalfMatrixSign(
        chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>& H);

    template <typename InputMultiVectorType>
    static void flipLowerHalfMatrixSign(InputMultiVectorType& V);

    template <typename InputMultiVectorType>
    static void flipLowerHalfMatrixSign(InputMultiVectorType& V,
                                        std::size_t offset,
                                        std::size_t subSize);
};

} // namespace internal
} // namespace linalg
} // namespace chase

#include "linalg/internal/cuda_aware_mpi/cholqr.hpp"
#include "linalg/internal/cuda_aware_mpi/householder_qr.hpp"
#include "linalg/internal/cuda_aware_mpi/flipSign.hpp"
#include "linalg/internal/cuda_aware_mpi/hemm.hpp"
#include "linalg/internal/cuda_aware_mpi/lanczos.hpp"
#include "linalg/internal/cuda_aware_mpi/pseudo_hermitian_lanczos.hpp"
#include "linalg/internal/cuda_aware_mpi/pseudo_hermitian_rayleighRitz.hpp"
#include "linalg/internal/cuda_aware_mpi/rayleighRitz.hpp"
#include "linalg/internal/cuda_aware_mpi/residuals.hpp"
#include "linalg/internal/cuda_aware_mpi/shiftDiagonal.hpp"
#include "linalg/internal/cuda_aware_mpi/symOrHerm.hpp"
