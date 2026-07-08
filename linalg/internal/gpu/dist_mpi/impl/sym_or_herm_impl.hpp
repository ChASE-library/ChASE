// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/typeTraits.hpp"
#include "algorithm/types.hpp"
#include "external/scalapack/scalapackpp/scalapackpp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"
#include "linalg/internal/gpu/backend/handle_utils.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <mpi.h>
#include <omp.h>
#include <stdexcept>
#include <vector>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace gpu_mpi
{
namespace detail
{

} // namespace detail

template <typename Backend, typename MpiKernelNamespace, typename BlasHandle,
          typename MatrixType>
bool check_symmetry_easy_impl(BlasHandle blas_handle, MatrixType& H)
{
    using T = typename MatrixType::value_type;
    using ColumnMultiVectorType =
        typename ColumnMultiVectorType<MatrixType>::type;
    using RowMultiVectorType = typename RowMultiVectorType<MatrixType>::type;
    using kernel_ops = gpu_device::kernels<Backend>;
    using util = gpu_backend::handle_utils<Backend>;

    T One = T(1.0);
    T Zero = T(0.0);

    std::size_t N = H.g_rows();

    auto v = H.template cloneMultiVector<ColumnMultiVectorType>(N, 1);
    auto u = H.template cloneMultiVector<RowMultiVectorType>(N, 1);
    auto uT = H.template cloneMultiVector<ColumnMultiVectorType>(N, 1);
    auto v_2 = H.template cloneMultiVector<RowMultiVectorType>(N, 1);

    int* coords = H.getMpiGrid()->get_coords();
    unsigned long long seed = 1337 + coords[0];

    std::unique_ptr<typename kernel_ops::rand_state,
                    typename util::device_deleter>
        states(kernel_ops::allocate_states());
    kernel_ops::template fill_random<T>(
        seed, states.get(), v.l_data(), v.l_ld() * v.l_cols());

    v.redistributeImpl(&v_2);

    MpiKernelNamespace::MatrixMultiplyMultiVectors(blas_handle, &One, H, v,
                                                   &Zero, u);
    MpiKernelNamespace::MatrixMultiplyMultiVectors(blas_handle, &One, H, v_2,
                                                   &Zero, uT);

    u.redistributeImpl(&v);

    v.allocate_cpu_data();
    uT.allocate_cpu_data();

    v.D2H();
    uT.D2H();

    bool is_sym = true;

    for (auto i = 0; i < static_cast<int>(v.l_rows()); i++)
    {
        if (std::abs(v.cpu_data()[i] - uT.cpu_data()[i]) > 1e-10)
        {
            is_sym = false;
            break;
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, &is_sym, 1, MPI_CXX_BOOL, MPI_LAND,
                  H.getMpiGrid()->get_col_comm());

    return is_sym;
}

template <typename T>
void sym_or_herm_matrix_impl(
    char uplo, chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>& H)
{
#ifdef HAS_SCALAPACK
    std::size_t* desc = H.scalapack_descriptor_init();

    std::size_t xlen = H.l_rows();
    std::size_t ylen = H.l_cols();
    std::size_t* g_offs = H.g_offs();
    std::size_t h_ld = H.cpu_ld();

    if (uplo == 'U')
    {
#pragma omp parallel for
        for (auto j = 0; j < static_cast<int>(ylen); j++)
        {
            for (auto i = 0; i < static_cast<int>(xlen); i++)
            {
                if (g_offs[0] + i == g_offs[1] + j)
                {
                    H.cpu_data()[i + j * h_ld] *= T(0.5);
                }
                else if (g_offs[0] + i > g_offs[1] + j)
                {
                    H.cpu_data()[i + j * h_ld] = T(0.0);
                }
            }
        }
    }
    else
    {
#pragma omp parallel for
        for (auto j = 0; j < static_cast<int>(ylen); j++)
        {
            for (auto i = 0; i < static_cast<int>(xlen); i++)
            {
                if (g_offs[0] + i == g_offs[1] + j)
                {
                    H.cpu_data()[i + j * h_ld] *= T(0.5);
                }
                else if (g_offs[0] + i < g_offs[1] + j)
                {
                    H.cpu_data()[i + j * h_ld] = T(0.0);
                }
            }
        }
    }

    T One = T(1.0);
    T Zero = T(0.0);
    int zero = 0;
    int one = 1;
    std::vector<T> tmp(H.l_rows() * (H.l_cols() + 2 * H.l_cols()));
    chase::linalg::scalapackpp::t_ptranc(H.g_rows(), H.g_cols(), One,
                                         H.cpu_data(), one, one, desc, Zero,
                                         tmp.data(), one, one, desc);
#pragma omp parallel for
    for (auto j = 0; j < static_cast<int>(H.l_cols()); j++)
    {
        for (auto i = 0; i < static_cast<int>(H.l_rows()); i++)
        {
            H.cpu_data()[i + j * H.l_ld()] += tmp[i + j * H.l_rows()];
        }
    }
#else
    throw std::runtime_error("For ChASE-MPI, symOrHermMatrix requires ScaLAPACK, "
                             "which is not detected\n");
#endif
}

template <typename T>
void sym_or_herm_matrix_impl(
    char uplo, chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>& H)
{
#ifdef HAS_SCALAPACK
    std::size_t* desc = H.scalapack_descriptor_init();

    auto m_contiguous_global_offs = H.m_contiguous_global_offs();
    auto n_contiguous_global_offs = H.n_contiguous_global_offs();
    auto m_contiguous_local_offs = H.m_contiguous_local_offs();
    auto n_contiguous_local_offs = H.n_contiguous_local_offs();
    auto m_contiguous_lens = H.m_contiguous_lens();
    auto n_contiguous_lens = H.n_contiguous_lens();
    auto mblocks = H.mblocks();
    auto nblocks = H.nblocks();

    if (uplo == 'U')
    {
        for (std::size_t j = 0; j < nblocks; j++)
        {
            for (std::size_t i = 0; i < mblocks; i++)
            {
                for (std::size_t q = 0; q < n_contiguous_lens[j]; q++)
                {
                    for (std::size_t p = 0; p < m_contiguous_lens[i]; p++)
                    {
                        if (q + n_contiguous_global_offs[j] ==
                            p + m_contiguous_global_offs[i])
                        {
                            H.cpu_data()[(q + n_contiguous_local_offs[j]) *
                                             H.cpu_ld() +
                                         p + m_contiguous_local_offs[i]] *=
                                T(0.5);
                        }
                        else if (q + n_contiguous_global_offs[j] <
                                 p + m_contiguous_global_offs[i])
                        {
                            H.cpu_data()[(q + n_contiguous_local_offs[j]) *
                                             H.cpu_ld() +
                                         p + m_contiguous_local_offs[i]] =
                                T(0.0);
                        }
                    }
                }
            }
        }
    }
    else
    {
        for (std::size_t j = 0; j < nblocks; j++)
        {
            for (std::size_t i = 0; i < mblocks; i++)
            {
                for (std::size_t q = 0; q < n_contiguous_lens[j]; q++)
                {
                    for (std::size_t p = 0; p < m_contiguous_lens[i]; p++)
                    {
                        if (q + n_contiguous_global_offs[j] ==
                            p + m_contiguous_global_offs[i])
                        {
                            H.cpu_data()[(q + n_contiguous_local_offs[j]) *
                                             H.cpu_ld() +
                                         p + m_contiguous_local_offs[i]] *=
                                T(0.5);
                        }
                        else if (q + n_contiguous_global_offs[j] >
                                 p + m_contiguous_global_offs[i])
                        {
                            H.cpu_data()[(q + n_contiguous_local_offs[j]) *
                                             H.cpu_ld() +
                                         p + m_contiguous_local_offs[i]] =
                                T(0.0);
                        }
                    }
                }
            }
        }
    }

    T One = T(1.0);
    T Zero = T(0.0);
    int zero = 0;
    int one = 1;
    std::vector<T> tmp(H.l_rows() * (H.l_cols() + 2 * H.nb()));
    chase::linalg::scalapackpp::t_ptranc(H.g_rows(), H.g_cols(), One,
                                         H.cpu_data(), one, one, desc, Zero,
                                         tmp.data(), one, one, desc);
#pragma omp parallel for
    for (auto j = 0; j < static_cast<int>(H.l_cols()); j++)
    {
        for (auto i = 0; i < static_cast<int>(H.l_rows()); i++)
        {
            H.cpu_data()[i + j * H.l_ld()] += tmp[i + j * H.l_rows()];
        }
    }
#else
    throw std::runtime_error("For ChASE-MPI, symOrHermMatrix requires ScaLAPACK, "
                             "which is not detected\n");
#endif
}

} // namespace gpu_mpi
} // namespace internal
} // namespace linalg
} // namespace chase
