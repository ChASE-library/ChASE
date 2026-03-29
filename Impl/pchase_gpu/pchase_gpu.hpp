// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "algorithm/chaseBase.hpp"
#include "external/lapackpp/lapackpp.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/cuda/cuda_kernels.hpp"
#include "linalg/internal/cuda/lanczos_kernels.hpp"
#include "linalg/internal/cuda_aware_mpi/cuda_mpi_kernels.hpp"
#include "linalg/matrix/matrix.hpp"
#include <cstring>
#include <memory>
#include <random>
#include <vector>
#ifdef HAS_NCCL
#include "linalg/internal/nccl/nccl_kernels.hpp"
#endif
#ifdef HAS_SCALAPACK
#include "external/scalapackpp/scalapackpp.hpp"
#endif
#include "algorithm/types.hpp"
#include "linalg/internal/mpi/cholqr.hpp"

#include "Impl/chase_gpu/nvtx.hpp"

#include "../../linalg/internal/typeTraits.hpp"
#include "Impl/pchase_cpu/pchase_cpu.hpp"
#include <iomanip>
#include <sstream>

using namespace chase::linalg;

template <typename Backend>
struct MGPUKernelNamspaceSelector;

// Specialization for MPI backend
template <>
struct MGPUKernelNamspaceSelector<chase::grid::backend::MPI>
{
    using type = chase::linalg::internal::cuda_mpi; // Use the struct as a type
    template <typename GridType>
    static auto getColCommunicator(GridType* grid)
    {
        return grid->get_col_comm(); // MPI communicator
    }
};

#ifdef HAS_NCCL
// Specialization for NCCL backend
template <>
struct MGPUKernelNamspaceSelector<chase::grid::backend::NCCL>
{
    using type = chase::linalg::internal::cuda_nccl; // Use the struct as a type
    template <typename GridType>
    static auto getColCommunicator(GridType* grid)
    {
        return grid->get_nccl_col_comm(); // NCCL communicator
    }
};

#endif
namespace chase
{
namespace Impl
{
/**
 * @page pChASEGPU
 *
 * @section intro_sec Introduction
 * This class implements the GPU-based parallel version of the Chase algorithm
 * using NVIDIA NCCL (NVIDIA Collective Communication Library) for efficient
 * multi-GPU communication. The class inherits from `ChaseBase` and is designed
 * for solving large-scale generalized eigenvalue problems on distributed GPU
 * environments. It operates on matrix and multi-vector data types, supporting
 * configurations where matrix data and eigenvectors are mapped to a shared MPI
 * grid for distributed computation.
 *
 * @section constructor_sec Constructors and Destructor
 * The constructor for `pChASEGPU` initializes essential components, including
 * the Hamiltonian matrix, input multi-vectors, result multi-vectors, and
 * configurations for NVIDIA libraries such as cuBLAS and cuSolver. It also sets
 * up memory allocations for GPU-based operations and verifies that the matrix
 * is square and that the matrix and eigenvectors share the same MPI grid.
 *
 * @section members_sec Private Members
 * The private members include matrix and multi-vector data, configuration
 * settings, and GPU-specific resources like cuBLAS and cuSolver handles.
 * Additional private members manage MPI-related properties, including process
 * rank and grid coordinates, as well as device memory allocations used for
 * intermediate computations.
 */

/**
 * @brief GPU-based parallel Chase algorithm with NCCL.
 *
 * This class provides an implementation of the Chase algorithm for solving
 * generalized eigenvalue problems on GPU-based platforms. It leverages NVIDIA's
 * NCCL for efficient GPU-to-GPU communication and cuSolver and cuBLAS for
 * GPU-based linear algebra operations. Designed for distributed GPU
 * environments, it operates on matrix and multi-vector data types with MPI
 * parallelism.
 *
 * @tparam MatrixType The matrix type, such as
 * `chase::distMatrix::BlockBlockMatrix` or
 *                    `chase::distMatrix::BlockCyclicMatrix`.
 * @tparam InputMultiVectorType The input multi-vector type, typically used for
 * storing eigenvectors.
 */
template <typename MatrixType, typename InputMultiVectorType,
          typename BackendType = chase::grid::backend::NCCL>
class pChASEGPU : public ChaseBase<typename MatrixType::value_type>
{
    using T = typename MatrixType::value_type;
    using ResultMultiVectorType =
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type;
    // using backend = typename chase::grid::backend::MPI;
    using backend = BackendType;
    //    using kernelNamespace = typename
    //    MGPUKernelNamspaceSelector<backend>::type;
    using kernelNamespace = typename MGPUKernelNamspaceSelector<backend>::type;

public:
    /**
     * @brief Constructs the `pChASEGPU` object.
     *
     * Initializes the Chase algorithm parameters, matrix data, multi-vector
     * data, and GPU-specific resources. Sets up memory allocations for
     * intermediate data and configures cuBLAS and cuSolver for GPU computation.
     *
     * @param nev Number of eigenvalues to compute.
     * @param nex Number of extra vectors for iterative refinement.
     * @param H Pointer to the Hamiltonian matrix.
     * @param V Pointer to the input multi-vector (typically for eigenvectors).
     * @param ritzv Pointer to the Ritz values, used for storing eigenvalues.
     */
    pChASEGPU(std::size_t nev, std::size_t nex, MatrixType* H,
              InputMultiVectorType* V, chase::Base<T>* ritzv)
        : nev_(nev), nex_(nex), nevex_(nev + nex),
          config_(H->g_rows(), nev, nex), N_(H->g_rows())
    {
        SCOPED_NVTX_RANGE();

        if (H->g_rows() != H->g_cols())
        {
            throw std::runtime_error(
                "ChASE requires the matrix solved to be squared");
        }

        if (H->getMpiGrid() != V->getMpiGrid())
        {
            throw std::runtime_error("ChASE requires the matrix and "
                                     "eigenvectors mapped to same MPI grid");
        }

        // Pseudo-Hermitian support currently assumes block-block data layout.
        // Block-cyclic 1D multivectors are not supported for pseudo-Hermitian.
        if constexpr (std::is_same<typename MatrixType::hermitian_type,
                                   chase::matrix::PseudoHermitian>::value)
        {
            if constexpr (chase::distMultiVector::is_block_cyclic_1d_multivector<
                              InputMultiVectorType>::value)
            {
                throw std::runtime_error(
                    "Pseudo-Hermitian pChASEGPU currently supports only "
                    "block-block distributed multivectors (block-cyclic 1D not "
                    "supported).");
            }
        }

        N_ = H->g_rows();
        Hmat_ = H;
        V1_ = V;
        V2_ = V1_->template clone2<InputMultiVectorType>();
        W1_ = V1_->template clone2<ResultMultiVectorType>();
        W2_ = V1_->template clone2<ResultMultiVectorType>();

        if constexpr (std::is_same<typename MatrixType::hermitian_type,
                                   chase::matrix::PseudoHermitian>::value)
        {
            is_sym_ = false;
            is_pseudoHerm_ = true;
        }
        else
        {
            is_sym_ = true;
            is_pseudoHerm_ = false;
        }

        const std::size_t block_size =
            is_pseudoHerm_ ? 2 * nevex_ : nevex_;
        ritzv_ = std::make_unique<chase::distMatrix::RedundantMatrix<
            chase::Base<T>, chase::platform::GPU>>(
            block_size, 1, block_size, ritzv, Hmat_->getMpiGrid_shared_ptr());
        resid_ = std::make_unique<chase::distMatrix::RedundantMatrix<
            chase::Base<T>, chase::platform::GPU>>(
            block_size, 1, Hmat_->getMpiGrid_shared_ptr());

        MPI_Comm_rank(Hmat_->getMpiGrid()->get_comm(), &my_rank_);
        MPI_Comm_size(Hmat_->getMpiGrid()->get_comm(), &nprocs_);
        coords_ = Hmat_->getMpiGrid()->get_coords();
        dims_ = Hmat_->getMpiGrid()->get_dims();

        CHECK_CUBLAS_ERROR(cublasCreate(&cublasH_));
        CHECK_CUSOLVER_ERROR(cusolverDnCreate(&cusolverH_));


#ifdef XGEEV_EXISTS
#ifdef CHASE_OUTPUT
        if (my_rank_ == 0)
        {
            std::ostringstream oss;
            oss << "XGEEV ACTIVATED !\n";
            chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(), my_rank_);
        }
#endif
#endif

        CHECK_CUDA_ERROR(cudaMalloc((void**)&devInfo_, sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_return_, sizeof(T) * block_size));

        int lwork_geqrf = 0;
        int lwork_orgqr = 0;

        CHECK_CUSOLVER_ERROR(
            chase::linalg::cusolverpp::cusolverDnTgeqrf_bufferSize(
                cusolverH_, N_, block_size, V1_->l_data(), V1_->l_ld(),
                &lwork_geqrf));

        CHECK_CUSOLVER_ERROR(
            chase::linalg::cusolverpp::cusolverDnTgqr_bufferSize(
                cusolverH_, N_, block_size, block_size, V1_->l_data(),
                V1_->l_ld(), d_return_, &lwork_orgqr));

        lwork_ = (lwork_geqrf > lwork_orgqr) ? lwork_geqrf : lwork_orgqr;

        int lwork_heevd = 0;

        if constexpr (std::is_same<typename MatrixType::hermitian_type,
                                   chase::matrix::PseudoHermitian>::value)
        {
#ifdef XGEEV_EXISTS
            A_ = std::make_unique<
                chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>(
                3 * 2 * nevex_, 2 * nevex_, Hmat_->getMpiGrid_shared_ptr());

            CHECK_CUSOLVER_ERROR(cusolverDnCreateParams(&params_));

            std::size_t temp_ldwork = 0;
            std::size_t temp_lhwork = 0;

            CHECK_CUSOLVER_ERROR(
                chase::linalg::cusolverpp::cusolverDnTgeev_bufferSize(
                    cusolverH_, params_, CUSOLVER_EIG_MODE_NOVECTOR,
                    CUSOLVER_EIG_MODE_VECTOR, block_size, A_->l_data(),
                    A_->l_ld(), V2_->l_data(), NULL, 1, V1_->l_data(),
                    V1_->l_ld(), &temp_ldwork, &temp_lhwork));

            lwork_heevd = (int)temp_ldwork;
            lhwork_ = (int)temp_lhwork;

#ifdef CHASE_OUTPUT
            if (my_rank_ == 0)
            {
                std::ostringstream oss;
                oss << "GEEV GPU WORKSPACE SIZE = " << lwork_heevd
                    << ", GEEV CPU WORKSPACE SIZE = " << temp_lhwork << "\n";
                chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(), my_rank_);
            }
#endif
            h_work_ = std::unique_ptr<T[]>(new T[lhwork_]);
#else
            A_ = std::make_unique<
                chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>(
                2 * 2 * nevex_, 2 * nevex_, Hmat_->getMpiGrid_shared_ptr());

            CHECK_CUSOLVER_ERROR(
                chase::linalg::cusolverpp::cusolverDnTheevd_bufferSize(
                    cusolverH_, CUSOLVER_EIG_MODE_VECTOR,
                    CUBLAS_FILL_MODE_LOWER, block_size, A_->l_data(), A_->l_ld(),
                    ritzv_->l_data(), &lwork_heevd));
#ifdef CHASE_OUTPUT
            if (my_rank_ == 0)
            {
                std::ostringstream oss;
                oss << "HEEVD GPU WORKSPACE SIZE = " << lwork_heevd << "\n";
                chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(), my_rank_);
            }
#endif
#endif
        }
        else
        {
            A_ = std::make_unique<
                chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>(
                nevex_, nevex_, Hmat_->getMpiGrid_shared_ptr());

            CHECK_CUSOLVER_ERROR(
                chase::linalg::cusolverpp::cusolverDnTheevd_bufferSize(
                    cusolverH_, CUSOLVER_EIG_MODE_VECTOR,
                    CUBLAS_FILL_MODE_LOWER, nevex_, A_->l_data(), A_->l_ld(),
                    ritzv_->l_data(), &lwork_heevd));
        }

        if (lwork_heevd > lwork_)
        {
            lwork_ = lwork_heevd;
        }

        int lwork_potrf = 0;

        CHECK_CUSOLVER_ERROR(
            chase::linalg::cusolverpp::cusolverDnTpotrf_bufferSize(
                cusolverH_, CUBLAS_FILL_MODE_UPPER, block_size, A_->l_data(),
                A_->l_ld(), &lwork_potrf));
        if (lwork_potrf > lwork_)
        {
            lwork_ = lwork_potrf;
        }

        if (block_size * (block_size + 1) / 2 > lwork_)
        {
            lwork_ = block_size * (block_size + 1) / 2;
        }

        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_work_, sizeof(T) * lwork_));

        CHECK_CUDA_ERROR(cudaMalloc(
            (void**)&states_, sizeof(curandStatePhilox4_32_10_t) * (256 * 32)));

        std::vector<std::size_t> diag_xoffs, diag_yoffs;

        if constexpr (std::is_same<typename MatrixType::matrix_type,
                                   chase::distMatrix::BlockBlock>::value)
        {
            std::size_t* g_offs = Hmat_->g_offs();

            for (auto j = 0; j < Hmat_->l_cols(); j++)
            {
                for (auto i = 0; i < Hmat_->l_rows(); i++)
                {
                    if (g_offs[0] + i == g_offs[1] + j)
                    {
                        diag_xoffs.push_back(i);
                        diag_yoffs.push_back(j);
                    }
                }
            }
        }
        else if constexpr (std::is_same<typename MatrixType::matrix_type,
                                        chase::distMatrix::BlockCyclic>::value)
        {
            auto m_contiguous_global_offs = Hmat_->m_contiguous_global_offs();
            auto n_contiguous_global_offs = Hmat_->n_contiguous_global_offs();
            auto m_contiguous_local_offs = Hmat_->m_contiguous_local_offs();
            auto n_contiguous_local_offs = Hmat_->n_contiguous_local_offs();
            auto m_contiguous_lens = Hmat_->m_contiguous_lens();
            auto n_contiguous_lens = Hmat_->n_contiguous_lens();
            auto mblocks = Hmat_->mblocks();
            auto nblocks = Hmat_->nblocks();

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
                                diag_xoffs.push_back(
                                    p + m_contiguous_local_offs[i]);
                                diag_yoffs.push_back(
                                    q + n_contiguous_local_offs[j]);
                            }
                        }
                    }
                }
            }
        }
        else
        {
            throw std::runtime_error("Matrix type is not supported");
        }

        diag_cnt = diag_xoffs.size();

        CHECK_CUDA_ERROR(
            cudaMalloc((void**)&d_diag_xoffs, sizeof(std::size_t) * diag_cnt));
        CHECK_CUDA_ERROR(
            cudaMalloc((void**)&d_diag_yoffs, sizeof(std::size_t) * diag_cnt));

        CHECK_CUDA_ERROR(cudaMemcpy(d_diag_xoffs, diag_xoffs.data(),
                                    sizeof(std::size_t) * diag_cnt,
                                    cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_diag_yoffs, diag_yoffs.data(),
                                    sizeof(std::size_t) * diag_cnt,
                                    cudaMemcpyHostToDevice));

#ifdef QR_DOUBLE_PRECISION
        if constexpr (std::is_same<T, std::complex<float>>::value ||
                      std::is_same<T, float>::value)
        {
            if (!V1_->isDoublePrecisionEnabled())
            {
                V1_->enableDoublePrecision();
            }

            CHECK_CUDA_ERROR(cudaMalloc(
                (void**)&d_work_d,
                sizeof(typename chase::ToDoublePrecisionTrait<T>::Type) *
                    lwork_));
        }
        if constexpr (std::is_same<T, std::complex<float>>::value ||
                      std::is_same<T, float>::value)
        {
            if (!A_->isDoublePrecisionEnabled())
            {
                A_->enableDoublePrecision();
            }
        }
#elif RR_DOUBLE_PRECISION
        if constexpr (std::is_same<T, std::complex<float>>::value ||
                      std::is_same<T, float>::value)
        {
            if (!A_->isDoublePrecisionEnabled())
            {
                A_->enableDoublePrecision();
            }
        }
#endif

#ifdef HAS_NCCL
        // NCCL warm-up: 1x1 matrix to warm up entire path
        if constexpr (std::is_same<backend, chase::grid::backend::NCCL>::value)
        {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
            std::size_t warmup_rows = 1;
            std::size_t warmup_cols = 1;
            
            // Warm up cholQR1 (SYHERK/AllReduce/Cholesky/TRSM on column comm)
            for (int i = 0; i < 3; i++)
            {
                cudaEventRecord(start);
                
                int info = kernelNamespace::cholQR1(
                    cublasH_, cusolverH_, V1_->l_rows(), V1_->l_cols(),
                    V1_->l_data(), V1_->l_ld(),
                    MGPUKernelNamspaceSelector<backend>::getColCommunicator(
                        V1_->getMpiGrid()),
                    d_work_, lwork_, A_->l_data(), devInfo_);
                
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, start, stop);
                
                std::ostringstream oss;
                oss << "[cholQR1 warm-up " << (i+1) << "/3] 1x1 matrix: " 
                    << ms << " ms (info=" << info << ")\n";
                chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(), my_rank_);
            }
            
            // Warm up Lanczos (MatMul + redistribute P2P/AlltoAll patterns)
            chase::Base<T> dummy_upperb = 1.0;
            for (int i = 0; i < 3; i++)
            {
                cudaEventRecord(start);
                cudaStream_t saved_stream = nullptr;
                CHECK_CUBLAS_ERROR(cublasGetStream(cublasH_, &saved_stream));
                // Minimal Lanczos: 1 iteration to warm up redistribute patterns
                //kernelNamespace::lanczos_dispatch(cublasH_, 1, *Hmat_, *V1_, &dummy_upperb);
                std::vector<Base<T>> Theta(40 * 20);
                std::vector<Base<T>> Tau(40 * 20);
                std::vector<Base<T>> ritzV(40 * 40);
                kernelNamespace::lanczos_dispatch(cublasH_, 40, 20, *Hmat_, *V1_, &dummy_upperb, Theta.data(), Tau.data(), ritzV.data());
                
                CHECK_CUBLAS_ERROR(cublasSetStream(cublasH_, saved_stream));
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, start, stop);
#ifdef CHASE_OUTPUT
                if (my_rank_ == 0)
                {
                    std::ostringstream oss;
                    oss << "[Lanczos warm-up " << (i+1) << "/3] 1 iteration: " 
                        << ms << " ms\n";
                    chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(), my_rank_);
                }
#endif
            }
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
#endif

    }

    pChASEGPU(const pChASEGPU&) = delete;

    ~pChASEGPU()
    {
        SCOPED_NVTX_RANGE();

        if (cublasH_)
            CHECK_CUBLAS_ERROR(cublasDestroy(cublasH_));
        if (cusolverH_)
            CHECK_CUSOLVER_ERROR(cusolverDnDestroy(cusolverH_));

        if (d_work_)
            CHECK_CUDA_ERROR(cudaFree(d_work_));
        if (devInfo_)
            CHECK_CUDA_ERROR(cudaFree(devInfo_));
        if (d_return_)
            CHECK_CUDA_ERROR(cudaFree(d_return_));
        if (states_)
            CHECK_CUDA_ERROR(cudaFree(states_));

#ifdef QR_DOUBLE_PRECISION
        if constexpr (std::is_same<T, std::complex<float>>::value ||
                      std::is_same<T, float>::value)
        {
            if (V1_->isDoublePrecisionEnabled())
            {
                V1_->disableDoublePrecision();
            }

            CHECK_CUDA_ERROR(cudaFree(d_work_d));
        }
        if constexpr (std::is_same<T, std::complex<float>>::value ||
                      std::is_same<T, float>::value)
        {
            if (A_->isDoublePrecisionEnabled())
            {
                A_->disableDoublePrecision();
            }
        }
#elif RR_DOUBLE_PRECISION
        if constexpr (std::is_same<T, std::complex<float>>::value ||
                      std::is_same<T, float>::value)
        {
            if (A_->isDoublePrecisionEnabled())
            {
                A_->disableDoublePrecision();
            }
        }
#endif
    }

    std::size_t GetN() const override { return N_; }

    std::size_t GetNev() override { return nev_; }

    std::size_t GetNex() override { return nex_; }

    std::size_t GetRitzvBlockSize() const override
    {
        return is_pseudoHerm_ ? 2 * nevex_ : nevex_;
    }

    std::size_t GetLanczosIter() override { return lanczosIter_; }

    std::size_t GetNumLanczos() override { return numLanczos_; }

    chase::Base<T>* GetRitzv() override { return ritzv_->cpu_data(); }
    chase::Base<T>* GetResid() override
    {
        resid_->allocate_cpu_data();
        return resid_->cpu_data();
    }
    ChaseConfig<T>& GetConfig() override { return config_; }
    int get_nprocs() override { return nprocs_; }
    int get_rank() { return my_rank_; }

    void loadProblemFromFile(std::string filename)
    {
        SCOPED_NVTX_RANGE();
        Hmat_->readFromBinaryFile(filename);
    }

    void saveProblemToFile(std::string filename)
    {
        Hmat_->saveToBinaryFile(filename);
    }

#ifdef CHASE_OUTPUT
    //! Print some intermediate infos during the solving procedure
    void Output(LogLevel level, std::string str,
                const char* category = "algorithm") override
    {
        chase::GetLogger().Log(level, category, str, get_rank());
    }
#endif
    bool checkSymmetryEasy() override
    {
        SCOPED_NVTX_RANGE();

        is_sym_ = kernelNamespace::checkSymmetryEasy(cublasH_, *Hmat_);
        return is_sym_;
    }

    bool isSym() override { return is_sym_; }

    bool checkPseudoHermicityEasy() override
    {
        SCOPED_NVTX_RANGE();
        // is_pseudoHerm_= 0;
        return is_pseudoHerm_;
    }

    bool isPseudoHerm() override { return is_pseudoHerm_; }

    void symOrHermMatrix(char uplo) override
    {
        SCOPED_NVTX_RANGE();

        kernelNamespace::symOrHermMatrix(uplo, *Hmat_);
    }

    void Start() override { locked_ = 0; }

    void initVecs(bool random) override
    {
        SCOPED_NVTX_RANGE();

        if (random)
        {
            int mpi_col_rank;
            MPI_Comm_rank(Hmat_->getMpiGrid()->get_col_comm(), &mpi_col_rank);
            unsigned long long seed = 1337 + mpi_col_rank;

            chase::linalg::internal::cuda::chase_rand_normal(
                seed, states_, V1_->l_data(), V1_->l_ld() * V1_->l_cols(),
                (cudaStream_t)0);
        }

        chase::linalg::internal::cuda::t_lacpy(
            'A', V1_->l_rows(), V1_->l_cols(), V1_->l_data(), V1_->l_ld(),
            V2_->l_data(), V2_->l_ld());

        Hmat_->H2D();
        next_ = NextOp::bAc;
    }

    void ReinitColumns(std::size_t fixednev, std::size_t const* col_indices,
                      std::size_t n_indices) override
    {
        SCOPED_NVTX_RANGE();
        if (n_indices == 0)
            return;
        int mpi_col_rank;
        MPI_Comm_rank(Hmat_->getMpiGrid()->get_col_comm(), &mpi_col_rank);
        unsigned long long base_seed = 1337 + static_cast<unsigned long long>(mpi_col_rank);
        for (std::size_t c = 0; c < n_indices; ++c)
        {
            std::size_t j = fixednev + col_indices[c];
            unsigned long long seed = base_seed + static_cast<unsigned long long>(j) * 1000uLL;
            chase::linalg::internal::cuda::chase_rand_normal(
                seed, states_, V1_->l_data() + j * V1_->l_ld(), V1_->l_rows(),
                (cudaStream_t)0);
        }
        for (std::size_t c = 0; c < n_indices; ++c)
        {
            std::size_t j = fixednev + col_indices[c];
            chase::linalg::internal::cuda::t_lacpy(
                'A', V1_->l_rows(), 1, V1_->l_data() + j * V1_->l_ld(),
                V1_->l_ld(), V2_->l_data() + j * V2_->l_ld(), V2_->l_ld());
        }
    }

    void Lanczos(std::size_t m, chase::Base<T>* upperb) override
    {
        SCOPED_NVTX_RANGE();

        lanczosIter_ = m;
        numLanczos_ = 1;

        cudaStream_t saved_stream = nullptr;
        CHECK_CUBLAS_ERROR(cublasGetStream(cublasH_, &saved_stream));
        kernelNamespace::lanczos_dispatch(cublasH_, m, *Hmat_, *V1_, upperb);
        CHECK_CUBLAS_ERROR(cublasSetStream(cublasH_, saved_stream));
    }

    void Lanczos(std::size_t M, std::size_t numvec, chase::Base<T>* upperb,
                 chase::Base<T>* ritzv, chase::Base<T>* Tau,
                 chase::Base<T>* ritzV) override
    {

        SCOPED_NVTX_RANGE();

        lanczosIter_ = M;
        numLanczos_ = numvec;

        cudaStream_t saved_stream = nullptr;
        CHECK_CUBLAS_ERROR(cublasGetStream(cublasH_, &saved_stream));
        kernelNamespace::lanczos_dispatch(cublasH_, M, numvec, *Hmat_, *V1_,
                                          upperb, ritzv, Tau, ritzV);
        CHECK_CUBLAS_ERROR(cublasSetStream(cublasH_, saved_stream));
    }

    void LanczosDos(std::size_t idx, std::size_t m, T* ritzVc) override
    {
        SCOPED_NVTX_RANGE();

        T One = T(1.0);
        T Zero = T(0.0);

        std::unique_ptr<T, chase::cuda::utils::CudaDeleter> d_ritzVc_ptr =
            nullptr;
        T* d_ritzVc_;
        CHECK_CUDA_ERROR(cudaMalloc(&d_ritzVc_, m * idx * sizeof(T)));
        d_ritzVc_ptr.reset(d_ritzVc_);
        d_ritzVc_ = d_ritzVc_ptr.get();

        CHECK_CUDA_ERROR(cudaMemcpy(d_ritzVc_, ritzVc, m * idx * sizeof(T),
                                    cudaMemcpyHostToDevice));

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
            cublasH_, CUBLAS_OP_N, CUBLAS_OP_N, V1_->l_rows(), idx, m, &One,
            V1_->l_data(), V1_->l_ld(), d_ritzVc_, m, &Zero, V2_->l_data(),
            V2_->l_ld()));

        chase::linalg::internal::cuda::t_lacpy('A', V2_->l_rows(), m,
                                               V2_->l_data(), V2_->l_ld(),
                                               V1_->l_data(), V1_->l_ld());
    }

    void Shift(T c, bool isunshift = false) override
    {
        SCOPED_NVTX_RANGE();

        if (isunshift)
        {
            next_ = NextOp::bAc;
        }

        kernelNamespace::shiftDiagonal(*Hmat_, d_diag_xoffs, d_diag_yoffs,
                                       diag_cnt, std::real(c));

#ifdef ENABLE_MIXED_PRECISION
        if constexpr (std::is_same<T, double>::value ||
                      std::is_same<T, std::complex<double>>::value)
        {
            auto min = *std::min_element(resid_->cpu_data() + locked_,
                                         resid_->cpu_data() + nev_);
            bool shouldEnableSP = (min > 1e-3 && !isunshift);
            auto updatePrecision = [&](auto& mat, bool copyback = false)
            {
                if (shouldEnableSP)
                {
                    mat->enableSinglePrecision();
                }
                else if (mat->isSinglePrecisionEnabled())
                {
                    mat->disableSinglePrecision(copyback);
                }
            };

            // Update precision for all matrices
            updatePrecision(Hmat_);
            updatePrecision(V1_, true); // Special case for V1_
            updatePrecision(W1_);

            // Message on enabling single precision
            if (shouldEnableSP && my_rank_ == 0 && !isunshift)
            {
                chase::GetLogger().Log(chase::LogLevel::Info, "linalg",
                    "Enable Single Precision in Filter", my_rank_);
            }
        }
#endif
    }

    void HEMM(std::size_t block, T alpha, T beta, std::size_t offset_left,
              std::size_t offset_right = 0) override
    {
        std::size_t ncols =
            (offset_right < block) ? (block - offset_right) : std::size_t(0);
        if (ncols == 0)
        {
            next_ = (next_ == NextOp::bAc) ? NextOp::cAb : NextOp::bAc;
            return;
        }
#ifdef ENABLE_MIXED_PRECISION
        if constexpr (std::is_same<T, double>::value ||
                      std::is_same<T, std::complex<double>>::value)
        {
            using singlePrecisionT =
                typename chase::ToSinglePrecisionTrait<T>::Type;
            auto min = *std::min_element(resid_->cpu_data() + locked_,
                                         resid_->cpu_data() + nev_);

            if (min > 1e-3)
            {
                auto Hmat_sp = Hmat_->getSinglePrecisionMatrix();
                auto V1_sp = V1_->getSinglePrecisionMatrix();
                auto W1_sp = W1_->getSinglePrecisionMatrix();
                singlePrecisionT alpha_sp =
                    static_cast<singlePrecisionT>(alpha);
                singlePrecisionT beta_sp = static_cast<singlePrecisionT>(beta);

                if (next_ == NextOp::bAc)
                {
                    kernelNamespace::template MatrixMultiplyMultiVectors<
                        singlePrecisionT>(cublasH_, &alpha_sp, *Hmat_sp, *V1_sp,
                                          &beta_sp, *W1_sp,
                                          offset_left + locked_, ncols);
                    next_ = NextOp::cAb;
                }
                else
                {
                    kernelNamespace::template MatrixMultiplyMultiVectors<
                        singlePrecisionT>(cublasH_, &alpha_sp, *Hmat_sp, *W1_sp,
                                          &beta_sp, *V1_sp,
                                          offset_left + locked_, ncols);
                    next_ = NextOp::bAc;
                }
            }
            else
            {
                if (next_ == NextOp::bAc)
                {
                    kernelNamespace::MatrixMultiplyMultiVectors(
                        cublasH_, &alpha, *Hmat_, *V1_, &beta, *W1_,
                        offset_left + locked_, ncols);
                    next_ = NextOp::cAb;
                }
                else
                {
                    kernelNamespace::MatrixMultiplyMultiVectors(
                        cublasH_, &alpha, *Hmat_, *W1_, &beta, *V1_,
                        offset_left + locked_, ncols);
                    next_ = NextOp::bAc;
                }
            }
        }
        else
#endif

        {
            if (next_ == NextOp::bAc)
            {
                kernelNamespace::MatrixMultiplyMultiVectors(
                    cublasH_, &alpha, *Hmat_, *V1_, &beta, *W1_,
                    offset_left + locked_, ncols);
                next_ = NextOp::cAb;
            }
            else
            {
                kernelNamespace::MatrixMultiplyMultiVectors(
                    cublasH_, &alpha, *Hmat_, *W1_, &beta, *V1_,
                    offset_left + locked_, ncols);
                next_ = NextOp::bAc;
            }
        }
    }

    void HEMM_H2(std::size_t block, T alpha, T beta, T gamma,
                 std::size_t offset_left,
                 std::size_t offset_right = 0) override
    {
        SCOPED_NVTX_RANGE();
        std::size_t ncols =
            (offset_right < block) ? (block - offset_right) : std::size_t(0);
        if (ncols == 0)
        {
            next_ = (next_ == NextOp::bAc) ? NextOp::cAb : NextOp::bAc;
            return;
        }
        const std::size_t col0 = offset_left + locked_;
        T one = T(1);
        T zero = T(0);

        if (next_ == NextOp::bAc)
        {
            kernelNamespace::MatrixMultiplyMultiVectors(
                cublasH_, &one, *Hmat_, *V1_, &zero, *W1_, col0, ncols);
            kernelNamespace::MatrixMultiplyMultiVectors(
                cublasH_, &alpha, *Hmat_, *W1_, &beta, *V2_, col0, ncols);
            chase::linalg::internal::cuda::batchedAxpyScalar(
                gamma, V1_->l_data() + col0 * V1_->l_ld(),
                V2_->l_data() + col0 * V2_->l_ld(),
                static_cast<int>(V2_->l_rows()), static_cast<int>(ncols),
                static_cast<int>(V1_->l_ld()), static_cast<int>(V2_->l_ld()));
            next_ = NextOp::cAb;
        }
        else
        {
            kernelNamespace::MatrixMultiplyMultiVectors(
                cublasH_, &one, *Hmat_, *V2_, &zero, *W1_, col0, ncols);
            kernelNamespace::MatrixMultiplyMultiVectors(
                cublasH_, &alpha, *Hmat_, *W1_, &beta, *V1_, col0, ncols);
            chase::linalg::internal::cuda::batchedAxpyScalar(
                gamma, V2_->l_data() + col0 * V2_->l_ld(),
                V1_->l_data() + col0 * V1_->l_ld(),
                static_cast<int>(V1_->l_rows()), static_cast<int>(ncols),
                static_cast<int>(V2_->l_ld()), static_cast<int>(V1_->l_ld()));
            next_ = NextOp::bAc;
        }
    }

    void ApplyKconjugate(std::size_t block) override {
        if constexpr (std::is_same<typename MatrixType::hermitian_type,
            chase::matrix::PseudoHermitian>::value)
        {
            V1_->Kconjugate(block, locked_);
            // Symmetric locking: layout [locked_ | first half | second half | locked_].
            // Rows: 0..N/2-1 = upper block, N/2..N-1 = lower block.
            // Active first half [locked_, locked_+block), second half [locked_+block, locked_+2*block).
            // K-conjugation: set second-half cols = conjugate(first-half) with block swap.

            //First, we need to know to whom each rank has to send and receive data.
        }
    }

    void QR(std::size_t fixednev, chase::Base<T> cond) override
    {
        SCOPED_NVTX_RANGE();

        // Create CUDA events for detailed timing
        cudaEvent_t qr_start, qr_flip_sign, qr_cholqr_start, qr_cholqr_end, qr_lacpy_start, qr_end;
        cudaEventCreate(&qr_start);
        cudaEventCreate(&qr_flip_sign);
        cudaEventCreate(&qr_cholqr_start);
        cudaEventCreate(&qr_cholqr_end);
        cudaEventCreate(&qr_lacpy_start);
        cudaEventCreate(&qr_end);
        cudaEventRecord(qr_start);
        cudaEventRecord(qr_cholqr_start);
        cudaEventRecord(qr_cholqr_end);

        float time_hh_core_ms = 0.0f;
        float time_hh_fallback_core_ms = 0.0f;
        float time_chol_shifted_ms = 0.0f;
        float time_chol1_ms = 0.0f;
        float time_chol2_ms = 0.0f;
        float time_copy_to_ms = 0.0f;
        float time_copy_back_ms = 0.0f;

        auto measure_cuda = [&](float& acc_ms, auto&& fn) {
            cudaEvent_t ev_a, ev_b;
            cudaEventCreate(&ev_a);
            cudaEventCreate(&ev_b);
            cudaEventRecord(ev_a);
            fn();
            cudaEventRecord(ev_b);
            cudaEventSynchronize(ev_b);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev_a, ev_b);
            acc_ms += ms;
            cudaEventDestroy(ev_a);
            cudaEventDestroy(ev_b);
        };

        int disable = config_.DoCholQR() ? 0 : 1;
        char* cholddisable = getenv("CHASE_DISABLE_CHOLQR");
        if (cholddisable)
        {
            disable = std::atoi(cholddisable);
        }

        int info = 1;

        if constexpr (std::is_same<typename MatrixType::hermitian_type,
                                   chase::matrix::PseudoHermitian>::value)
        {
            chase::linalg::internal::cuda::t_lacpy('A', V1_->l_rows(), locked_,
            V1_->l_data() + (V1_->l_cols() - locked_ ) * V1_->l_ld(), V1_->l_ld(), V2_->l_data() + locked_ * V2_->l_ld(),
            V1_->l_ld());

            /* The right eigenvectors are not orthonormal in the QH case, but
             * S-orthonormal. Therefore, we S-orthonormalize the locked vectors
             * against the current subspace By flipping the sign of the lower
             * part of the locked vectors. First, we need to copy the unconverged 
               vectors to the end of the vector space*/

            chase::linalg::internal::cuda::t_lacpy('A', V1_->l_rows(), V1_->l_cols() - 2*locked_,
            V1_->l_data() + locked_ * V1_->l_ld(), V1_->l_ld(),
            V2_->l_data() + 2 * locked_ * V2_->l_ld(), V2_->l_ld());

            V1_->swap_l_data_ptr(*V2_);

            kernelNamespace::flipLowerHalfMatrixSign(*V1_, 0, 2*locked_);

            cudaEventRecord(qr_flip_sign);
        }
        else
        {
            cudaEventRecord(qr_flip_sign);
        }
#ifdef ChASE_DISPLAY_COND_V_SVD
        if constexpr (std::is_same<typename MatrixType::matrix_type,
                                   chase::distMatrix::BlockCyclic>::value &&
                      std::is_same<typename MatrixType::hermitian_type,
                                   chase::matrix::Hermitian>::value)
        {
            static constexpr chase::distMultiVector::CommunicatorType
                communicator_type = InputMultiVectorType::communicator_type;
            auto V_tmp = std::make_unique<
                chase::distMultiVector::DistMultiVectorBlockCyclic1D<
                    T, communicator_type, chase::platform::CPU>>(
                V1_->g_rows(), V1_->g_cols() - locked_, V1_->mb(),
                V1_->getMpiGrid_shared_ptr());
            CHECK_CUDA_ERROR(cudaMemcpy(
                V_tmp->l_data(), V1_->l_data() + locked_ * V1_->l_ld(),
                (V1_->g_cols() - locked_) * V1_->l_ld() * sizeof(T),
                cudaMemcpyDeviceToHost));
            // auto cond_v = kernelNamespace::computeConditionNumber(*V_tmp);
            auto cond_v =
                chase::linalg::internal::cpu_mpi::computeConditionNumber(
                    *V_tmp);

            if (my_rank_ == 0)
            {
                std::cout << "Exact condition number of V from SVD: " << cond_v
                          << std::endl;
            }
        }
#endif
        if (disable == 1 || cond != 1.0)
        {
#ifdef QR_DOUBLE_PRECISION  
            if constexpr (std::is_same<T, std::complex<float>>::value ||
                              std::is_same<T, float>::value)
            {
                measure_cuda(time_copy_to_ms, [&]() { V1_->copyTo(); });
                auto V1_d = V1_->getDoublePrecisionMatrix();

                if constexpr (chase::distMultiVector::is_block_cyclic_1d_multivector<InputMultiVectorType>::value)
                {
#if defined(HAS_NCCL)
                    if constexpr (std::is_same<backend,
                                  chase::grid::backend::NCCL>::value)
                    {
#ifdef CHASE_OUTPUT
                        if (my_rank_ == 0)
                        {
                            chase::GetLogger().Log(
                                chase::LogLevel::Debug, "linalg",
                                "Householder QR: NCCL GPU (block-cyclic 1D).\n",
                                0);
                        }
#endif
                        measure_cuda(time_hh_core_ms, [&]() {
                            kernelNamespace::houseQR1_formQ(
                                cublasH_, *V1_d,
                                reinterpret_cast<typename chase::ToDoublePrecisionTrait<T>::Type*>(d_work_d),
                                lwork_, 16);
                        });
                    }
                    else
#endif
                    {
#ifdef HAS_SCALAPACK
#ifdef CHASE_OUTPUT
                        if (my_rank_ == 0)
                        {
                            chase::GetLogger().Log(
                                chase::LogLevel::Warn, "linalg",
                                "[Warning]: Using ScaLAPACK Householder QR for block-cyclic 1D (MPI / non-NCCL build).\n",
                                0);
                        }
#endif
                        measure_cuda(time_hh_core_ms, [&]() {
                            kernelNamespace::houseHoulderQR(*V1_d);
                        });
#else
                        throw std::runtime_error(
                            "Block-cyclic 1D distributed Householder QR requires "
                            "NCCL (backend::NCCL) or ScaLAPACK.\n");
#endif
                    }
                }
                else
                {
                    kernelNamespace::houseQR1_formQ(
                        cublasH_, *V1_d,
                        reinterpret_cast<typename chase::ToDoublePrecisionTrait<T>::Type*>(
                            d_work_d),
                        lwork_, 16);
                }
                measure_cuda(time_copy_back_ms, [&]() { V1_->copyback(); });
            }
            else
#endif
            {
                if constexpr (chase::distMultiVector::is_block_cyclic_1d_multivector<InputMultiVectorType>::value)
                {
#if defined(HAS_NCCL)
                    if constexpr (std::is_same<backend,
                                  chase::grid::backend::NCCL>::value)
                    {
#ifdef CHASE_OUTPUT
                        if (my_rank_ == 0)
                        {
                            chase::GetLogger().Log(
                                chase::LogLevel::Debug, "linalg",
                                "Householder QR: NCCL GPU (block-cyclic 1D).\n",
                                0);
                        }
#endif
                        measure_cuda(time_hh_core_ms, [&]() {
                            kernelNamespace::houseQR1_formQ(
                                cublasH_, *V1_, d_work_, lwork_, 16);
                        });
                    }
                    else
#endif
                    {
#ifdef HAS_SCALAPACK
#ifdef CHASE_OUTPUT
                        if (my_rank_ == 0)
                        {
                            chase::GetLogger().Log(
                                chase::LogLevel::Warn, "linalg",
                                "[Warning]: Using ScaLAPACK Householder QR for block-cyclic 1D (MPI / non-NCCL build).\n",
                                0);
                        }
#endif
                        measure_cuda(time_hh_core_ms, [&]() {
                            kernelNamespace::houseHoulderQR(*V1_);
                        });
#else
                        throw std::runtime_error(
                            "Block-cyclic 1D distributed Householder QR requires "
                            "NCCL (backend::NCCL) or ScaLAPACK.\n");
#endif
                    }
                }
                else
                {
                    measure_cuda(time_hh_core_ms, [&]() {
                        kernelNamespace::houseQR1_formQ(
                            cublasH_, *V1_, d_work_, lwork_, 16);
                    });
                }

            }
        }   
        else
        {
            Base<T> cond_threshold_upper = (sizeof(Base<T>) == 8) ? 1e8 : 1e4;
            Base<T> cond_threshold_lower = (sizeof(Base<T>) == 8) ? 2e1 : 1e1;

            char* chol_threshold = getenv("CHASE_CHOLQR1_THLD");
            if (chol_threshold)
            {
                cond_threshold_lower = std::atof(chol_threshold);
            }

#ifdef CHASE_OUTPUT
            if (my_rank_ == 0)
            {
                std::ostringstream oss;
                oss << std::setprecision(2) << "cond(V): " << cond;
                chase::GetLogger().Log(chase::LogLevel::Info, "linalg",
                    oss.str(), my_rank_);
            }
#endif

            int info = 1;

            cudaEventRecord(qr_cholqr_start);

            if (cond > cond_threshold_upper)
            {
#ifdef CHASE_OUTPUT
                /*
                if(my_rank_ == 0){
                    std::cout << "Entering Shifted Cholesky QR 2" << std::endl;
                }*/
#endif
#ifdef QR_DOUBLE_PRECISION
                if constexpr (std::is_same<T, std::complex<float>>::value ||
                              std::is_same<T, float>::value)
                {
                    measure_cuda(time_copy_to_ms, [&]() { V1_->copyTo(); });

                    auto V1_d = V1_->getDoublePrecisionMatrix();
                    auto A_d = A_->getDoublePrecisionMatrix();
                    measure_cuda(time_chol_shifted_ms, [&]() {
                        info = kernelNamespace::shiftedcholQR2(
                            cublasH_, cusolverH_, V1_->g_rows(), V1_->l_rows(),
                            V1_->l_cols(), V1_d->l_data(), V1_->l_ld(),
                            // V1_->getMpiGrid()->get_nccl_col_comm(),
                            MGPUKernelNamspaceSelector<backend>::getColCommunicator(
                                V1_->getMpiGrid()),
                            reinterpret_cast<
                                typename chase::ToDoublePrecisionTrait<T>::Type*>(
                                d_work_d),
                            lwork_, A_d->l_data(), devInfo_);
                    });
                    measure_cuda(time_copy_back_ms, [&]() { V1_->copyback(); });
                }
                else
                {
#endif
                    measure_cuda(time_chol_shifted_ms, [&]() {
                        info = kernelNamespace::shiftedcholQR2(
                            cublasH_, cusolverH_, V1_->g_rows(), V1_->l_rows(),
                            V1_->l_cols(), V1_->l_data(), V1_->l_ld(),
                            // V1_->getMpiGrid()->get_nccl_col_comm(),
                            MGPUKernelNamspaceSelector<backend>::getColCommunicator(
                                V1_->getMpiGrid()),
                            d_work_, lwork_, A_->l_data(), devInfo_);
                    });
#ifdef QR_DOUBLE_PRECISION
                }
#endif
            }
            else if (cond < cond_threshold_lower)
            {
                /*if constexpr (std::is_same<T, std::complex<float>>::value ||
                std::is_same<T, float>::value)
                {
                    if(V1_->isDoublePrecisionEnabled())
                    {
                        V1_->disableDoublePrecision();
                    }
                    if(A_->isDoublePrecisionEnabled())
                    {
                        A_->disableDoublePrecision();
                    }
                }
                */

#ifdef CHASE_OUTPUT
                if (0)
                {
#ifdef QR_DOUBLE_PRECISION
                    std::cout << "QR_RR_DOUBLE ACTVIATED" << std::endl;
#else
                    std::cout << "QR_RR_DOUBLE DISABLED" << std::endl;
#endif
                    std::cout << "Entering Cholesky QR 1" << std::endl;
                }
#endif
#ifdef QR_DOUBLE_PRECISION
                if constexpr (std::is_same<T, std::complex<float>>::value ||
                              std::is_same<T, float>::value)
                {
                    measure_cuda(time_copy_to_ms, [&]() { V1_->copyTo(); });

                    auto V1_d = V1_->getDoublePrecisionMatrix();
                    auto A_d = A_->getDoublePrecisionMatrix();
                    measure_cuda(time_chol1_ms, [&]() {
                        info = kernelNamespace::cholQR1(
                            cublasH_, cusolverH_, V1_->l_rows(), V1_->l_cols(),
                            V1_d->l_data(), V1_->l_ld(),
                            // V1_->getMpiGrid()->get_nccl_col_comm(),
                            MGPUKernelNamspaceSelector<backend>::getColCommunicator(
                                V1_->getMpiGrid()),
                            reinterpret_cast<
                                typename chase::ToDoublePrecisionTrait<T>::Type*>(
                                d_work_d),
                            lwork_, A_d->l_data(), devInfo_);
                    });
                    measure_cuda(time_copy_back_ms, [&]() { V1_->copyback(); });
                }
                else
                {
#endif
                    measure_cuda(time_chol1_ms, [&]() {
                        info = kernelNamespace::cholQR1(
                            cublasH_, cusolverH_, V1_->l_rows(), V1_->l_cols(),
                            V1_->l_data(), V1_->l_ld(),
                            // V1_->getMpiGrid()->get_nccl_col_comm(),
                            MGPUKernelNamspaceSelector<backend>::getColCommunicator(
                                V1_->getMpiGrid()),
                            d_work_, lwork_, A_->l_data(), devInfo_);
                    });
#ifdef QR_DOUBLE_PRECISION
                }
#endif
            }
            else
            {
                /*if constexpr (std::is_same<T, std::complex<float>>::value ||
                std::is_same<T, float>::value)
                {
                    if(V1_->isDoublePrecisionEnabled())
                    {
                        V1_->disableDoublePrecision();
                    }
                    if(A_->isDoublePrecisionEnabled())
                    {
                        A_->disableDoublePrecision();
                    }
                }*/

#ifdef CHASE_OUTPUT
                /*
                if(my_rank_ == 0){
                    std::cout << "Entering Cholesky QR 2" << std::endl;
                }*/
#endif
#ifdef QR_DOUBLE_PRECISION
                if constexpr (std::is_same<T, std::complex<float>>::value ||
                              std::is_same<T, float>::value)
                {
                    measure_cuda(time_copy_to_ms, [&]() { V1_->copyTo(); });

                    auto V1_d = V1_->getDoublePrecisionMatrix();
                    auto A_d = A_->getDoublePrecisionMatrix();
                    measure_cuda(time_chol2_ms, [&]() {
                        info = kernelNamespace::cholQR2(
                            cublasH_, cusolverH_, V1_->l_rows(), V1_->l_cols(),
                            V1_d->l_data(), V1_->l_ld(),
                            // V1_->getMpiGrid()->get_nccl_col_comm(),
                            MGPUKernelNamspaceSelector<backend>::getColCommunicator(
                                V1_->getMpiGrid()),
                            reinterpret_cast<
                                typename chase::ToDoublePrecisionTrait<T>::Type*>(
                                d_work_d),
                            lwork_, A_d->l_data(), devInfo_);
                    });
                    measure_cuda(time_copy_back_ms, [&]() { V1_->copyback(); });
                }
                else
                {
#endif
                    measure_cuda(time_chol2_ms, [&]() {
                        info = kernelNamespace::cholQR2(
                            cublasH_, cusolverH_, V1_->l_rows(), V1_->l_cols(),
                            V1_->l_data(), V1_->l_ld(),
                            // V1_->getMpiGrid()->get_nccl_col_comm(),
                            MGPUKernelNamspaceSelector<backend>::getColCommunicator(
                                V1_->getMpiGrid()),
                            d_work_, lwork_, A_->l_data(), devInfo_);
                    });
#ifdef QR_DOUBLE_PRECISION
                }
#endif
            }

            cudaEventRecord(qr_cholqr_end);

            if (info != 0)
            {
#ifdef CHASE_OUTPUT
                if (my_rank_ == 0)
                {
                    chase::GetLogger().Log(chase::LogLevel::Warn, "linalg",
                        "CholeskyQR doesn't work, Householder QR will be used.\n",
                        my_rank_);
                }
#endif
#ifdef QR_DOUBLE_PRECISION
                if constexpr (std::is_same<T, std::complex<float>>::value ||
                              std::is_same<T, float>::value)
                {
                    measure_cuda(time_copy_to_ms, [&]() { V1_->copyTo(); });
                    auto V1_d = V1_->getDoublePrecisionMatrix();
                    {
                        if constexpr (chase::distMultiVector::is_block_cyclic_1d_multivector<InputMultiVectorType>::value)
                        {
#if defined(HAS_NCCL)
                            if constexpr (std::is_same<backend,
                                          chase::grid::backend::NCCL>::value)
                            {
#ifdef CHASE_OUTPUT
                                if (my_rank_ == 0)
                                {
                                    chase::GetLogger().Log(
                                        chase::LogLevel::Debug, "linalg",
                                        "Householder QR (CholQR fallback): NCCL GPU (block-cyclic 1D).\n",
                                        0);
                                }
#endif
                                measure_cuda(time_hh_fallback_core_ms, [&]() {
                                    kernelNamespace::houseQR1_formQ(
                                        cublasH_, *V1_d,
                                        reinterpret_cast<typename chase::ToDoublePrecisionTrait<T>::Type*>(d_work_d),
                                        lwork_, 16u);
                                });
                            }
                            else
#endif
                            {
#ifdef HAS_SCALAPACK
#ifdef CHASE_OUTPUT
                                if (my_rank_ == 0)
                                {
                                    chase::GetLogger().Log(
                                        chase::LogLevel::Warn, "linalg",
                                        "[Warning]: Using ScaLAPACK Householder QR for block-cyclic 1D (MPI / non-NCCL build).\n",
                                        0);
                                }
#endif
                                measure_cuda(time_hh_fallback_core_ms, [&]() {
                                    kernelNamespace::houseHoulderQR(*V1_d);
                                });
#else
                                throw std::runtime_error(
                                    "Block-cyclic 1D distributed Householder QR requires "
                                    "NCCL (backend::NCCL) or ScaLAPACK.\n");
#endif
                            }
                        }
                        else
                        {
                            measure_cuda(time_hh_fallback_core_ms, [&]() {
                                kernelNamespace::houseQR1_formQ(
                                    cublasH_, *V1_d,
                                    reinterpret_cast<typename chase::ToDoublePrecisionTrait<T>::Type*>(d_work_d),
                                    lwork_, 16u);
                            });
                        }
                    }
                    measure_cuda(time_copy_back_ms, [&]() { V1_->copyback(); });
                }
                else
#endif
                {
                    {
                        if constexpr (chase::distMultiVector::is_block_cyclic_1d_multivector<InputMultiVectorType>::value)
                        {
#if defined(HAS_NCCL)
                            if constexpr (std::is_same<backend,
                                          chase::grid::backend::NCCL>::value)
                            {
#ifdef CHASE_OUTPUT
                                if (my_rank_ == 0)
                                {
                                    chase::GetLogger().Log(
                                        chase::LogLevel::Debug, "linalg",
                                        "Householder QR (CholQR fallback): NCCL GPU (block-cyclic 1D).\n",
                                        0);
                                }
#endif
                                measure_cuda(time_hh_fallback_core_ms, [&]() {
                                    kernelNamespace::houseQR1_formQ(
                                        cublasH_, *V1_, d_work_, lwork_, 16u);
                                });
                            }
                            else
#endif
                            {
#ifdef HAS_SCALAPACK
#ifdef CHASE_OUTPUT
                                if (my_rank_ == 0)
                                {
                                    chase::GetLogger().Log(
                                        chase::LogLevel::Warn, "linalg",
                                        "[Warning]: Using ScaLAPACK Householder QR for block-cyclic 1D (MPI / non-NCCL build).\n",
                                        0);
                                }
#endif
                                measure_cuda(time_hh_fallback_core_ms, [&]() {
                                    kernelNamespace::houseHoulderQR(*V1_);
                                });
#else
                                throw std::runtime_error(
                                    "Block-cyclic 1D distributed Householder QR requires "
                                    "NCCL (backend::NCCL) or ScaLAPACK.\n");
#endif
                            }
                        }
                        else
                        {
                            measure_cuda(time_hh_fallback_core_ms, [&]() {
                                kernelNamespace::houseQR1_formQ(
                                    cublasH_, *V1_, d_work_, lwork_, 16u);
                            });
                        }
                    }
                }
            }
        }

        cudaEventRecord(qr_lacpy_start);

        std::size_t unconverged_cols = GetRitzvBlockSize() - locked_;

        if constexpr (std::is_same<typename MatrixType::hermitian_type,
            chase::matrix::PseudoHermitian>::value)
        {
            unconverged_cols = GetRitzvBlockSize() - 2*locked_;

            chase::linalg::internal::cuda::t_lacpy('A', V1_->l_rows(), V1_->l_cols() - 2*locked_,
            V1_->l_data() + 2 * locked_ * V1_->l_ld(),V1_->l_ld(),
            V2_->l_data() + locked_ * V2_->l_ld(), V2_->l_ld());

            V1_->swap_l_data_ptr(*V2_);

            chase::linalg::internal::cuda::t_lacpy('A', V2_->l_rows(), locked_,
            V1_->l_data(), V1_->l_ld(),
            V2_->l_data(), V2_->l_ld());

            chase::linalg::internal::cuda::t_lacpy('A', V2_->l_rows(), locked_,
            V1_->l_data() + (V1_->l_cols() - locked_ ) * V1_->l_ld(), V1_->l_ld(),
            V2_->l_data() + (V2_->l_cols() - locked_ ) * V2_->l_ld(), V2_->l_ld());

        }else{
            unconverged_cols = GetRitzvBlockSize() - locked_;

            chase::linalg::internal::cuda::t_lacpy('A', V2_->l_rows(), locked_,
            V2_->l_data(), V2_->l_ld(),
            V1_->l_data(), V1_->l_ld());
        }
        
        chase::linalg::internal::cuda::t_lacpy(
            'A', V2_->l_rows(), unconverged_cols,
            V1_->l_data() + V1_->l_ld() * locked_, V1_->l_ld(),
            V2_->l_data() + V2_->l_ld() * locked_, V2_->l_ld());

        cudaEventRecord(qr_end);
        cudaEventSynchronize(qr_end);

        cudaDeviceSynchronize();

        // Calculate and print detailed timings
        float time_flip_sign = 0.0f, time_cholqr = 0.0f, time_lacpy = 0.0f, time_total = 0.0f;
        cudaEventElapsedTime(&time_flip_sign, qr_start, qr_flip_sign);
        cudaEventElapsedTime(&time_cholqr, qr_cholqr_start, qr_cholqr_end);
        cudaEventElapsedTime(&time_lacpy, qr_lacpy_start, qr_end);
        cudaEventElapsedTime(&time_total, qr_start, qr_end);
#ifdef CHASE_OUTPUT
        if (my_rank_ == 0)
        {
            std::ostringstream oss;
            oss << std::setprecision(6) << std::fixed;
            oss << "\n[QR Timing Rank 0] Total: " << time_total/1000.0 << " s, "
                << "FlipSign: " << time_flip_sign/1000.0 << " s, "
                << "CholQR: " << time_cholqr/1000.0 << " s, "
                << "Lacpy: " << time_lacpy/1000.0 << " s\n";
            oss << "  Breakdown: copyTo=" << time_copy_to_ms/1000.0 << " s, "
                << "copyBack=" << time_copy_back_ms/1000.0 << " s, "
                << "HH(core)=" << time_hh_core_ms/1000.0 << " s, "
                << "HH(fallback)=" << time_hh_fallback_core_ms/1000.0 << " s, "
                << "ShiftedCholQR2=" << time_chol_shifted_ms/1000.0 << " s, "
                << "CholQR1=" << time_chol1_ms/1000.0 << " s, "
                << "CholQR2=" << time_chol2_ms/1000.0 << " s\n";
            chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(), my_rank_);
        }
#endif
        // Cleanup events
        cudaEventDestroy(qr_start);
        cudaEventDestroy(qr_flip_sign);
        cudaEventDestroy(qr_cholqr_start);
        cudaEventDestroy(qr_cholqr_end);
        cudaEventDestroy(qr_lacpy_start);
        cudaEventDestroy(qr_end);
    }

    void RR(chase::Base<T>* ritzv, std::size_t block) override
    {
        SCOPED_NVTX_RANGE();

        cudaDeviceSynchronize();

        if constexpr (std::is_same<typename MatrixType::hermitian_type,
                                   chase::matrix::PseudoHermitian>::value)
        {
#ifdef XGEEV_EXISTS
            kernelNamespace::pseudo_hermitian_rayleighRitz(
                cublasH_, cusolverH_, params_, *Hmat_, *V1_, *V2_, *W1_, *W2_,
                *ritzv_, locked_, 2*block, devInfo_, d_work_, lwork_,
                h_work_.get(), lhwork_, A_.get());
#else
            kernelNamespace::pseudo_hermitian_rayleighRitz_v2(
                cublasH_, cusolverH_, params_, *Hmat_, *V1_, *V2_, *W1_, *W2_,
                *ritzv_, locked_, 2*block, devInfo_, d_work_, lwork_, A_.get());
#endif

            chase::linalg::internal::cuda::t_lacpy(
                'A', V2_->l_rows(), 2*block, V1_->l_data() + locked_ * V1_->l_ld(),
                V1_->l_ld(), V2_->l_data() + locked_ * V2_->l_ld(), V2_->l_ld());
        }
        else
        {

            kernelNamespace::rayleighRitz(
                cublasH_, cusolverH_, *Hmat_, *V1_, *V2_, *W1_, *W2_, *ritzv_,
                locked_, block, devInfo_, d_work_, lwork_, A_.get());

            chase::linalg::internal::cuda::t_lacpy(
                'A', V2_->l_rows(), block, V1_->l_data() + locked_ * V1_->l_ld(),
                V1_->l_ld(), V2_->l_data() + locked_ * V2_->l_ld(), V2_->l_ld());
        }

        cudaDeviceSynchronize();
    }

    void Sort(chase::Base<T>* ritzv, chase::Base<T>* residLast,
              chase::Base<T>* resid) override
    {
    }
    //TODO / TO DO : Remove fixednev. 
    void Resd(chase::Base<T>* ritzv, chase::Base<T>* resd,
              std::size_t fixednev) override
    {
        SCOPED_NVTX_RANGE();

        // Pseudo-Hermitian: subspace size is 2*nevex_; standard: nevex_
        /*std::size_t subSize =
            (is_pseudoHerm_ ? 2 * nevex_ : nevex_) - locked_;*/
        std::size_t subSize = nevex_ - locked_;
        kernelNamespace::residuals(cublasH_, *Hmat_, *V1_, *V2_, *W1_, *W2_,
                                   ritzv_->loc_matrix(), resid_->loc_matrix(),
                                   locked_, subSize);
    }

    void Swap(std::size_t i, std::size_t j) override
    {
        V1_->swap_ij(i, j);
        V2_->swap_ij(i, j);
    }

    void Lock(std::size_t new_converged) override
    {
        SCOPED_NVTX_RANGE();

        locked_ += new_converged;
    }

    void End() override
    {
        SCOPED_NVTX_RANGE();
        V1_->D2H();
    }

private:
    /**
     * @enum NextOp
     * @brief Represents the next operation to be performed in the computation.
     */
    enum NextOp
    {
        cAb, /**< Represents the operation `c = A * b`. */
        bAc  /**< Represents the operation `b = A * c`. */
    };

    NextOp next_; /**< Holds the next operation in the computation sequence. */

    bool is_sym_;        /**< Indicates whether the matrix is symmetric. */
    bool is_pseudoHerm_; /**< Indicates whether the matrix is pseudo-hermitian.
                          */

    std::size_t nev_; /**< Number of eigenvalues to compute. */
    std::size_t
        nex_; /**< Number of additional vectors for iterative refinement. */
    std::size_t nevex_; /**< Total number of vectors (nev + nex) used in the
                           algorithm. */
    std::size_t
        locked_; /**< Count of locked vectors in the eigenvalue problem. */
    std::size_t lanczosIter_; /**< Number of Lanczos Iterations.*/
    std::size_t numLanczos_;  /**< Number of Runs of Lanczos.*/

    std::size_t N_; /**< Dimension of the square matrix. */

    int nprocs_;  /**< Total number of MPI processes. */
    int my_rank_; /**< Rank of the current MPI process. */
    int* coords_; /**< Pointer to coordinates of the current process in the MPI
                     grid. */
    int* dims_;   /**< Pointer to dimensions of the MPI grid. */

    MatrixType* Hmat_; /**< Pointer to the Hamiltonian matrix. */
    InputMultiVectorType*
        V1_; /**< Pointer to the first input multi-vector for eigenvectors. */
    std::unique_ptr<InputMultiVectorType>
        V2_; /**< Unique pointer to the second input multi-vector for
                computations. */
    std::unique_ptr<ResultMultiVectorType>
        W1_; /**< Unique pointer to the first result multi-vector. */
    std::unique_ptr<ResultMultiVectorType>
        W2_; /**< Unique pointer to the second result multi-vector. */

    std::unique_ptr<chase::distMatrix::RedundantMatrix<chase::Base<T>,
                                                       chase::platform::GPU>>
        ritzv_; /**< Matrix holding Ritz values (eigenvalues). */
    std::unique_ptr<chase::distMatrix::RedundantMatrix<chase::Base<T>,
                                                       chase::platform::GPU>>
        resid_; /**< Matrix holding residuals for eigenvalues. */
    std::unique_ptr<chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>
        A_; /**< Auxiliary matrix for intermediate calculations. */

    cudaStream_t stream_; /**< CUDA stream for asynchronous GPU operations. */
    cublasHandle_t cublasH_; /**< Handle to the cuBLAS library for GPU linear
                                algebra operations. */
    cusolverDnHandle_t cusolverH_; /**< Handle to the cuSolver library for
                                      GPU-based eigenvalue solvers. */
    cusolverDnParams_t
        params_; /**< CUSOLVER structure with information for Xgeev. */


    curandStatePhilox4_32_10_t* states_ =
        NULL; /**< Random number generator state for GPU, used for
                 initializations. */

    int* devInfo_; /**< Pointer to device memory for storing operation status
                      (e.g., success or failure) in cuSolver calls. */
    T* d_return_;  /**< Pointer to device memory for storing results of GPU
                      operations. */
    T* d_work_; /**< Pointer to workspace on the device for GPU operations. */
    int lwork_ =
        0; /**< Size of the workspace on the device, used for GPU operations. */
    void* d_work_d;

    std::unique_ptr<T[]> h_work_; /**< Pointer to work buffer on host for geev
                                     in the Pseudo Hermitian case. */
    int lhwork_ = 0; /**< Workspace size for host geev operations in the Pseudo
                        Hermitian case. */

    std::size_t* d_diag_xoffs; /**< Pointer to device memory holding x offsets
                                  for diagonal elements in computations. */
    std::size_t* d_diag_yoffs; /**< Pointer to device memory holding y offsets
                                  for diagonal elements in computations. */
    std::size_t
        diag_cnt; /**< Count of diagonal elements used in the algorithm. */

    chase::ChaseConfig<T>
        config_; /**< Configuration settings for the Chase algorithm, including
                    problem parameters. */
};

} // namespace Impl
} // namespace chase
