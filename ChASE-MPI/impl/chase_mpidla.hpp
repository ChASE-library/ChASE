/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2023, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "ChASE-MPI/chase_mpi_matrices.hpp"
#include "ChASE-MPI/chase_mpi_properties.hpp"
#include "ChASE-MPI/chase_mpidla_interface.hpp"
#include <iterator>
#include <map>
#include <mpi.h>
#include <numeric>

namespace chase
{
namespace mpi
{
//! @brief A derived class of ChaseMpiDLAInterface which implements mostly the
//! MPI collective communications part of ChASE-MPI targeting the
//! distributed-memory systens with or w/o GPUs.
/*! The computation in node are mostly implemented in ChaseMpiDLABlaslapack and
   ChaseMpiDLAMultiGPU. It supports both `Block Distribution` and `Block-Cyclic
   Distribution` schemes.
*/
template <class T>
class ChaseMpiDLA : public ChaseMpiDLAInterface<T>
{
public:
    //! A constructor of ChaseMpiDLA.
    /*!
      @param matrix_properties: it is an instance of ChaseMpiProperties, which
      defines the MPI environment and data distribution scheme in ChASE-MPI.
      @param matrices: it is an instance of ChaseMpiMatrices, which
      allocates the required buffers in ChASE-MPI.
      @param dla: it is an object of ChaseMpiDLAInterface, which defines the
      implementation of in-node computation for ChASE-MPI. Currently, it can be
      one of ChaseMpiDLABlaslapack and ChaseMpiDLAMultiGPU.
    */
    ChaseMpiDLA(ChaseMpiProperties<T>* matrix_properties,
                ChaseMpiDLAInterface<T>* dla)
        : dla_(dla)
    {
#ifdef USE_NSIGHT
        nvtxRangePushA("ChaseMpiDLA: Init");
#endif

        ldc_ = matrix_properties->get_ldc();
        ldb_ = matrix_properties->get_ldb();
        N_ = matrix_properties->get_N();
        n_ = matrix_properties->get_n();
        m_ = matrix_properties->get_m();

        nev_ = matrix_properties->GetNev();
        nex_ = matrix_properties->GetNex();
        std::size_t max_block_ = matrix_properties->get_max_block();
        matrix_properties_ = matrix_properties;

        row_comm_ = matrix_properties->get_row_comm();
        col_comm_ = matrix_properties->get_col_comm();

        dims_ = matrix_properties->get_dims();
        coord_ = matrix_properties->get_coord();
        off_ = matrix_properties->get_off();

        data_layout = matrix_properties->get_dataLayout();

        matrix_properties->get_offs_lens(r_offs_, r_lens_, r_offs_l_, c_offs_,
                                         c_lens_, c_offs_l_);
        mb_ = matrix_properties->get_mb();
        nb_ = matrix_properties->get_nb();

        icsrc_ = matrix_properties->get_icsrc();
        irsrc_ = matrix_properties->get_irsrc();

        mblocks_ = matrix_properties->get_mblocks();
        nblocks_ = matrix_properties->get_nblocks();

        int sign = 0;
        if (data_layout.compare("Block-Cyclic") == 0)
        {
            sign = 1;
        }

        Buff_.resize(sign * N_);

        istartOfFilter_ = true;

        MPI_Comm_size(row_comm_, &row_size_);
        MPI_Comm_rank(row_comm_, &row_rank_);
        MPI_Comm_size(col_comm_, &col_size_);
        MPI_Comm_rank(col_comm_, &col_rank_);

        send_lens_ = matrix_properties_->get_sendlens();
        block_counts_ = matrix_properties_->get_blockcounts();
        blocklens_ = matrix_properties_->get_blocklens();
        blockdispls_ = matrix_properties_->get_blockdispls();
        g_offset_ = matrix_properties_->get_g_offsets();

        for (auto dim = 0; dim < 2; dim++)
        {
            block_cyclic_displs_[dim].resize(dims_[dim]);
            int displs_cnt = 0;
            for (auto j = 0; j < dims_[dim]; ++j)
            {
                block_cyclic_displs_[dim][j].resize(block_counts_[dim][j]);
                for (auto i = 0; i < block_counts_[dim][j]; ++i)
                {
                    block_cyclic_displs_[dim][j][i] = displs_cnt;
                    displs_cnt += blocklens_[dim][j][i];
                }
            }
        }

        for (auto dim = 0; dim < 2; dim++)
        {
            newType_[dim].resize(dims_[dim]);
            for (auto j = 0; j < dims_[dim]; ++j)
            {
                int array_of_sizes[2] = {static_cast<int>(N_), 1};
                int array_of_subsizes[2] = {
                    static_cast<int>(send_lens_[dim][j]), 1};
                int array_of_starts[2] = {block_cyclic_displs_[dim][j][0], 0};

                MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes,
                                         array_of_starts, MPI_ORDER_FORTRAN,
                                         getMPI_Type<T>(), &(newType_[dim][j]));

                MPI_Type_commit(&(newType_[dim][j]));
            }
        }

#if defined(HAS_SCALAPACK)
        desc1D_Nxnevx_ = matrix_properties->get_desc1D_Nxnevx();
#endif

        isSameDist_ = row_size_ == col_size_ && irsrc_ == icsrc_ && mb_ == nb_;
        if (isSameDist_)
        {
            for (auto i = 0; i < col_size_; i++)
            {
                c_dests.push_back(i);
                c_srcs.push_back(i);
                c_lens.push_back(send_lens_[0][i]);
                b_disps.push_back(0);
                c_disps.push_back(0);
            }
        }
        else
        {
            int c_dest = icsrc_;
            c_dests.push_back(c_dest);
            int c_src = irsrc_;
            c_srcs.push_back(c_src);
            int c_len = 1;
            int b_disp = 0;
            int c_disp = 0;
            b_disps.push_back(b_disp);
            c_disps.push_back(c_disp);

            for (auto i = 1; i < N_; i++)
            {
                auto src_tmp = (i / mb_) % col_size_;
                auto dest_tmp = (i / nb_) % row_size_;
                if (dest_tmp == c_dest && src_tmp == c_src)
                {
                    c_len += 1;
                }
                else
                {
                    c_lens.push_back(c_len);
                    c_dest = (i / nb_) % row_size_;
                    b_disp = i % nb_ + ((i / nb_) / row_size_) * nb_;
                    c_disp = i % mb_ + ((i / mb_) / col_size_) * mb_;
                    c_src = (i / mb_) % col_size_;
                    c_srcs.push_back(c_src);
                    c_dests.push_back(c_dest);
                    b_disps.push_back(b_disp);
                    c_disps.push_back(c_disp);
                    c_len = 1;
                }
            }
            c_lens.push_back(c_len);
        }


        if (isSameDist_)
        {
            for (auto i = 0; i < row_size_; i++)
            {
                b_dests.push_back(i);
                b_srcs.push_back(i);
                b_lens.push_back(send_lens_[1][i]);
                c_disps_2.push_back(0);
                b_disps_2.push_back(0);
            }
        }
        else
        {
            int b_dest = irsrc_;
            b_dests.push_back(b_dest);
            int b_src = icsrc_;
            b_srcs.push_back(b_src);
            int b_len = 1;
            int c_disp = 0;
            int b_disp = 0;
            c_disps_2.push_back(c_disp);
            b_disps_2.push_back(b_disp);

            for (auto i = 1; i < N_; i++)
            {
                auto src_tmp = (i / nb_) % row_size_;
                auto dest_tmp = (i / mb_) % col_size_;
                if (dest_tmp == b_dest && src_tmp == b_src)
                {
                    b_len += 1;
                }
                else
                {
                    b_lens.push_back(b_len);
                    b_dest = (i / mb_) % col_size_;
                    c_disp = i % mb_ + ((i / mb_) / col_size_) * mb_;
                    b_disp = i % nb_ + ((i / nb_) / row_size_) * nb_;
                    b_src = (i / nb_) % row_size_;
                    b_srcs.push_back(b_src);
                    b_dests.push_back(b_dest);
                    c_disps_2.push_back(c_disp);
                    b_disps_2.push_back(b_disp);
                    b_len = 1;
                }
            }
            b_lens.push_back(b_len);
        }

        reqsb2c_.resize(b_lens.size());
        b_sends_.resize(b_lens.size());
        c_recvs_.resize(b_lens.size());

        for (auto i = 0; i < b_lens.size(); i++)
        {
            if (col_rank_ == b_dests[i])
            {
                if (row_rank_ == b_srcs[i])
                {
                    int n1 = send_lens_[1][row_rank_];
                    int array_of_sizes[2] = {n1, 1};
                    int array_of_subsizes[2] = {b_lens[i], 1};
                    int array_of_starts[2] = {b_disps_2[i], 0};

                    MPI_Type_create_subarray(
                        2, array_of_sizes, array_of_subsizes, array_of_starts,
                        MPI_ORDER_FORTRAN, getMPI_Type<T>(), &(b_sends_[i]));
                    MPI_Type_commit(&(b_sends_[i]));
                }
                else
                {
                    int array_of_sizes2[2] = {static_cast<int>(m_), 1};
                    int array_of_starts2[2] = {c_disps_2[i], 0};
                    int array_of_subsizes2[2] = {b_lens[i], 1};
                    MPI_Type_create_subarray(
                        2, array_of_sizes2, array_of_subsizes2,
                        array_of_starts2, MPI_ORDER_FORTRAN, getMPI_Type<T>(),
                        &(c_recvs_[i]));
                    MPI_Type_commit(&(c_recvs_[i]));
                }
            }
        }

        reqsc2b_.resize(c_lens.size());
        c_sends_.resize(c_lens.size());
        b_recvs_.resize(c_lens.size());

        for (auto i = 0; i < c_lens.size(); i++)
        {
            if (row_rank_ == c_dests[i])
            {
                if (col_rank_ == c_srcs[i])
                {
                    int n1 = send_lens_[0][col_rank_];
                    int array_of_sizes[2] = {n1, 1};
                    int array_of_subsizes[2] = {c_lens[i], 1};
                    int array_of_starts[2] = {c_disps_2[i], 0};

                    MPI_Type_create_subarray(
                        2, array_of_sizes, array_of_subsizes, array_of_starts,
                        MPI_ORDER_FORTRAN, getMPI_Type<T>(), &(c_sends_[i]));
                    MPI_Type_commit(&(c_sends_[i]));
                }
                else
                {
                    int array_of_sizes2[2] = {static_cast<int>(n_), 1};
                    int array_of_starts2[2] = {b_disps_2[i], 0};
                    int array_of_subsizes2[2] = {c_lens[i], 1};
                    MPI_Type_create_subarray(
                        2, array_of_sizes2, array_of_subsizes2,
                        array_of_starts2, MPI_ORDER_FORTRAN, getMPI_Type<T>(),
                        &(b_recvs_[i]));
                    MPI_Type_commit(&(b_recvs_[i]));
                }
            }
        }

        mpi_wrapper_ = matrix_properties->get_mpi_wrapper();
        matrices_ = dla_->getChaseMatrices();
        C = matrices_->C_comm();
        B = matrices_->B_comm();
        A = matrices_->A_comm();
        C2 = matrices_->C2_comm();
        B2 = matrices_->B2_comm();
        vv = matrices_->vv_comm();
        rsd = matrices_->Resid_comm();

        if (matrices_->get_Mode() == 2)
        {
            cuda_aware_ = true;
        }
        else
        {
            cuda_aware_ = false;
        }

        if (cuda_aware_)
        {
            memcpy_mode[0] = CPY_D2D;
            memcpy_mode[1] = CPY_D2H;
            memcpy_mode[2] = CPY_H2D;
#if defined(HAS_NCCL)
            allreduce_backend = NCCL_BACKEND;
            bcast_backend = NCCL_BACKEND;
#else
            allreduce_backend = MPI_BACKEND;
            bcast_backend = MPI_BACKEND;
#endif
        }
        else
        {
            memcpy_mode[0] = CPY_H2H;
            memcpy_mode[1] = CPY_H2H;
            memcpy_mode[2] = CPY_H2H;
            allreduce_backend = MPI_BACKEND;
            bcast_backend = MPI_BACKEND;
        }

        if(!isSameDist_){
            auto max_c_len = *max_element(c_lens.begin(), c_lens.end());
            if(cuda_aware_){
                buff__ = std::make_unique<Matrix<T>>(2, max_c_len, nex_ + nev_);
            }else{
                buff__ = std::make_unique<Matrix<T>>(0, max_c_len, nex_ + nev_);                
            }
        }

        //std::cout << "numbvecs = " << config_->GetNumLanczos() << std::endl;

#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
    }

    ~ChaseMpiDLA()
    {}

    //! In ChaseMpiDLA, this function consists of operations
    /*!
     *   - backup of `C_` to `C2_`
     *   - set the switch pointer of apply() to `bAc`
     *   - set the switch pointer of ChaseMpiDLABlaslapack::apply()
     *     and ChaseMpiDLABlaslapack::MultiGPU() to `bAc`
     */
    void initVecs() override
    {
#ifdef USE_NSIGHT
        nvtxRangePushA("ChaseMpiDLA: initVecs");
#endif
        next_ = NextOp::bAc;
        dla_->initVecs();
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
    }

    //! generating initial random vectors when it is necessary
    //!     - different MPI proc/GPUs generate parts of the required random
    //!     numbers
    //!     - MPI proc/GPUs with the same rank within the same column
    //!     communicator
    //!       share the same seed of RNG (random number generator)
    //!     - the generation of random numbers taking place within
    //!     ChaseMpiDLABlaslapack::initRndVecs()
    //!       and ChaseMpiDLAMultiGPU::initRndVecs(), targetting different
    //!       architectures
    //!         - in ChaseMpiDLABlaslapack, random numbers on each MPI rank is
    //!         generated in
    //!           sequence with C++ STL random generator
    //!         - in ChaseMpiDLAMultiGPU, random numbers on each MPI rank is
    //!         generated in parallel
    //!           on related GPU based on the device API of <a
    //!           href="https://docs.nvidia.com/c
    //! uda/curand/index.html">cuRAND</a>.
    void initRndVecs() override
    {
#ifdef USE_NSIGHT
        nvtxRangePushA("ChaseMpiDLA: initRndVecs");
#endif
        auto nevex = nev_ + nex_;
        T one = T(1.0);
        T zero = T(0.0);
#ifdef USE_NSIGHT
        nvtxRangePushA("random generation");
#endif
        dla_->initRndVecs();
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif

#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
    }

    void preApplication(T* V, std::size_t locked, std::size_t block) override
    {
#ifdef USE_NSIGHT
        nvtxRangePushA("ChaseMpiDLA: PreApplication");
#endif
        next_ = NextOp::bAc;
        locked_ = locked;

        // T *C_host;
        // dla_->retrieveC(&C_host, locked, block, false);
        for (auto j = 0; j < block; j++)
        {
            for (auto i = 0; i < mblocks_; i++)
            {
                std::memcpy(matrices_->C().ptr() + j * m_ + r_offs_l_[i] +
                                locked * m_,
                            V + j * N_ + locked * N_ + r_offs_[i],
                            r_lens_[i] * sizeof(T));
            }
        }

        dla_->preApplication(V, locked, block);
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
    }

    /*!
       - In ChaseMpiDLA, collective communication of `HEMM` operation based on
       MPI which **ALLREDUCE** the product of local matrices either within the
       column communicator or row communicator.
       - The workflow is:
         - compute `B_ = H * C_` (local computation)
         - `Allreduce`(B_, MPI_SUM) (communication within colum communicator)
         - switch operation
         - compute `C_ = H**H * B_` (local computation)
         - `Allreduce`(C_, MPI_SUM) (communication within row communicator)
         - switch operation
         - ...
       - This function implements mainly the collective communications, while
       the local computation is implemented in ChaseMpiDLABlaslapack and
       ChaseMpiDLAMultiGPU, targetting different architectures
       - the computation of local `GEMM` invokes
          - BLAS `GEMM` for pure-CPU distributed-memory ChASE, and it is
       implemented in ChaseMpiDLABlaslapack::apply()
          - cuBLAS `GEMM` for multi-GPU distributed-memory ChASE, and it is
       implemented in ChaseMpiDLAMultiGPU::apply()
   */
    void apply(T alpha, T beta, std::size_t offset, std::size_t block,
               std::size_t locked) override
    {
#ifdef USE_NSIGHT
        nvtxRangePushA("ChaseMpiDLA: apply");
#endif
        T One = T(1.0);
        T Zero = T(0.0);

        std::size_t dim;
        if (next_ == NextOp::bAc)
        {

            dim = n_ * block;
#ifdef USE_NSIGHT
            nvtxRangePushA("ChaseMpiDLA: gemm");
#endif
            dla_->apply(alpha, beta, offset, block, locked);
#ifdef USE_NSIGHT
            nvtxRangePop();
            nvtxRangePushA("ChaseMpiDLA: allreduce");
#endif
            AllReduce(allreduce_backend, B + locked * n_ + offset * n_, dim,
                      getMPI_Type<T>(), MPI_SUM, col_comm_, mpi_wrapper_);
#ifdef USE_NSIGHT
            nvtxRangePop();
#endif
            next_ = NextOp::cAb;
        }
        else
        { // cAb

            dim = m_ * block;
#ifdef USE_NSIGHT
            nvtxRangePushA("ChaseMpiDLA: gemm");
#endif
            dla_->apply(alpha, beta, offset, block, locked);
#ifdef USE_NSIGHT
            nvtxRangePop();
            nvtxRangePushA("ChaseMpiDLA: allreduce");
#endif
            AllReduce(allreduce_backend, C + locked * m_ + offset * m_, dim,
                      getMPI_Type<T>(), MPI_SUM, row_comm_, mpi_wrapper_);
#ifdef USE_NSIGHT
            nvtxRangePop();
#endif
            next_ = NextOp::bAc;
        }
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
    }

    //! collect partially distributed matrices into redundant matrices
    //! @param buff the sending buff
    //! @param targetBuf the receiving buff
    //! @param dimsIdx the collecting direction, `0` indicates collecting
    //! within column communicator, and `1` indicates collecting within
    //! row communicator
    //! @param block number of columns within sending/receiving buffers
    //! to be collected
    void collecRedundantVecs(T* buff, T* targetBuf, std::size_t dimsIdx,
                             std::size_t block)
    {
        MPI_Comm comm;
        if (dimsIdx == 0)
        {
            comm = col_comm_;
        }
        else if (dimsIdx == 1)
        {
            comm = row_comm_;
        }
        int rank;
        MPI_Comm_rank(comm, &rank);

        std::vector<MPI_Request> reqs(dims_[dimsIdx]);

        if (data_layout.compare("Block-Cyclic") == 0)
        {
            if (block > 1 && Buff_.size() != block * N_)
            {
                Buff_.resize(block * N_);
            }
        }

        if (data_layout.compare("Block-Cyclic") == 0)
        {
            for (auto i = 0; i < dims_[dimsIdx]; i++)
            {
                if (rank == i)
                {
                    MPI_Ibcast(buff, send_lens_[dimsIdx][i] * block,
                               getMPI_Type<T>(), i, comm, &reqs[i]);
                }
                else
                {
                    MPI_Ibcast(Buff_.data(), block, newType_[dimsIdx][i], i,
                               comm, &reqs[i]);
                }
            }
        }
        else
        {
            for (auto i = 0; i < dims_[dimsIdx]; i++)
            {
                if (rank == i)
                {
                    MPI_Ibcast(buff, send_lens_[dimsIdx][i] * block,
                               getMPI_Type<T>(), i, comm, &reqs[i]);
                }
                else
                {
                    MPI_Ibcast(targetBuf, block, newType_[dimsIdx][i], i, comm,
                               &reqs[i]);
                }
            }
        }

        int i = rank;

        if (data_layout.compare("Block-Cyclic") == 0)
        {
            for (auto j = 0; j < block; ++j)
            {
                std::memcpy(Buff_.data() + j * N_ +
                                block_cyclic_displs_[dimsIdx][i][0],
                            buff + send_lens_[dimsIdx][i] * j,
                            send_lens_[dimsIdx][i] * sizeof(T));
            }
        }
        else
        {
            for (auto j = 0; j < block; ++j)
            {
                std::memcpy(targetBuf + j * N_ +
                                block_cyclic_displs_[dimsIdx][i][0],
                            buff + send_lens_[dimsIdx][i] * j,
                            send_lens_[dimsIdx][i] * sizeof(T));
            }
        }

        MPI_Waitall(dims_[dimsIdx], reqs.data(), MPI_STATUSES_IGNORE);

        if (data_layout.compare("Block-Cyclic") == 0)
        {
            for (auto j = 0; j < dims_[dimsIdx]; j++)
            {
                for (auto i = 0; i < block_counts_[dimsIdx][j]; ++i)
                {
                    t_lacpy('A', blocklens_[dimsIdx][j][i], block,
                            Buff_.data() + block_cyclic_displs_[dimsIdx][j][i],
                            N_, targetBuf + blockdispls_[dimsIdx][j][i], N_);
                }
            }
        }
    }

    /*!
     * - The objective of this function is to compute `H*C_`, which requires a
     local `GEMM`
     *   at first, then a **ALLREDUCE** operation to summing up the product of
     local matrices
     *  within the column communicator.
     * - Another objective is to re-distribute the data in `C_` to `B_`, in
     which `C_` is partially
     *   distributed in column communicator, and `B_` is partially distributed
     in row communicator.
     * - The reason of combining these two ojectives in one function is to
     overlapping the local
     *   computation and the communication of re-distribution.
     *    - In ChaseMpiDLA, `C2_` and `B2_` are often used to backup (part of)
     `C_` and `B_`.
          - the computation of local `GEMM` using `C_` and `B_`, invokes
              - BLAS `GEMM` for pure-CPU distributed-memory ChASE, and it is
     implemented in ChaseMpiDLABlaslapack::asynCxHGatherC()
              - cuBLAS `GEMM` for multi-GPU distributed-memory ChASE, and it is
     implemented in ChaseMpiDLABlaslapack::asynCxHGatherC()
          - re-distributing from `C2_`, which is distributed within column
     communicator, to `B2_`, which is distributed within row communicator is
     accomplished by multiple MPI asynchronous broadcasting operations, which is
     overlapped with the computation of local `GEMM`.
          - the two operations can be invoked asynchronously with the help of
     **non-Blocking** MPI Bcast.
   */
    void asynCxHGatherC(std::size_t locked, std::size_t block,
                        bool isCcopied = false) override
    {
#ifdef USE_NSIGHT
        nvtxRangePushA("ChaseMpiDLA: asynCxHGatherC");
#endif
        std::size_t dim = n_ * block;

        if (isSameDist_)
        {
            for (auto i = 0; i < col_size_; i++)
            {
                if (row_rank_ == i)
                {
                    if (col_rank_ == i)
                    {
                        Bcast(bcast_backend, C2 + locked * m_, block * m_,
                              getMPI_Type<T>(), i, col_comm_, mpi_wrapper_);
                    }
                    else
                    {
                        Bcast(bcast_backend, B2 + locked * n_, block * n_,
                              getMPI_Type<T>(), i, col_comm_, mpi_wrapper_);
                    }
                }
            }
            for (auto i = 0; i < col_size_; i++)
            {
                if (row_rank_ == col_rank_)
                {
                    dla_->lacpy('A', m_, block, C2 + locked * m_, m_,
                                B2 + locked * n_, n_);
                }
            }
        }
        else
        {
            for(auto i = 0; i < c_lens.size(); i++){
                if (row_rank_ == c_dests[i]){
                    //pack
                    dla_->lacpy('A', c_lens[i], block, C2 + c_disps[i] + locked * send_lens_[0][col_rank_], send_lens_[0][col_rank_],
                                buff__.get()->ptr(), c_lens[i]);

                    Bcast(bcast_backend, buff__.get()->ptr(), c_lens[i] * block, getMPI_Type<T>(), c_srcs[i], col_comm_, mpi_wrapper_);

                    //unpack
                    dla_->lacpy('A', c_lens[i], block, buff__.get()->ptr(), c_lens[i], B2 + b_disps[i] + locked * n_, n_);
                }
            }
        }
            
	dla_->asynCxHGatherC(locked, block, isCcopied);

        AllReduce(allreduce_backend, B + locked * n_, dim, getMPI_Type<T>(),
                   MPI_SUM, col_comm_, mpi_wrapper_);

#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
    }

    /*!
      - For ChaseMpiDLA,  `shiftMatrix` is
          - implemented in nested loop for pure-CPU distributed-memory ChASE,
      and it is implemented in ChaseMpiDLABlaslapack
          - implemented on each GPU for multi-GPU distributed-memory ChASE, and
      it is implemented in ChaseMpiDLAMultiGPU
    */
    void shiftMatrix(T c, bool isunshift = false) override
    {
#ifdef USE_NSIGHT
        nvtxRangePushA("ChaseMpiDLA: shiftMatrix");
#endif
        if (istartOfFilter_)
        {
            next_ = NextOp::bAc;
            T* V;
            dla_->preApplication(V, 0, nev_ + nex_);
        }
        istartOfFilter_ = false;
        dla_->shiftMatrix(c, isunshift);
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
    }

    /*!
      - For ChaseMpiDLA,  `applyVec` is implemented as with the functions
      defined in this class.
      - `applyVec` is used by ChaseMpi::Lanczos(), which requires the input
      arguments `B` and `C` to be vectors of size `N_` which is redundantly
      distributed across all MPI procs.
      - Here are the details:
          - `ChaseMpiDLA::preApplication(B, 0, 1)`
          - `ChaseMpiDLA::apply(One, Zero, 0, 1, 0)`
          - `ChaseMpiDLA::postApplication(C, 1, 0)`
    */
    void applyVec(T* v, T* v2, std::size_t n) override
    {}

    bool checkSymmetryEasy() override 
    {
        std::vector<T> v(m_, T(0.0));
        std::vector<T> u(n_, T(0.0));
        std::vector<T> uT(m_, T(0.0));
        std::vector<T> v_2(n_, T(0.0));

        int mpi_col_rank;
        MPI_Comm_rank(col_comm_, &mpi_col_rank);

        std::mt19937 gen(1337.0 + mpi_col_rank);
        std::normal_distribution<> d;
        
        for (auto i = 0; i < m_; i++)
        {
            v[i] = getRandomT<T>([&]() { return d(gen); });
        }

        this->C2B(v.data(), 0, v_2.data(), 0, 1);
        
        T One = T(1.0);
        T Zero = T(0.0);

        t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, //
               n_, 1, m_,                                 //
               &One,                                      //
               matrices_->H().host(), matrices_->H().h_ld(),                                  //
               v.data(), m_,                             //
               &Zero,                                     //
               u.data(), n_);

        MPI_Allreduce(MPI_IN_PLACE, u.data(), n_,
                      getMPI_Type<T>(), MPI_SUM, col_comm_);  

        t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, //
               m_, 1, n_,                                 //
               &One,                                      //
               matrices_->H().host(), matrices_->H().h_ld(),                                  //
               v_2.data(), n_,                             //
               &Zero,                                     //
               uT.data(), m_);

        MPI_Allreduce(MPI_IN_PLACE, uT.data(), m_,
                      getMPI_Type<T>(), MPI_SUM, row_comm_);  

        this->B2C(u.data(), 0, v.data(), 0, 1);

        bool is_sym = true;

        for(auto i = 0; i < m_; i++)
        {
            //std::cout << "std::abs(v[i] - uT[i]) = " << std::abs(v[i] - uT[i]) << "\n";
            //if(std::abs(v[i] - uT[i]) > std::numeric_limits<Base<T>>::epsilon())
            if(std::abs(v[i] - uT[i]) > 1e-10)
            {
                 is_sym = false;
                break;
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, &is_sym, 1, MPI_CXX_BOOL, MPI_LAND, col_comm_);

        return is_sym;
    }

    void symOrHermMatrix(char uplo) override
    {
#if defined(HAS_SCALAPACK)    
        auto ctxt = matrix_properties_->get_comm2D_ctxt();
        std::size_t desc_H[9];
        int info;
        int zero = 0;
        int one = 1;
        auto ldh = matrices_->H().h_ld();
        auto H = matrices_->H().host();

        t_descinit(desc_H, &N_, &N_, &mb_, &nb_, &zero, &zero, &ctxt, &ldh, &info);  

        if(uplo == 'U')
        {
            for(std::size_t j = 0; j < nblocks_; j++){
                for(std::size_t i = 0; i < mblocks_; i++){
                    for(std::size_t q = 0; q < c_lens_[j]; q++){
                        for(std::size_t p = 0; p < r_lens_[i]; p++){
                            if(q + c_offs_[j] == p + r_offs_[i]){
                                H[(q + c_offs_l_[j]) * ldh + p + r_offs_l_[i]] *= T(0.5);
                            }
                            else if(q + c_offs_[j] < p + r_offs_[i])
                            {
                                H[(q + c_offs_l_[j]) * ldh + p + r_offs_l_[i]] = T(0.0);
                            }
                        }
                    }
                }
            }

        }else
        {
            for(std::size_t j = 0; j < nblocks_; j++){
                for(std::size_t i = 0; i < mblocks_; i++){
                    for(std::size_t q = 0; q < c_lens_[j]; q++){
                        for(std::size_t p = 0; p < r_lens_[i]; p++){
                            if(q + c_offs_[j] == p + r_offs_[i]){
                                H[(q + c_offs_l_[j]) * ldh + p + r_offs_l_[i]] *= T(0.5);
                            }
                            else if(q + c_offs_[j] > p + r_offs_[i])
                            {
                                H[(q + c_offs_l_[j]) * ldh + p + r_offs_l_[i]] = T(0.0);
                            }
                        }
                    }
                }
            }
        }
        T One = T(1.0);
        T Zero = T(0.0);
        std::vector<T> tmp(m_ * (n_+2*nb_));
        t_ptranc(N_, N_, One, H, one, one, desc_H, Zero, tmp.data(), one, one, desc_H);

        for(auto i = 0; i < m_; i++)
        {
            for(auto j = 0; j < n_; j++)
            {
                H[i + j * ldh] += tmp[i + j * m_];
            }
        }
        
    #else
        std::cout << "!!! symmetrizeOrHermitianizeMatrix failed, it requires ScaLAPACK, which is not detected\n";
        return;
    #endif        
    }

    int get_nprocs() const override { return matrix_properties_->get_nprocs(); }
    void Start() override { dla_->Start(); }
    void End() override { dla_->End(); }
    Base<T>* get_Resids() override { return dla_->get_Resids(); }
    Base<T>* get_Ritzv() override { return dla_->get_Ritzv(); }

    void axpy(std::size_t N, T* alpha, T* x, std::size_t incx, T* y,
              std::size_t incy) override
    {
        dla_->axpy(N, alpha, x, incx, y, incy);
    }

    /*!
      - For ChaseMpiDLA, `scal` is implemented by calling the one in
      ChaseMpiDLABlaslapack and ChaseMpiDLAMultiGPU.
      - This implementation is the same for both with or w/o GPUs.
      - **Parallelism is SUPPORT within node if multi-threading is actived**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    void scal(std::size_t N, T* a, T* x, std::size_t incx) override
    {
        dla_->scal(N, a, x, incx);
    }

    Base<T> nrm2(std::size_t n, T* x, std::size_t incx) override
    {
        return dla_->nrm2(n, x, incx);
    }

    T dot(std::size_t n, T* x, std::size_t incx, T* y,
          std::size_t incy) override
    {
        return dla_->dot(n, x, incx, y, incy);
    }

    /*! Implementation of Rayleigh-Ritz (RR) in ChASE
     * - The workflow of RR is
     *     - compute `B_= H*C_` and re-distribute from `C2_` to `B2_` (by
     asynCxHGatherC in this class)
     *     - compute `A_ = B2_**H*B_` (local `GEMM`)
     *     - `allreduce`(A_, MPI_SUM) (within row communicator)
     *     - `(syhe)evd` to compute all eigenpairs of `A_`
     *     - `gemm`: `C_=C2_*A_` (local computation)
       - In ChaseMpiDLA, this function implements mainly the collective
     communications, while the local computation (`sy(he)rk`, `(syhe)evd`,
     `trsm`) is implemented in ChaseMpiDLABlaslapack and ChaseMpiDLAMultiGPU,
     targetting different architectures

       - the local computation of local `GEMM` invokes
          - BLAS/LAPACK `gemm` and `(syhe)evd` for pure-CPU distributed-memory
     ChASE,
            - the operation `A_ = B2_**H*B_` is implemented in
     ChaseMpiDLABlaslapack::syherk
            - the operations `(syhe)evd` and `C_=C2_*A` is implemented in
              ChaseMpiDLABlaslapack::heevd, respectively
          - cuBLAS/cuSOLVER `gemm` and `(syhe)evd` for multi-GPU
     distributed-memory ChASE,
            - the operation `A_ = B2_**H*B_` is implemented in
     ChaseMpiDLAMultiGPU::syherk
            - the operations `(syhe)evd` and `C_=C2_*A` is implemented in
              ChaseMpiDLAMultiGPU::heevd, respectively
   */
    void RR(std::size_t block, std::size_t locked, Base<T>* ritzv) override
    {
        T One = T(1.0);
        T Zero = T(0.0);
        this->asynCxHGatherC(locked, block, !isHHqr);
#ifdef USE_NSIGHT
        nvtxRangePushA("ChaseMpiDLA: RR");
#endif
        dla_->RR(block, locked, ritzv);
#ifdef USE_NSIGHT
        nvtxRangePop();
        nvtxRangePushA("allreduce");
#endif
        AllReduce(allreduce_backend, A, (nev_ + nex_) * block, getMPI_Type<T>(),
                  MPI_SUM, row_comm_, mpi_wrapper_);

#ifdef USE_NSIGHT
        nvtxRangePop();
        nvtxRangePushA("ChaseMpiDLA: heevd");
#endif
        dla_->heevd(LAPACK_COL_MAJOR, 'V', 'L', block, A, nev_ + nex_, ritzv);
#ifdef USE_NSIGHT
        nvtxRangePop();
        nvtxRangePushA("memcpy");
#endif
        Memcpy(memcpy_mode[0], C2 + locked * m_, C + locked * m_,
               m_ * block * sizeof(T));
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
    }

    void Resd(Base<T>* ritzv, Base<T>* resid, std::size_t locked,
              std::size_t unconverged) override
    {

        this->asynCxHGatherC(locked, unconverged, true);

        T one = T(1.0);
        T neg_one = T(-1.0);
        T beta = T(0.0);
#ifdef USE_NSIGHT
        nvtxRangePushA("ChaseMpiDLA: Resd");
#endif
        dla_->Resd(ritzv, resid, locked, unconverged);
#ifdef USE_NSIGHT
        nvtxRangePop();
        nvtxRangePushA("allreduce");
#endif

        AllReduce(allreduce_backend, rsd + locked, unconverged,
                  getMPI_Type<Base<T>>(), MPI_SUM, row_comm_, mpi_wrapper_);
        //  Base<T> *resid_h;
        // dla_->retrieveResid(&resid_h, locked, unconverged);
        if (rsd != matrices_->Resid().ptr())
        {
            //	std::cout << "rsd != Resid().ptr()" << std::endl;
            matrices_->Resid().sync2Ptr(1, unconverged, locked);
        }
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif

        for (std::size_t i = 0; i < unconverged; ++i)
        {
            //    resid[i] = std::sqrt(resid_h[i]);
            resid[i] = std::sqrt(matrices_->Resid().ptr()[i + locked]);
        }
    }

    void syherk(char uplo, char trans, std::size_t n, std::size_t k, T* alpha,
                T* a, std::size_t lda, T* beta, T* c, std::size_t ldc,
                bool first = true) override
    {
        dla_->syherk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
    /*!
        In ChaseMpiDLA, the computation of Cholesky factorization can be
            - LAPACK `xpotrf` for pure-CPU distributed-memory ChASE,
                and it is implemented in ChaseMpiDLABlaslapack
            - cuSOLVER zcusolverDnXpotrfz for multi-GPU distributed-memory
       ChASE, and it is implemented in ChaseMpiDLABlaslapack
    */
    int potrf(char uplo, std::size_t n, T* a, std::size_t lda, bool isinfo = true) override    
    {
        return dla_->potrf(uplo, n, a, lda);
    }

    void trsm(char side, char uplo, char trans, char diag, std::size_t m,
              std::size_t n, T* alpha, T* a, std::size_t lda, T* b,
              std::size_t ldb, bool first = false) override
    {
        dla_->trsm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
    }

    void heevd(int matrix_layout, char jobz, char uplo, std::size_t n, T* a,
               std::size_t lda, Base<T>* w) override
    {

        dla_->heevd(matrix_layout, jobz, uplo, n, a, lda, w);
    }

    void hhQR(std::size_t locked) override
    {
#ifdef USE_NSIGHT
        nvtxRangePop();
        nvtxRangePushA("ChaseMpiDLA: hhQR");
#endif
        auto nevex = nev_ + nex_;
        std::unique_ptr<T[]> tau(new T[nevex]);
#if defined(HAS_SCALAPACK)
        int one = 1;
#ifdef USE_NSIGHT
        nvtxRangePushA("pgeqrf+pgqr");
#endif
        if (C != matrices_->C().ptr())
        {
            matrices_->C().sync2Ptr();
        }
        t_pgeqrf(N_, nevex, matrices_->C().ptr(), one, one, desc1D_Nxnevx_,
                 tau.get());
        t_pgqr(N_, nevex, nevex, matrices_->C().ptr(), one, one, desc1D_Nxnevx_,
               tau.get());
        if (C != matrices_->C().ptr())
        {
            matrices_->C().syncFromPtr();
        }
#ifdef USE_NSIGHT
        nvtxRangePop();
        nvtxRangePushA("memcpy");
#endif

#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
#else
        int grank;
        MPI_Comm_rank(MPI_COMM_WORLD, &grank);
        if (grank == 0)
            std::cout << "ScaLAPACK is not available, use LAPACK Householder "
                         "QR instead"
                      << std::endl;
        if (C != matrices_->C().ptr())
        {
            matrices_->C().sync2Ptr();
        }

        if (!alloc_)
        {
            V___ = std::make_unique<Matrix<T>>(0, N_, nevex);
            alloc_ = true;
        }

        this->collecRedundantVecs(matrices_->C().ptr(), V___.get()->ptr(), 0,
                                  nevex);
        t_geqrf(LAPACK_COL_MAJOR, N_, nevex, V___.get()->ptr(), N_, tau.get());
        t_gqr(LAPACK_COL_MAJOR, N_, nevex, nevex, V___.get()->ptr(), N_,
              tau.get());
        this->preApplication(V___.get()->ptr(), 0, nevex);

        isHHqr = true;

#endif
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
    }

    /*! Implementation of partially 1D distributed Cholesky QR within each
     column communicator.
     * - The workflow of a 1D Cholesky QR is
     *     - `sy(he)rk`: `A_ = C_*C_` (local computation)
     *     - `allreduce`(A, MPI_SUM) (within column communicator)
     *     - `potrf(A)` (local computation, redundantly within column
     communicator)
     *     - `trsm`: `C_=A*C_` (local computation)
     *     - repeat previous step to impprove the accuracy of QR factorization
       - In ChaseMpiDLA, this function implements mainly the collective
     communications, while the local computation (`sy(he)rk`, `potrf`, `trsm`)
     is implemented in ChaseMpiDLABlaslapack and ChaseMpiDLAMultiGPU, targetting
     different architectures

       - the local computation invokes
          - BLAS/LAPACK `sy(he)rk`, `potrf`, `trsm` for pure-CPU
     distributed-memory ChASE, and it is implemented in
     ChaseMpiDLABlaslapack::syherk, ChaseMpiDLABlaslapack::potrf and
            ChaseMpiDLABlaslapack::trsm, respectively
          - cuBLAS/cuSOLVER `sy(he)rk`, `potrf`, `trsm` for multi-GPU
     distributed-memory ChASE, and it is implemented in
     ChaseMpiDLAMultiGPU::syherk, ChaseMpiDLAMultiGPU::potrf and
            ChaseMpiDLAMultiGPU::trsm, respectively
   */

    int cholQR1(std::size_t locked) override
    {
        isHHqr = false;

        int grank;
        MPI_Comm_rank(MPI_COMM_WORLD, &grank);

        auto nevex = nev_ + nex_;
        bool first_iter = !cuda_aware_;        
        T one = T(1.0);
        T zero = T(0.0);
        int info = 1;

        dla_->syherk('U', 'C', nevex, m_, &one, C, m_, &zero, A, nevex,
                     first_iter);
        AllReduce(allreduce_backend, A, nevex * nevex, getMPI_Type<T>(),
                  MPI_SUM, col_comm_, mpi_wrapper_);
        info = dla_->potrf('U', nevex, A, nevex, true); 

        if(info != 0)
        {
            return info;
        }else
        {
            dla_->trsm('R', 'U', 'N', 'N', m_, nevex, &one, A, nevex, C, m_,
                    false);
#ifdef CHASE_OUTPUT
            if (grank == 0)
            {
                std::cout << std::setprecision(2) << "choldegree: 1" << std::endl;
            }
#endif                    
            return info;  
        }                 
    }

    int cholQR2(std::size_t locked) override
    {
        isHHqr = false;

        int grank;
        MPI_Comm_rank(MPI_COMM_WORLD, &grank);

        auto nevex = nev_ + nex_; 
        bool first_iter = !cuda_aware_;        
        T one = T(1.0);
        T zero = T(0.0);
        int info = 1;

        dla_->syherk('U', 'C', nevex, m_, &one, C, m_, &zero, A, nevex,
                     first_iter);

        AllReduce(allreduce_backend, A, nevex * nevex, getMPI_Type<T>(),
                  MPI_SUM, col_comm_, mpi_wrapper_);
        info = dla_->potrf('U', nevex, A, nevex, true); 

        if(info != 0)
        {
            return info;
        }else
        {
            dla_->trsm('R', 'U', 'N', 'N', m_, nevex, &one, A, nevex, C, m_,
                    true);

            dla_->syherk('U', 'C', nevex, m_, &one, C, m_, &zero, A, nevex,
                    false);                    

            AllReduce(allreduce_backend, A, nevex * nevex, getMPI_Type<T>(),
                  MPI_SUM, col_comm_, mpi_wrapper_);
            
            info = dla_->potrf('U', nevex, A, nevex, false); 

            dla_->trsm('R', 'U', 'N', 'N', m_, nevex, &one, A, nevex, C, m_,
                    false);
#ifdef CHASE_OUTPUT
            if (grank == 0)
            {
                std::cout << std::setprecision(2) << "choldegree: 2" << std::endl;
            }
#endif 
            return info;  
        }
    }    

    int shiftedcholQR2(std::size_t locked) override
    {
        isHHqr = false;
    
        int grank;
        MPI_Comm_rank(MPI_COMM_WORLD, &grank);

        Base<T> shift;
        auto nevex = nev_ + nex_; 
        bool first_iter = !cuda_aware_;        
        T one = T(1.0);
        T zero = T(0.0);
        int info = 1;

        dla_->syherk('U', 'C', nevex, m_, &one, C, m_, &zero, A, nevex,
                     first_iter);

        AllReduce(allreduce_backend, A, nevex * nevex, getMPI_Type<T>(),
                  MPI_SUM, col_comm_, mpi_wrapper_);
        
        Base<T> nrmf = 0.0;
        dla_->computeDiagonalAbsSum(A, &nrmf, nevex, nevex);
        //Base<T> nrmf = dla_->nrm2(m_ * nevex, C, 1);
        //nrmf = std::pow(nrmf, 2);
        //MPI_Allreduce(MPI_IN_PLACE, &nrmf, 1, getMPI_Type<Base<T>>(),
        //                MPI_SUM, col_comm_);

        shift = std::sqrt(N_) * nrmf * std::numeric_limits<Base<T>>::epsilon();

        dla_->shiftMatrixForQR(A, nevex, (T)shift);
        
        info = dla_->potrf('U', nevex, A, nevex, true);        
	
        if(info != 0)
	    {
	        return info;
	    }

        
        dla_->trsm('R', 'U', 'N', 'N', m_, nevex, &one, A, nevex, C, m_,
                    true);
            
        dla_->syherk('U', 'C', nevex, m_, &one, C, m_, &zero, A, nevex,
                    false);

        AllReduce(allreduce_backend, A, nevex * nevex, getMPI_Type<T>(),
                MPI_SUM, col_comm_, mpi_wrapper_);
        
        //check info after this step
        info = dla_->potrf('U', nevex, A, nevex, true);
        if(info != 0)
	{
	    return info;
	}

        dla_->trsm('R', 'U', 'N', 'N', m_, nevex, &one, A, nevex, C, m_,
                    true);

        dla_->syherk('U', 'C', nevex, m_, &one, C, m_, &zero, A, nevex,
                    false);

        AllReduce(allreduce_backend, A, nevex * nevex, getMPI_Type<T>(),
                MPI_SUM, col_comm_, mpi_wrapper_);
    
        dla_->potrf('U', nevex, A, nevex, false); 

        dla_->trsm('R', 'U', 'N', 'N', m_, nevex, &one, A, nevex, C, m_,
                    false);

#ifdef CHASE_OUTPUT
        if (grank == 0)
        {
            std::cout << std::setprecision(2) << "choldegree: 2, shift = " << shift << std::endl;
        }
#endif 

        return info;
    }

    void estimated_cond_evaluator(std::size_t locked, Base<T> cond)
    {
        int grank;
        MPI_Comm_rank(MPI_COMM_WORLD, &grank);
        auto nevex = nev_ + nex_;

        std::vector<T> V2(N_ * (nev_ + nex_));
        if (C != matrices_->C().ptr())
        {
            matrices_->C().sync2Ptr();
        }
            
        this->collecRedundantVecs(matrices_->C().ptr(), V2.data(), 0,
                                    nev_ + nex_);
        std::vector<Base<T>> S(nev_ + nex_ - locked);
        T* U;
        std::size_t ld = 1;
        T* Vt;
        t_gesvd('N', 'N', N_, nev_ + nex_ - locked, V2.data() + N_ * locked,
                N_, S.data(), U, ld, Vt, ld);
        std::vector<Base<T>> norms(nev_ + nex_ - locked);
        for (auto i = 0; i < nev_ + nex_ - locked; i++)
        {
            norms[i] = std::sqrt(t_sqrt_norm(S[i]));
        }
        std::sort(norms.begin(), norms.end());
        if (grank == 0)
        {
            std::cout << "estimate: " << cond << ", rcond: "
                      << norms[nev_ + nex_ - locked - 1] / norms[0]
                      << ", ratio: "
                      << cond * norms[0] / norms[nev_ + nex_ - locked - 1]
                      << std::endl;
        }
    }

    void lockVectorCopyAndOrthoConcatswap(std::size_t locked, bool isHHqr)
    {
        Memcpy(memcpy_mode[0], C, C2, locked * m_ * sizeof(T));

        if(isHHqr)
        {
            Memcpy(memcpy_mode[0], C2 + locked * m_, C + locked * m_,
                   (nev_ + nex_ - locked) * m_ * sizeof(T));
        }
        else
        {
            Memcpy(memcpy_mode[1], C2 + locked * m_, C + locked * m_,
                   (nev_ + nex_ - locked) * m_ * sizeof(T));
        }
    }

    void Swap(std::size_t i, std::size_t j) override
    {
        Memcpy(memcpy_mode[0], vv, C + m_ * i, m_ * sizeof(T));
        Memcpy(memcpy_mode[0], C + m_ * i, C + m_ * j, m_ * sizeof(T));
        Memcpy(memcpy_mode[0], C + m_ * j, vv, m_ * sizeof(T));

        Memcpy(memcpy_mode[0], vv, C + m_ * i, m_ * sizeof(T));
        Memcpy(memcpy_mode[0], C + m_ * i, C + m_ * j, m_ * sizeof(T));
        Memcpy(memcpy_mode[0], C + m_ * j, vv, m_ * sizeof(T));
    }

    void LanczosDos(std::size_t idx, std::size_t m, T* ritzVc) override
    {
        dla_->LanczosDos(idx, m, ritzVc);
    }

    void mLanczos(std::size_t M, int numvec, Base<T>* d, Base<T>* e,
                 Base<T>* r_beta) override
    {
        bool is_second_system = false;

        if(numvec == -1)
        {
            numvec = 1;
            is_second_system = true;
        }
        std::vector<Base<T>> real_alpha(numvec);
        std::vector<T> alpha(numvec, T(1.0));
        std::vector<T> beta(numvec, T(0.0));
    
	int matrix_mode = matrices_->get_Mode();
	if(matrices_->get_Mode() == 2)
	{
	    matrix_mode = 1;
	}	
        v_0 = new Matrix<T>(matrix_mode, m_, numvec);
        v_1 = new Matrix<T>(matrix_mode, m_, numvec);
        v_2 = new Matrix<T>(matrix_mode, m_, numvec);
        v_w = new Matrix<T>(matrix_mode, n_, numvec);

        //Memcpy(memcpy_mode[1], v_1->ptr(), C, m_ * numvec * sizeof(T));
        Memcpy(memcpy_mode[1], v_1->ptr(), C, m_ * numvec * sizeof(T));

        dla_->nrm2_batch(m_, v_1, 1, numvec, real_alpha.data());

        for(auto i = 0; i < numvec; i++)
        {
            real_alpha[i] = std::pow(real_alpha[i], 2);
        }

        MPI_Allreduce(MPI_IN_PLACE, real_alpha.data(), numvec, getMPI_Type<Base<T>>(),
                      MPI_SUM, col_comm_);

        for(auto i = 0; i < numvec; i++)
        {
            real_alpha[i] = std::sqrt(real_alpha[i]);
            alpha[i] = T(1 / real_alpha[i]);
        }

        dla_->scal_batch(m_, alpha.data(), v_1, 1, numvec);

        for (std::size_t k = 0; k < M; k = k + 1)
        {
            if(!is_second_system)
            {
                for(auto i = 0; i < numvec; i++){
                    Memcpy(memcpy_mode[2], C + k * m_, v_1->ptr() + i * m_, m_ * sizeof(T));
                }
            }

            //dla_->applyVec(v_1->ptr(), v_w->ptr(), numvec);
            dla_->applyVec(v_1, v_w, numvec);
	    MPI_Allreduce(MPI_IN_PLACE, v_w->ptr(), n_ * numvec, getMPI_Type<T>(), MPI_SUM,
                      col_comm_);            
            //AllReduce(allreduce_backend, v_w->ptr(), n_ * numvec,
            //          getMPI_Type<T>(), MPI_SUM, col_comm_, mpi_wrapper_);
            //this->B2C(v_w->ptr(), 0, v_2->ptr(), 0, numvec);   
            this->B2C(v_w->ptr(), 0, v_2->ptr(), 0, numvec);   

            dla_->dot_batch(m_, v_1, 1, v_2, 1, alpha.data(), numvec);
            for(auto i = 0; i < numvec; i++)
            {
                alpha[i] = -alpha[i];
            }

            MPI_Allreduce(MPI_IN_PLACE, alpha.data(), numvec, getMPI_Type<T>(), MPI_SUM,
                          col_comm_);

            dla_->axpy_batch(m_, alpha.data(), v_1, 1, v_2, 1, numvec);
            for(auto i = 0; i < numvec; i++)
            {
                alpha[i] = -alpha[i];
            }

            for(auto i = 0; i < numvec; i++)
            {
                d[k + M * i] = std::real(alpha[i]);
            }

            if(k > 0){
                for(auto i = 0; i < numvec; i++)
                {
                    beta[i] = T(-r_beta[i]);
                }
                dla_->axpy_batch(m_, beta.data(), v_0, 1, v_2, 1, numvec);
            }

            for(auto i = 0; i < numvec; i++)
            {
                beta[i] = -beta[i];
            }

            dla_->nrm2_batch(m_, v_2, 1, numvec, r_beta);

            for(auto i = 0; i < numvec; i++)
            {
                r_beta[i] = std::pow(r_beta[i], 2);
            }

            MPI_Allreduce(MPI_IN_PLACE, r_beta, numvec, getMPI_Type<Base<T>>(),
                          MPI_SUM, col_comm_);

            for(auto i = 0; i < numvec; i++)
            {
                r_beta[i] = std::sqrt(r_beta[i]);
                beta[i] = T(1 / r_beta[i]);
            }

            if (k == M - 1)
                break;
            
            dla_->scal_batch(m_, beta.data(), v_2, 1, numvec);

            for(auto i = 0; i < numvec; i++)
            {
                e[k + M * i] = r_beta[i];
            }
            v_1->swap(*v_0);
            v_1->swap(*v_2);
        }

        if(!is_second_system)
        {
            Memcpy(memcpy_mode[2], C, v_1->ptr(), m_ * numvec * sizeof(T));  
        }
    }

    void B2C(T* B, std::size_t off1, T* C, std::size_t off2,
             std::size_t block) override
    {
        for (auto i = 0; i < b_lens.size(); i++)
        {
            if (col_rank_ == b_dests[i])
            {
                if (row_rank_ == b_srcs[i])
                {
                    MPI_Ibcast(B + off1 * n_, block, b_sends_[i], b_srcs[i],
                               row_comm_, &reqsb2c_[i]);
                }
                else
                {
                    MPI_Ibcast(C + off1 * m_, block, c_recvs_[i], b_srcs[i],
                               row_comm_, &reqsb2c_[i]);
                }
            }
        }

        for (auto i = 0; i < b_lens.size(); i++)
        {
            if (col_rank_ == b_dests[i])
            {
                MPI_Wait(&reqsb2c_[i], MPI_STATUSES_IGNORE);
            }
        }

        for (auto i = 0; i < b_lens.size(); i++)
        {
            if (col_rank_ == b_dests[i] && row_rank_ == b_srcs[i])
            {
                t_lacpy('A', b_lens[i], block, B + off1 * n_ + b_disps_2[i], n_,
                        C + off1 * m_ + c_disps_2[i], m_);
            }
        }        
    
    }

    void B2C(Matrix<T>* B, std::size_t off1, Matrix<T>* C, std::size_t off2,
                        std::size_t block) override
    {
        if (isSameDist_)
        {
            for (auto i = 0; i < row_size_; i++)
            {
                if (col_rank_ == i)
                {
                    if (row_rank_ == i)
                    {
                        Bcast(bcast_backend, B->ptr() + off1 * n_, block * n_,
                              getMPI_Type<T>(), i, row_comm_, mpi_wrapper_);

                    }else
                    {
                        Bcast(bcast_backend, C->ptr() + off1 * m_, block * m_,
                              getMPI_Type<T>(), i, row_comm_, mpi_wrapper_);

                    }    
                }
            }

            for (auto i = 0; i < row_size_; i++)
            {
                if (row_rank_ == col_rank_)
                {
                    dla_->lacpy('A', n_, block, B->ptr() + off1 * n_, n_,
                                C->ptr() + off1 * m_, m_);
                }
            }
        }else
        {
            for(auto i = 0; i < b_lens.size(); i++){
                if (col_rank_ == b_dests[i]){
                    //pack
                    dla_->lacpy('A', b_lens[i], block, B->ptr() + b_disps[i] + off1 * send_lens_[1][row_rank_], send_lens_[1][row_rank_],
                                buff__.get()->ptr(), b_lens[i]);

                    Bcast(bcast_backend, buff__.get()->ptr(), b_lens[i] * block, getMPI_Type<T>(), b_srcs[i], row_comm_, mpi_wrapper_);

                    //unpack
                    dla_->lacpy('A', b_lens[i], block, buff__.get()->ptr(), b_lens[i], C->ptr() + c_disps[i] + off1 * m_, m_);
                }
            }
        }           
    }

    void C2B(T* C, std::size_t off1, T* B, std::size_t off2,
             std::size_t block)
    {
        for (auto i = 0; i < c_lens.size(); i++)
        {
            if (row_rank_ == c_dests[i])
            {
                if (col_rank_ == c_srcs[i])
                {
                    MPI_Ibcast(C + off1 * m_, block, c_sends_[i], c_srcs[i],
                               col_comm_, &reqsc2b_[i]);
                }
                else
                {
                    MPI_Ibcast(B + off1 * n_, block, b_recvs_[i], c_srcs[i],
                               col_comm_, &reqsc2b_[i]);
                }
            }
        }

        for (auto i = 0; i < c_lens.size(); i++)
        {
            if (row_rank_ == c_dests[i])
            {
                MPI_Wait(&reqsc2b_[i], MPI_STATUSES_IGNORE);
            }
        }

        for (auto i = 0; i < c_lens.size(); i++)
        {
            if (row_rank_ == c_dests[i] && col_rank_ == c_srcs[i])
            {
                t_lacpy('A', c_lens[i], block, C + off1 * m_ + c_disps_2[i], m_,
                        B + off1 * n_ + b_disps_2[i], n_);
            }
        }
        
    }

    void lacpy(char uplo, std::size_t m, std::size_t n, T* a, std::size_t lda,
               T* b, std::size_t ldb) override
    {
    }

    void shiftMatrixForQR(T* A, std::size_t n, T shift) override {}
    void computeDiagonalAbsSum(T *A, Base<T> *sum, std::size_t n, std::size_t ld){}
    ChaseMpiMatrices<T>* getChaseMatrices() override { return matrices_; }

    void nrm2_batch(std::size_t n, Matrix<T>* x, std::size_t incx, int count, Base<T> *nrms) override
    {}
    void scal_batch(std::size_t N, T* a, Matrix<T>* x, std::size_t incx, int count) override
    {}  
    void applyVec(Matrix<T>* v, Matrix<T>* w, std::size_t n) override
    {}  
    void dot_batch(std::size_t n, Matrix<T>* x, std::size_t incx, Matrix<T>* y,
          std::size_t incy, T *products, int count) override
    {}
    void axpy_batch(std::size_t N, T* alpha, Matrix<T>* x, std::size_t incx, Matrix<T>* y,
              std::size_t incy, int count) override
    {}

private:
    enum NextOp
    {
        cAb,
        bAc
    };

    std::size_t locked_; //!< number of converged eigenpairs, it is synchronized
                         //!< with ChaseMpi::locked_
    std::size_t ldc_;    //!< leading dimension of `C_` and `C2_`
    std::size_t ldb_;    //!< leading dimension of `B_` and `B2_`
    std::size_t n_;      //!< number of columns of local matrix of the
                         //!< symmetric/Hermtian matrix
    std::size_t
        m_; //!< number of rows of local matrix of the symmetric/Hermtian matrix
    std::size_t N_; //!< global dimension of the symmetric/Hermtian matrix

    std::vector<T> Buff_; //!< a vector of size `N_`, it is allocated only ChASE
                          //!< working with `Block-Cyclic`

    NextOp next_; //!< it is to manage the switch of operation from `V2=H*V1` to
                  //!< `V1=H'*V2` in filter
    MPI_Comm row_comm_; //!< row communicator of 2D MPI proc grid, which is
                        //!< setup in ChaseMpiProperties
    MPI_Comm col_comm_; //!< row communicator of 2D MPI proc grid, which is
                        //!< setup in ChaseMpiProperties
    int* dims_;         //!< dimension of 2D MPI proc grid, which is setup in
                        //!< ChaseMpiProperties
    int* coord_; //!< coordinates of each MPI rank within 2D MPI proc grid,
                 //!< which is setup in ChaseMpiProperties
    std::size_t* off_;      //!< identical to ChaseMpiProperties::off_
    std::size_t* r_offs_;   //!< identical to ChaseMpiProperties::r_offs_
    std::size_t* r_lens_;   //!< identical to ChaseMpiProperties::r_lens_
    std::size_t* r_offs_l_; //!< identical to ChaseMpiProperties::r_offs_l_
    std::size_t* c_offs_;   //!< identical to ChaseMpiProperties::c_offs_
    std::size_t* c_lens_;   //!< identical to ChaseMpiProperties::c_lens_
    std::size_t* c_offs_l_; //!< identical to ChaseMpiProperties::c_offs_l_
    std::size_t nb_;        //!< identical to ChaseMpiProperties::nb_
    std::size_t mb_;        //!< identical to ChaseMpiProperties::mb_
    std::size_t nblocks_;   //!< identical to ChaseMpiProperties::nblocks_
    std::size_t mblocks_;   //!< identical to ChaseMpiProperties::mblocks_
    std::size_t nev_;       //!< number of required eigenpairs
    std::size_t nex_;       //!< number of extral searching space

    std::vector<std::vector<int>>
        send_lens_; //!< identical to ChaseMpiProperties::send_lens_
    std::vector<std::vector<int>>
        block_counts_; //!< identical to ChaseMpiProperties::block_counts_
    std::vector<std::vector<int>>
        g_offset_; //!< identical to ChaseMpiProperties::g_offsets_
    std::vector<std::vector<std::vector<int>>>
        blocklens_; //!< identical to ChaseMpiProperties::blocklens_
    std::vector<std::vector<std::vector<int>>>
        blockdispls_; //!< identical to ChaseMpiProperties::blockdispls_

    bool isSameDist_; //!< a flag indicating if the row and column communicator
                      //!< has the same distribution scheme
    bool istartOfFilter_; //!< a flag indicating if it is the starting pointer
                          //!< of apply Chebyshev filter
    std::vector<MPI_Request> reqsc2b_; //!< a collection of MPI requests for
                                       //!< asynchonous communication
    std::vector<MPI_Datatype>
        c_sends_; //!< a collection of MPI new datatype for sending operations
    std::vector<MPI_Datatype>
        b_recvs_; //!< a collection of MPI new datatype for receiving operations
    std::vector<MPI_Datatype> newType_[2]; //!< a collection of MPI new datatype
                                           //!< for collective communication
    std::vector<int>
        c_dests; //!< destination for each continous part of `C_` which will
                 //!< send to `B_` within column communicator
    std::vector<int> c_srcs;  //!< source for each continous part of `C_` which
                              //!< will send to `B_` within column communicator
    std::vector<int> c_lens;  //!< length of each continous part of `C_` which
                              //!< will send to `B_` within column communicator
    std::vector<int> b_disps; //!< displacement of row indices within `B_` for
                              //!< receiving each continous buffer from `C_`
    std::vector<int> c_disps; //!< displacement of row indices within `C_` for
                              //!< sending each continous buffer to `B_`
    std::vector<std::vector<int>>
        block_cyclic_displs_[2]; //!< dispacement (row/col_comm) of each block
                                 //!< of block-cyclic distribution within local
                                 //!< matrix
    std::vector<int> b_dests;
    std::vector<int> b_srcs;
    std::vector<int> b_lens;
    std::vector<MPI_Request> reqsb2c_;
    std::vector<MPI_Datatype> b_sends_;
    std::vector<MPI_Datatype> c_recvs_;
    std::vector<int> b_disps_2;
    std::vector<int> c_disps_2;

    int icsrc_;              //!< identical to ChaseMpiProperties::icsrc_
    int irsrc_;              //!< identical to ChaseMpiProperties::irsrc_
    int row_size_;           //!< row communicator size
    int row_rank_;           //!< rank within each row communicator
    int col_size_;           //!< column communicator size
    int col_rank_;           //!< rank within each column communicator
    std::string data_layout; //!< identical to ChaseMpiProperties::data_layout
    std::unique_ptr<ChaseMpiDLAInterface<T>>
        dla_; //!< an object of class ChaseMpiDLABlaslapack or
              //!< ChaseMpiDLAMultiGPU
    ChaseMpiProperties<T>*
        matrix_properties_; //!< an object of class ChaseMpiProperties

    bool isHHqr; //!< a flag indicating if a Householder QR has been performed
                 //!< in last iteration
#if defined(HAS_SCALAPACK)
    std::size_t*
        desc1D_Nxnevx_; //!< a ScaLAPACK descriptor for each column communicator
#endif
    Comm_t mpi_wrapper_;
    bool cuda_aware_;
    T *C, *B, *A, *C2, *B2, *vv;
    Matrix<T> *v_0, *v_1, *v_2, *v_w;
    Base<T>* rsd;
    int allreduce_backend, bcast_backend;
    int memcpy_mode[3];

    ChaseMpiMatrices<T>* matrices_;

    //buff
    std::unique_ptr<Matrix<T>> buff__;
    
#if !defined(HAS_SCALAPACK)
    std::unique_ptr<Matrix<T>> V___;
    bool alloc_ = false;
#endif
};
} // namespace mpi
} // namespace chase
