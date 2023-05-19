/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2023, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

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
                ChaseMpiMatrices<T>& matrices, ChaseMpiDLAInterface<T>* dla)
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
        B_ = matrices.get_V2();
        C_ = matrices.get_V1();
        C2_ = matrix_properties->get_C2();
        B2_ = matrix_properties->get_B2();
        A_ = matrix_properties->get_A();
#if !defined(HAS_SCALAPACK)
        V_ = matrix_properties->get_V();
#endif
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

        reqsc2b_.resize(c_lens.size());
        c_sends_.resize(c_lens.size());
        b_recvs_.resize(c_lens.size());

        for (auto i = 0; i < c_lens.size(); i++)
        {
            if (row_rank_ == c_dests[i])
            {
                if (col_rank_ == c_srcs[i])
                {
                    int m1 = send_lens_[0][col_rank_];
                    int array_of_sizes[2] = {m1, 1};
                    int array_of_subsizes[2] = {c_lens[i], 1};
                    int array_of_starts[2] = {c_disps[i], 0};

                    MPI_Type_create_subarray(
                        2, array_of_sizes, array_of_subsizes, array_of_starts,
                        MPI_ORDER_FORTRAN, getMPI_Type<T>(), &(c_sends_[i]));
                    MPI_Type_commit(&(c_sends_[i]));
                }
                else
                {
                    int array_of_sizes2[2] = {static_cast<int>(n_), 1};
                    int array_of_starts2[2] = {b_disps[i], 0};
                    int array_of_subsizes2[2] = {c_lens[i], 1};
                    MPI_Type_create_subarray(
                        2, array_of_sizes2, array_of_subsizes2,
                        array_of_starts2, MPI_ORDER_FORTRAN, getMPI_Type<T>(),
                        &(b_recvs_[i]));
                    MPI_Type_commit(&(b_recvs_[i]));
                }
            }
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

	v0_ = new T[m_];
	v1_ = new T[m_];
	v2_ = new T[m_];
	w_ = new T[n_];
	T *ww;
	mpi_wrapper_ = matrix_properties->get_mpi_wrapper();
	cuda_aware_ = dla_->isCudaAware(); 
        dla_->getMpiWorkSpace(&C, &B, &A, &C2, &B2, &vv, &rsd, &ww);
        dla_->getMpiCollectiveBackend(&allreduce_backend, &bcast_backend);
        if(cuda_aware_)
	{
	    memcpy_mode = CPY_D;	
	}
	else
	{
	    memcpy_mode = CPY_H;
	}	
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
    }
    ~ChaseMpiDLA() {
        delete [] v0_;
        delete [] v1_;
        delete [] v2_;
	delete [] w_;
    }

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
        t_lacpy('A', m_, nev_ + nex_, C_, m_, C2_, m_);
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

	T *C_host;
	dla_->retrieveC(&C_host, locked, block, false);
    	for (auto j = 0; j < block; j++){
   	    for(auto i = 0; i < mblocks_; i++){
	        std::memcpy(C_host + j * m_ + r_offs_l_[i] + locked * m_, V + j * N_ + locked * N_ + r_offs_[i], r_lens_[i] * sizeof(T));
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

    bool postApplication(T* V, std::size_t block, std::size_t locked) override
    {
#ifdef USE_NSIGHT
        nvtxRangePushA("ChaseMpiDLA: postApplication");
#endif
        dla_->postApplication(V, block, locked);

        std::size_t dimsIdx;

        T* buff;
        T* targetBuf = V + locked_ * N_;

        if (next_ == NextOp::bAc)
        {
            T *C_host;
            dla_->retrieveC(&C_host, locked, block, true);

            buff = C_host + locked * m_;
            dimsIdx = 0;
        }
        else
        {
            T *B_host;
            dla_->retrieveB(&B_host, locked, block, true);		
            buff = B_host + locked * n_;
            dimsIdx = 1;
        }

        this->collecRedundantVecs(buff, targetBuf, dimsIdx, block);

#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
        return true;
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
#ifdef USE_NSIGHT
        nvtxRangePushA("MPI_Ibcast");
#endif
        for (auto i = 0; i < c_lens.size(); i++)
        {
            if (row_rank_ == c_dests[i])
            {
                if (col_rank_ == c_srcs[i])
                {
                    MPI_Ibcast(C2 + locked * m_, block, c_sends_[i], c_srcs[i],
                               col_comm_, &reqsc2b_[i]);
                }
                else
                {
                    MPI_Ibcast(B2 + locked * n_, block, b_recvs_[i], c_srcs[i],
                               col_comm_, &reqsc2b_[i]);
                }
            }
        }
#ifdef USE_NSIGHT
        nvtxRangePop();
        nvtxRangePushA("asynCxHGatherC");
#endif
        dla_->asynCxHGatherC(locked, block, isCcopied);
#ifdef USE_NSIGHT
        nvtxRangePop();
        nvtxRangePushA("MPI_Wait");
#endif
        for (auto i = 0; i < c_lens.size(); i++)
        {
            if (row_rank_ == c_dests[i])
            {
                MPI_Wait(&reqsc2b_[i], MPI_STATUSES_IGNORE);
            }
        }
#ifdef USE_NSIGHT
        nvtxRangePop();
        nvtxRangePushA("allreduce");
#endif
	AllReduce(allreduce_backend, B + locked * n_, dim, getMPI_Type<T>(),
                      MPI_SUM, col_comm_, mpi_wrapper_);
	
#ifdef USE_NSIGHT
        nvtxRangePop();
        nvtxRangePushA("t_lacpy");
#endif

        for (auto i = 0; i < c_lens.size(); i++)
        {
            if (row_rank_ == c_dests[i] && col_rank_ == c_srcs[i])
            {
            	dla_->lacpy('A', c_lens[i], block, C2 + locked * m_ + c_disps[i],
                        m_, B2 + locked * n_ + b_disps[i], n_);
	    }
        }
#ifdef USE_NSIGHT
        nvtxRangePop();
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
    void applyVec(T* v, T* w) override
    {
#ifdef USE_NSIGHT
        nvtxRangePushA("ChaseMpiDLA: applyVec");
#endif
        T One = T(1.0);
        T Zero = T(0.0);

        dla_->applyVec(v, w);
        MPI_Allreduce(MPI_IN_PLACE, w, n_,
                      getMPI_Type<T>(), MPI_SUM, col_comm_);

#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
    }

    int get_nprocs() const override { return matrix_properties_->get_nprocs(); }
    void Start() override { dla_->Start(); }
    void End() override { dla_->End(); }

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
        dla_->heevd(LAPACK_COL_MAJOR, 'V', 'L', block, A_, nev_ + nex_, ritzv);
#ifdef USE_NSIGHT
        nvtxRangePop();
        nvtxRangePushA("memcpy");
#endif
        Memcpy(memcpy_mode, C2 + locked * m_, C + locked * m_,
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
        //MPI_Allreduce(MPI_IN_PLACE, resid, unconverged, getMPI_Type<Base<T>>(),
        //              MPI_SUM, row_comm_);
	AllReduce(allreduce_backend, rsd + locked, unconverged, getMPI_Type<Base<T>>(),
                      MPI_SUM, row_comm_, mpi_wrapper_ );
        Base<T> *resid_h;
	dla_->retrieveResid(&resid_h, locked, unconverged);
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
	
        for (std::size_t i = 0; i < unconverged; ++i)
        {
            resid[i] = std::sqrt(resid_h[i]);
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
    int potrf(char uplo, std::size_t n, T* a, std::size_t lda) override
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
	T *C_host;
	dla_->retrieveC(&C_host, 0, nevex, true);
        t_pgeqrf(N_, nevex, C_host, one, one, desc1D_Nxnevx_, tau.get());
        t_pgqr(N_, nevex, nevex, C_host, one, one, desc1D_Nxnevx_, tau.get());
	dla_->putC(C_host, 0, nevex);
#ifdef USE_NSIGHT
        nvtxRangePop();
        nvtxRangePushA("memcpy");
#endif
        Memcpy(memcpy_mode, C, C2, locked * m_ * sizeof(T));
        Memcpy(memcpy_mode, C2 + locked * m_, C + locked * m_,
                        (nevex - locked) * m_ * sizeof(T));
	
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
        this->postApplication(V_, nevex, 0);
        t_geqrf(LAPACK_COL_MAJOR, N_, nevex, V_, N_, tau.get());
        t_gqr(LAPACK_COL_MAJOR, N_, nevex, nevex, V_, N_, tau.get());
        this->preApplication(V_, 0, nevex);
        Memcpy(memcpy_mode, C, C2, locked * m_ * sizeof(T));
        Memcpy(memcpy_mode, C2 + locked * m_, C + locked * m_,
                        (nevex - locked) * m_ * sizeof(T));
#endif
        isHHqr = true;
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
    
    void cholQR(std::size_t locked, Base<T> cond) override
    {
        int grank;
        MPI_Comm_rank(MPI_COMM_WORLD, &grank);

        char* display_bounds_env;
        display_bounds_env = getenv("CHASE_DISPLAY_BOUNDS");
        int display_bounds = 0;
        if(display_bounds_env){
            display_bounds = std::atoi(display_bounds_env);
        }
        if(display_bounds != 0){
            std::vector<T> V2(N_ * (nev_+nex_));
            T *C_host;
	    dla_->retrieveC(&C_host, 0, nev_ + nex_, true);
	    this->collecRedundantVecs(C_host, V2.data(), 0, nev_+nex_);
            std::vector<Base<T>> S(nev_ + nex_ - locked);
            T *U;
            std::size_t ld = 1;
            T *Vt ;
            t_gesvd('N','N',N_, nev_ + nex_ - locked, V2.data() + N_ * locked, N_, S.data(), U, ld, Vt, ld);  
            std::vector<Base<T>> norms(nev_+nex_-locked);
            for(auto i = 0; i < nev_ + nex_-locked; i++){
                norms[i] = std::sqrt(t_sqrt_norm(S[i]));
            }
            std::sort(norms.begin(),norms.end());
            if(grank == 0){
                std::cout << "estimate: " << cond << ", rcond: " << norms[nev_+nex_-locked-1] / norms[0] 
                          << ", ratio: " << cond * norms[0] / norms[nev_+nex_-locked-1] << std::endl;
            }
        }

#ifdef USE_NSIGHT
        nvtxRangePushA("ChaseMpiDLA: cholQR");
#endif
        Base<T> shift;
        bool isShiftQR = false;
        int choldeg = 2;
        int choldeg_env;
        char* choldegenv;
        choldegenv = getenv("CHASE_CHOLQR_DEGREE");
        if (choldegenv)
        {
            choldeg_env = std::atoi(choldegenv);
        }

        // condition for using CholQR1
        Base<T> cond_threshold_1, cond_threshold_2;

        if (sizeof(Base<T>) == 8)
        {
            cond_threshold_1 = 1e8;
            cond_threshold_2 = 2e1;
        }
        else
        {
            cond_threshold_1 = 1e4;
            cond_threshold_2 = 1e1;
        }

        char *chol1_threshold;
        chol1_threshold = getenv("CHASE_CHOLQR1_THLD");
        if(chol1_threshold)
        {
            cond_threshold_2 = std::atof(chol1_threshold);
        }    
        auto nevex = nev_ + nex_;
        bool first_iter = !cuda_aware_;

        T one = T(1.0);
        T zero = T(0.0);
        int info = 1;
#ifdef USE_NSIGHT
        nvtxRangePushA("ChaseMpiDLA: syherk");
#endif
        dla_->syherk('U', 'C', nevex, m_, &one, C, m_, &zero, A, nevex,
                     first_iter);
#ifdef USE_NSIGHT
        nvtxRangePop();
        nvtxRangePushA("allreduce");
#endif
        AllReduce(allreduce_backend, A, nevex * nevex, getMPI_Type<T>(),
                      MPI_SUM, col_comm_, mpi_wrapper_);
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
	//remove shifting temporily for faciliating the impl with cuda-aware

        if(cond > cond_threshold_1){
            isShiftQR = true;
#ifdef USE_NSIGHT
            nvtxRangePushA("ChaseMpiDLA: t_lange");
#endif
//            Base<T> nrmf = t_lange('F', m_, nevex, C_, m_);
	    Base<T> nrmf = dla_->nrm2(m_ * nevex, C, 1);
	    nrmf = std::pow(nrmf, 2);
            //Base<T> nrmf = t_norm_p2(m_ * nevex, C_);
#ifdef USE_NSIGHT
            nvtxRangePop();
            nvtxRangePushA("allreduce");
#endif
            MPI_Allreduce(MPI_IN_PLACE, &nrmf, 1, getMPI_Type<Base<T>>(),
                          MPI_SUM, col_comm_);
            shift = 11 * (N_ * nevex + nevex * nevex + nevex) *
                    std::numeric_limits<Base<T>>::epsilon() * nrmf;

            if(shift < 10){
#ifdef USE_NSIGHT
                nvtxRangePop();
                nvtxRangePushA("ChaseMpiDLA: shift in QR");
#endif
		dla_->shiftMatrixForQR(A, nevex, (T)shift);
#ifdef USE_NSIGHT
                nvtxRangePop();
#endif
            } else
            {
                info = -1;
            }
        }

        if(info != -1){
#ifdef USE_NSIGHT
                nvtxRangePushA("ChaseMpiDLA: potrf");
#endif
                info = dla_->potrf('U', nevex, A, nevex);
#ifdef USE_NSIGHT
                nvtxRangePop();
#endif    
        }

        if (info == 0)
        {
            if (cond < cond_threshold_2)
            {
                choldeg = 1;
            }

            if (choldegenv)
            {
                choldeg = choldeg_env;
            }

            if (isShiftQR && choldeg == 1)
            {
                choldeg = 2;
            }
#ifdef CHASE_OUTPUT
            if (grank == 0)
            {
                std::cout << std::setprecision(2) << "cond(V): " << cond
                          << ", choldegee: " << choldeg;

                if (isShiftQR)
                {
                    std::cout << ", shift: " << shift << std::endl;
                }
                else
                {
                    std::cout << std::endl;
                }
            }
#endif

            if (choldeg == 1)
            {
                first_iter = false;
            }
            else if (choldeg > 1)
            {
                first_iter = true;
            }

#ifdef USE_NSIGHT
            nvtxRangePushA("ChaseMpiDLA: trsm");
#endif
            dla_->trsm('R', 'U', 'N', 'N', m_, nevex, &one, A, nevex, C, m_,
                       first_iter);
#ifdef USE_NSIGHT
            nvtxRangePop();
#endif
            for (auto i = 0; i < choldeg - 1; i++)
            {
#ifdef USE_NSIGHT
                nvtxRangePushA("ChaseMpiDLA: syherk");
#endif
                dla_->syherk('U', 'C', nevex, m_, &one, C, m_, &zero, A,
                             nevex, false);
#ifdef USE_NSIGHT
                nvtxRangePop();
                nvtxRangePushA("allreduce");
#endif
                AllReduce(allreduce_backend, A, nevex * nevex, getMPI_Type<T>(),
                              MPI_SUM, col_comm_, mpi_wrapper_);
#ifdef USE_NSIGHT
                nvtxRangePop();
                nvtxRangePushA("ChaseMpiDLA: potrf");
#endif
                info = dla_->potrf('U', nevex, A, nevex);
#ifdef USE_NSIGHT
                nvtxRangePop();
#endif
                if (i == choldeg - 2)
                {
                    first_iter = false;
                }
                else
                {
                    first_iter = true;
                }
#ifdef USE_NSIGHT
                nvtxRangePushA("ChaseMpiDLA: trsm");
#endif
                dla_->trsm('R', 'U', 'N', 'N', m_, nevex, &one, A, nevex, C,
                           m_, first_iter);
#ifdef USE_NSIGHT
                nvtxRangePop();
#endif
            }

#ifdef USE_NSIGHT
            nvtxRangePushA("memcpy");
#endif
            Memcpy(memcpy_mode, C, C2, locked * m_ * sizeof(T));
            Memcpy(memcpy_mode, C2 + locked * m_, C + locked * m_,
                        (nevex - locked) * m_ * sizeof(T));

	    isHHqr = false;
#ifdef USE_NSIGHT
            nvtxRangePop();
#endif
        }
        else
        {
#ifdef CHASE_OUTPUT
            if (grank == 0)
                std::cout << "cholQR failed because of ill-conditioned vector, "
                             "use Householder QR instead"
                          << std::endl;
#endif
            this->hhQR(locked);
        }

#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
    }
/*
    void cholQR(std::size_t locked, Base<T> cond) override
    {
        int grank;
        MPI_Comm_rank(MPI_COMM_WORLD, &grank);

        char* display_bounds_env;
        display_bounds_env = getenv("CHASE_DISPLAY_BOUNDS");
        int display_bounds = 0;
        if(display_bounds_env){
            display_bounds = std::atoi(display_bounds_env);
        }
        if(display_bounds != 0){
            std::vector<T> V2(N_ * (nev_+nex_));
            this->collecRedundantVecs(C_, V2.data(), 0, nev_+nex_);
            std::vector<Base<T>> S(nev_ + nex_ - locked);
            T *U;
            std::size_t ld = 1;
            T *Vt ;
            t_gesvd('N','N',N_, nev_ + nex_ - locked, V2.data() + N_ * locked, N_, S.data(), U, ld, Vt, ld);  
            std::vector<Base<T>> norms(nev_+nex_-locked);
            for(auto i = 0; i < nev_ + nex_-locked; i++){
                norms[i] = std::sqrt(t_sqrt_norm(S[i]));
            }
            std::sort(norms.begin(),norms.end());
            if(grank == 0){
                std::cout << "estimate: " << cond << ", rcond: " << norms[nev_+nex_-locked-1] / norms[0] 
                          << ", ratio: " << cond * norms[0] / norms[nev_+nex_-locked-1] << std::endl;
            }
        }        
#ifdef USE_NSIGHT
        nvtxRangePushA("ChaseMpiDLA: cholQR");
#endif
        Base<T> shift;
        bool isShiftQR = false;
        int choldeg = 2;
        int choldeg_env;
        char* choldegenv;
        choldegenv = getenv("CHASE_CHOLQR_DEGREE");
        if (choldegenv)
        {
            choldeg_env = std::atoi(choldegenv);
        }

        // condition for using CholQR1
        Base<T> cond_threshold_2;

        if (sizeof(Base<T>) == 8)
        {
            cond_threshold_2 = 5e1;
        }
        else
        {
            cond_threshold_2 = 1e1;
        }

        char *chol1_threshold;
        chol1_threshold = getenv("CHASE_CHOLQR1_THLD");
        if(chol1_threshold)
        {
            cond_threshold_2 = std::atof(chol1_threshold);
        }    
        auto nevex = nev_ + nex_;
        bool first_iter = true;
        T one = T(1.0);
        T zero = T(0.0);
        int info = -1;
#ifdef USE_NSIGHT
        nvtxRangePushA("ChaseMpiDLA: syherk");
#endif
        dla_->syherk('U', 'C', nevex, m_, &one, C_, m_, &zero, A_, nevex,
                     first_iter);
#ifdef USE_NSIGHT
        nvtxRangePop();
        nvtxRangePushA("allreduce");
#endif
        MPI_Allreduce(MPI_IN_PLACE, A_, nevex * nevex, getMPI_Type<T>(),
                      MPI_SUM, col_comm_);
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif

#ifdef USE_NSIGHT
        nvtxRangePushA("ChaseMpiDLA: potrf");
#endif
        info = dla_->potrf('U', nevex, A_, nevex);
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif

        if (info != 0)
        {
            isShiftQR = true;
#ifdef USE_NSIGHT
            nvtxRangePushA("ChaseMpiDLA: t_lange");
#endif
            Base<T> nrmf = t_lange('F', m_, nevex, C_, m_);
            nrmf = std::pow(nrmf, 2);
            //Base<T> nrmf = t_norm_p2(m_ * nevex, C_);
#ifdef USE_NSIGHT
            nvtxRangePop();
            nvtxRangePushA("allreduce");
#endif
            MPI_Allreduce(MPI_IN_PLACE, &nrmf, 1, getMPI_Type<Base<T>>(),
                          MPI_SUM, col_comm_);
            shift = 11 * (N_ * nevex + nevex * nevex + nevex) *
                    std::numeric_limits<Base<T>>::epsilon() * nrmf;

            if(shift < 10){
#ifdef USE_NSIGHT
                nvtxRangePop();
                nvtxRangePushA("ChaseMpiDLA: shift in QR");
#endif
                for (auto i = 0; i < nevex; i++)
                {
                    A_[i * nevex + i] += (T)shift;
                }
#ifdef USE_NSIGHT
                nvtxRangePop();
#endif
#ifdef USE_NSIGHT
                nvtxRangePushA("ChaseMpiDLA: potrf");
#endif
                info = dla_->potrf('U', nevex, A_, nevex);
#ifdef USE_NSIGHT
                nvtxRangePop();
#endif
            } else
            {
                info = -1;
            }    

        }

        if (info == 0)
        {
            if (cond < cond_threshold_2)
            {
                choldeg = 1;
            }

            if (choldegenv)
            {
                choldeg = choldeg_env;
            }

            if (isShiftQR && choldeg == 1)
            {
                choldeg = 2;
            }
#ifdef CHASE_OUTPUT
            if (grank == 0)
            {
                std::cout << std::setprecision(2) << "cond(V): " << cond
                          << ", choldegee: " << choldeg;

                if (isShiftQR)
                {
                    std::cout << ", shift: " << shift << std::endl;
                }
                else
                {
                    std::cout << std::endl;
                }
            }
#endif

            if (choldeg == 1)
            {
                first_iter = false;
            }
            else if (choldeg > 1)
            {
                first_iter = true;
            }

#ifdef USE_NSIGHT
            nvtxRangePushA("ChaseMpiDLA: trsm");
#endif
            dla_->trsm('R', 'U', 'N', 'N', m_, nevex, &one, A_, nevex, C_, m_,
                       first_iter);
#ifdef USE_NSIGHT
            nvtxRangePop();
#endif
            for (auto i = 0; i < choldeg - 1; i++)
            {
#ifdef USE_NSIGHT
                nvtxRangePushA("ChaseMpiDLA: syherk");
#endif
                dla_->syherk('U', 'C', nevex, m_, &one, C_, m_, &zero, A_,
                             nevex, false);
#ifdef USE_NSIGHT
                nvtxRangePop();
                nvtxRangePushA("allreduce");
#endif
                MPI_Allreduce(MPI_IN_PLACE, A_, nevex * nevex, getMPI_Type<T>(),
                              MPI_SUM, col_comm_);
#ifdef USE_NSIGHT
                nvtxRangePop();
                nvtxRangePushA("ChaseMpiDLA: potrf");
#endif
                info = dla_->potrf('U', nevex, A_, nevex);
#ifdef USE_NSIGHT
                nvtxRangePop();
#endif
                if (i == choldeg - 2)
                {
                    first_iter = false;
                }
                else
                {
                    first_iter = true;
                }
#ifdef USE_NSIGHT
                nvtxRangePushA("ChaseMpiDLA: trsm");
#endif
                dla_->trsm('R', 'U', 'N', 'N', m_, nevex, &one, A_, nevex, C_,
                           m_, first_iter);
#ifdef USE_NSIGHT
                nvtxRangePop();
#endif
            }

#ifdef USE_NSIGHT
            nvtxRangePushA("memcpy");
#endif
            std::memcpy(C_, C2_, locked * m_ * sizeof(T));
            std::memcpy(C2_ + locked * m_, C_ + locked * m_,
                        (nevex - locked) * m_ * sizeof(T));
            isHHqr = false;
#ifdef USE_NSIGHT
            nvtxRangePop();
#endif
        }
        else
        {
#ifdef CHASE_OUTPUT
            if (grank == 0)
                std::cout << "cholQR failed because of ill-conditioned vector, "
                             "use Householder QR instead"
                          << std::endl;
#endif
            this->hhQR(locked);
        }

#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
    }
*/

    void Swap(std::size_t i, std::size_t j) override
    {
        Memcpy(memcpy_mode, vv, C + m_ * i, m_ * sizeof(T));
        Memcpy(memcpy_mode, C + m_ * i, C + m_ * j, m_ * sizeof(T));
        Memcpy(memcpy_mode, C + m_ * j, vv, m_ * sizeof(T));

        Memcpy(memcpy_mode, vv, C + m_ * i, m_ * sizeof(T));
        Memcpy(memcpy_mode, C + m_ * i, C + m_ * j, m_ * sizeof(T));
        Memcpy(memcpy_mode, C + m_ * j, vv, m_ * sizeof(T));	
    }

    void LanczosDos(std::size_t idx, std::size_t m, T* ritzVc) override
    {
        T alpha = T(1.0);
        T beta = T(0.0);
#ifdef USE_NSIGHT
        nvtxRangePushA("ChaseMpiDLA: LanczosDOS");
#endif
        t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_, idx, m, &alpha,
               C_, m_, ritzVc, m, &beta, C2_, m_);
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
	dla_->LanczosDos(idx, m, ritzVc);
    }

    void Lanczos(std::size_t M, int idx, Base<T>* d, Base<T>* e, Base<T> *r_beta) override
    {
        Base<T> real_beta;

        T alpha = T(1.0);
        T beta = T(0.0);

        std::fill(v0_, v0_ + m_, T(0));

        if(idx >= 0)
        {
	      std::memcpy(v1_, C2_ + idx * m_, m_ * sizeof(T) );
	}else
        {
            std::mt19937 gen(2342.0);
            std::normal_distribution<> normal_distribution;
            //v1_ = get_V1();
            for (std::size_t k = 0; k < m_; ++k)
            {
                v1_[k] = getRandomT<T>([&]() { return normal_distribution(gen); });
            }            
        }
        // ENSURE that v1 has one norm
#ifdef USE_NSIGHT
        nvtxRangePushA("Lanczos: loop");
#endif
        //Base<T> real_alpha = t_norm_p2(m_, v1_);
        Base<T> real_alpha = t_nrm2(m_, v1_, 1);
	real_alpha = std::pow(real_alpha,2);
	MPI_Allreduce(MPI_IN_PLACE, &real_alpha, 1, getMPI_Type<Base<T>>(),
                    MPI_SUM, col_comm_);
        real_alpha = std::sqrt(real_alpha);
        alpha = T(1 / real_alpha);
        t_scal(m_, &alpha, v1_, 1);
        for (std::size_t k = 0; k < M; k = k + 1)
        {
    	    if(idx >= 0){
		std::memcpy(C_ + k * m_, v1_, m_ * sizeof(T) );
            }
            this->applyVec(v1_, w_);
            this->B2C(w_, 0, v2_, 0, 1);
            alpha = t_dot(m_, v1_, 1, v2_, 1);
            MPI_Allreduce(MPI_IN_PLACE, &alpha, 1, getMPI_Type<T>(),
                          MPI_SUM, col_comm_);
            alpha = -alpha;
            t_axpy(m_, &alpha, v1_, 1, v2_, 1);

            alpha = -alpha;

            d[k] = std::real(alpha);

            if (k == M - 1)
                break;

            beta = T(-real_beta);
            t_axpy(m_, &beta, v0_, 1, v2_, 1);
            beta = -beta;

            //real_beta = t_norm_p2(m_, v2_);
            real_beta = t_nrm2(m_, v2_, 1);
	    real_beta = std::pow(real_beta, 2);
	    MPI_Allreduce(MPI_IN_PLACE, &real_beta, 1, getMPI_Type<Base<T>>(),
                          MPI_SUM, col_comm_);
            real_beta = std::sqrt(real_beta);

            beta = T(1.0 / real_beta);

            t_scal(m_, &beta, v2_, 1);

            e[k] = real_beta;

            std::swap(v1_, v0_);
            std::swap(v1_, v2_);
	    
        }
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
        *r_beta = real_beta;            
    }

    void B2C(T* B, std::size_t off1, T* C, std::size_t off2, std::size_t block) override
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
                t_lacpy('A', b_lens[i], block, B + off1 * n_ + b_disps_2[i],
                        n_, C + off1 * m_ + c_disps_2[i], m_);
            }
        }
    }

    void getMpiWorkSpace(T **C, T **B, T **A, T **C2, T **B2, T **vv, Base<T> **rsd, T **w) override
    {}

    void getMpiCollectiveBackend(int *allreduce_backend, int *bcast_backend) override
    {}

    bool isCudaAware() override
    {
        return cuda_aware_;
    }

    void lacpy(char uplo, std::size_t m, std::size_t n,
             T* a, std::size_t lda, T* b, std::size_t ldb) override
    {}

    void shiftMatrixForQR(T *A, std::size_t n, T shift) override
    {}

    void retrieveC(T **C, std::size_t locked, std::size_t block, bool copy) override
    {}

    void retrieveB(T **B, std::size_t locked, std::size_t block, bool copy) override
    {}
	
    void retrieveResid(Base<T> **rsd, std::size_t locked, std::size_t block) override
    {}

    void putC(T *C, std::size_t locked, std::size_t block) override
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

    T* B_;  //!< a matrix of size `n_*(nev_+nex_)`, which is allocated in
            //!< ChaseMpiMatrices
    T* C_;  //!< a matrix of size `m_*(nev_+nex_)`, which is allocated in
            //!< ChaseMpiMatrices
    T* C2_; //!< a matrix of size `m_*(nev_+nex_)`, which is allocated in
            //!< ChaseMpiProperties
    T* B2_; //!< a matrix of size `n_*(nev_+nex_)`, which is allocated in
            //!< ChaseMpiProperties
    T* A_;  //!< a matrix of size `(nev_+nex_)*(nev_+nex_)`, which is allocated
            //!< in ChaseMpiProperties
    T* v0_; //!< a vector of size `N_`, which is allocated in this
                        //!< class for Lanczos
    T* v1_; //!< a vector of size `N_`, which is allocated in this
                        //!< class for Lanczos
    T* w_;  //!< a vector of size `N_`, which is allocated in this
                        //!< class for Lanczos
    T* v2_;

    std::vector<T> Buff_; //!< a vector of size `N_`, it is allocated only ChASE
                          //!< working with `Block-Cyclic`
#if !defined(HAS_SCALAPACK)
    T* V_; //!< a matrix of size `N_*(nev_+nex_)`, only allocated when no
           //!< ScaLAPACK is detected
#endif

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
    Base<T> *rsd;
    int allreduce_backend, bcast_backend;
    int memcpy_mode;

};
} // namespace mpi
} // namespace chase
