/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2023, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <memory>
#include <mpi.h>
#include <tuple>
#include <vector>

#include "algorithm/types.hpp"
#include "chase_mpi_matrices.hpp"
#include "mpi_wrapper.hpp"

namespace chase
{
namespace mpi
{

template <typename T>
MPI_Datatype getMPI_Type();

template <>
MPI_Datatype getMPI_Type<float>()
{
    return MPI_FLOAT;
}

template <>
MPI_Datatype getMPI_Type<double>()
{
    return MPI_DOUBLE;
}

template <>
MPI_Datatype getMPI_Type<std::complex<float>>()
{
    return MPI_COMPLEX;
}

template <>
MPI_Datatype getMPI_Type<std::complex<double>>()
{
    return MPI_DOUBLE_COMPLEX;
}

std::pair<std::size_t, std::size_t> numroc(std::size_t n, std::size_t nb,
                                           int iproc, int isrcproc, int nprocs)
{

    std::size_t numroc;
    std::size_t extrablks, mydist, nblocks;
    mydist = (nprocs + iproc - isrcproc) % nprocs;
    nblocks = n / nb;
    numroc = (nblocks / nprocs) * nb;
    extrablks = nblocks % nprocs;

    if (mydist < extrablks)
        numroc = numroc + nb;
    else if (mydist == extrablks)
        numroc = numroc + n % nb;

    std::size_t nb_loc = numroc / nb;

    if (numroc % nb != 0)
    {
        nb_loc += 1;
    }

    return std::make_pair(numroc, nb_loc);
}

void create2DGrid(int row_dim, int col_dim, bool col_major, MPI_Comm comm,
                  MPI_Comm* row_comm_, MPI_Comm* col_comm_, int* myrow,
                  int* mycol)
{
    int tmp_dims_[2];
    int dims_[2];
    int coord_[2];
    dims_[0] = row_dim;
    dims_[1] = col_dim;

    if (col_major)
    {
        tmp_dims_[1] = row_dim;
        tmp_dims_[0] = col_dim;
    }
    else
    {
        tmp_dims_[0] = row_dim;
        tmp_dims_[1] = col_dim;
    }

    int periodic[] = {0, 0};
    int reorder = 0;
    int free_coords[2];
    int row_procs, col_procs;
    int tmp_coord[2];
    int rank_, nprocs_;

    MPI_Comm cartComm;
    MPI_Cart_create(comm, 2, tmp_dims_, periodic, reorder, &cartComm);

    MPI_Comm_size(cartComm, &nprocs_);
    MPI_Comm_rank(cartComm, &rank_);
    MPI_Cart_coords(cartComm, rank_, 2, tmp_coord);

    if (col_major)
    {
        coord_[1] = tmp_coord[0];
        coord_[0] = tmp_coord[1];
    }
    else
    {
        coord_[1] = tmp_coord[1];
        coord_[0] = tmp_coord[0];
    }

    // row communicator
    if (col_major)
    {
        free_coords[0] = 1;
        free_coords[1] = 0;
    }
    else
    {
        free_coords[0] = 0;
        free_coords[1] = 1;
    }

    MPI_Cart_sub(cartComm, free_coords, row_comm_);
    MPI_Comm_size(*row_comm_, &row_procs);

    // column communicator
    if (col_major)
    {
        free_coords[0] = 0;
        free_coords[1] = 1;
    }
    else
    {
        free_coords[0] = 1;
        free_coords[1] = 0;
    }

    MPI_Cart_sub(cartComm, free_coords, col_comm_);
    MPI_Comm_size(*col_comm_, &col_procs);

    *myrow = coord_[0];
    *mycol = coord_[1];
}

//! @brief A class to setup **MPI** properties and matrix distribution scheme
//! for the implementation of ChASE on distributed-memory systems.
/*! @details
    The ChaseMpiProperties class creates a 2D grid of MPI nodes with fixed width
   and height. It defines also the scheme to distribute an Hermitian matrix `A`
   of side `N` over this 2D grid of MPI nodes. Currently, two data distribution
   fashion are supported in ChASE.
      - **Block distribution** scheme in which one submatrix of `A` is assigned
   to one single MPI node;
      - **Block-Cyclic distribution** scheme, which distributes a series of
   submatrices of `A` to the MPI nodes in a round-robin manner so that each MPI
   rank gets several non-adjacent blocks. More details about this distribution
   can be found on <a
   href="https://www.netlib.org/scalapack/slug/node75.html">Netlib</a>.
    @tparam T: the scalar type used for the application. ChASE is templated
    for real and complex scalars with both Single Precision and Double
   Precision, thus `T` can be one of `float`, `double`, `std::complex<float>`
   and `std::complex<double>`.
 */
template <class T>
class ChaseMpiProperties
{
public:
    // row_dim = number of row in the grid
    // col_dim = number of col in the grid
    //! A constructor of the class ChaseMpiProperties which distributes matrix
    //! `A` in **Block-Cyclic Distribution**.
    /*!
        It constructs a 2D grid of MPI ranks within the MPI communicator
       `comm_`.
        - The dimension of this 2D grid is determined by explicit values
       `row_dim` and `col_dim`, thus `row_dim * col_dim`.
        - The major to index the coordinations
        of the MPI ranks is determined by the value of `grid_major`: if
       `grid_major="C"`, they are indexed by `column major`, if
       `grid_major="R"`, they are indexed by `Row major`.
        - It distributes the Hermitian matrix `A` in a **Block-Dsitribution**
       scheme.

        This constructor requires the explicit values for the initalization of
       the size `N` of the matrix *A*, the number of sought after extremal
        eigenvalues `nev`, and the number of extra eigenvalue `nex` which
        defines, together with `nev`, the search space, and the working MPI
        communicator `comm_` for the construction of eigenproblem.
        For the distribution in block-cyclic scheme, it requires also the
       dimensions of 2D grid `row_dim` and `col_dim`, the row and column
       blocking factors `mb` and `nb`, the major of grid indexing `grid_major`,
       and the process row/column over which the first row/column of the global
       matrix `A` is distributed.

        The parameters related to **Block-Cyclic Distribution** conforms equally
       with the <a
       href="https://www.netlib.org/scalapack/slug/node77.html">Array
       Descriptor</a> of in-core Dense Matrices in **ScaLAPACK**.

        All the private members are either initialized
        directly by these parameters, or setup within the construction of this
        constructor.
        \param N Size of the square matrix defining the eigenproblem.
        \param mb Row blocking factor for *Block-Cyclic Distribution*.
        \param nb Column blocking factor for *Block-Cyclic Distribution*`.
        \param nev Number of desired extremal eigenvalues.
        \param nex Number of eigenvalues augmenting the search space. Usually a
       relatively small fraction of `nev`. \param row_dim number of row in the
       2D grid of MPI nodes to be constructed. \param col_dim number of column
       in the 2D grid of MPI nodes to be constructed. \param grid_major the type
       of numbering of the MPI ranks within the constructed 2D grid.
          - Row major numbering if `grid_major=="C"`.
          - Column major numbering if `grid_major=="R"`.
        \param irsrc Process row over which the first row of the global matrix
       `A` is distributed. \param icsrc Process column over which the first
       column of the global matrix `A` is distributed. \param comm the working
       MPI communicator for ChASE.
     */
    ChaseMpiProperties(std::size_t N, std::size_t mb, std::size_t nb,
                       std::size_t nev, std::size_t nex, int row_dim,
                       int col_dim, char* grid_major, int irsrc, int icsrc,
                       MPI_Comm comm)
        : N_(N), mb_(mb), nb_(nb), nev_(nev), nex_(nex), max_block_(nev + nex),
          irsrc_(irsrc), icsrc_(icsrc), comm_(comm)
    {
#ifdef USE_NSIGHT
        nvtxRangePushA("ChaseMpiProperties(block-cyclic): Init");
#endif
        data_layout = "Block-Cyclic";

        MPI_Comm_size(comm, &nprocs_);
        MPI_Comm_rank(comm, &rank_);

        /* Set sharem memory communicator and ranks */
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm_);
        MPI_Comm_size(shm_comm_, &shm_nprocs_);
        MPI_Comm_rank(shm_comm_, &shm_rank_);

#ifdef HAS_CUDA
        int num_devices = -1;
        cudaGetDeviceCount(&num_devices);

        // Check the number of MPI ranks per node and the number of available GPUs
        if(num_devices < shm_nprocs_) {
            if (!rank_) {
                std::cerr << "Error! The number of MPI ranks per node ( " << shm_nprocs_ 
                        << " ) is larger than the number of available GPU devices ( " 
                        << num_devices << " )!" << std::endl;
                std::cerr << "The ChASE will terminate. Please, re-run ChASE with the number of " 
                        << "MPI ranks per node not larger than the number of available GPU devices."
                        << std::endl;
            }
        } else if (num_devices > shm_nprocs_) {
            cudaSetDevice(shm_rank_);
            if (!rank_) {
                std::cout << "Warning! The number of MPI ranks ( " << shm_nprocs_ 
                        << " ) is less than the number of available GPUs ( " << num_devices << " )! " << std::endl;
                std::cout << "The resources will not be used optimally. " 
                        << "The number of MPI ranks should be equal to the number of available GPU devices to exploit all computational resources!" 
                        << std::endl;
            }
        } else { // num_devices mpi_shm_rankmpi_shm_size
            cudaSetDevice(shm_rank_);
        }
#endif
        std::size_t blocknb[2];
        std::size_t N_loc[2];
        std::size_t blocksize[2];
        blocksize[0] = mb_;
        blocksize[1] = nb_;

        int tmp_dims_[2];
        dims_[0] = row_dim;
        dims_[1] = col_dim;

        bool col_major = false;

        if (strcmp(grid_major, "C") == 0)
        {
            col_major = true;
        }

        int isrc[2];
        isrc[0] = irsrc;
        isrc[1] = icsrc;

        create2DGrid(dims_[0], dims_[1], col_major, comm, &row_comm_,
                     &col_comm_, &coord_[0], &coord_[1]);

        for (std::size_t dim_idx = 0; dim_idx < 2; dim_idx++)
        {
            std::tie(N_loc[dim_idx], blocknb[dim_idx]) =
                numroc(N_, blocksize[dim_idx], coord_[dim_idx], isrc[dim_idx],
                       dims_[dim_idx]);
        }

        mblocks_ = blocknb[0];
        nblocks_ = blocknb[1];
        m_ = N_loc[0];
        n_ = N_loc[1];

        ldh_ = m_;

        r_offs_.reset(new std::size_t[mblocks_]());
        r_lens_.reset(new std::size_t[mblocks_]());
        r_offs_l_.reset(new std::size_t[mblocks_]());
        c_offs_.reset(new std::size_t[nblocks_]());
        c_lens_.reset(new std::size_t[nblocks_]());
        c_offs_l_.reset(new std::size_t[nblocks_]());

        // get offsets
        std::size_t sendr = irsrc_;
        std::size_t sendc = icsrc_;

        int cnt = 0;
        std::size_t nr, nc;

        // for the first dimension of grid
        for (std::size_t r = 0; r < N_;
             r += mb_, sendr = (sendr + 1) % dims_[0])
        {
            nr = mb_;
            if (N_ - r < mb_)
            {
                nr = N_ - r;
            }

            if (coord_[0] == sendr)
            {
                r_offs_[cnt] = r;
                r_lens_[cnt] = nr;
                cnt++;
            }
        }

        cnt = 0;
        // for the second dimension of grid
        for (std::size_t c = 0; c < N_;
             c += nb_, sendc = (sendc + 1) % dims_[1])
        {
            nc = nb_;
            if (N_ - c < nb_)
            {
                nc = N_ - c;
            }
            if (coord_[1] == sendc)
            {
                c_offs_[cnt] = c;
                c_lens_[cnt] = nc;
                cnt++;
            }
        }

        cnt = 0;
        r_offs_l_[0] = 0;
        c_offs_l_[0] = 0;

        for (std::size_t i = 1; i < mblocks_; i++)
        {
            r_offs_l_[i] = r_offs_l_[i - 1] + r_lens_[i - 1];
        }

        for (std::size_t j = 1; j < nblocks_; j++)
        {
            c_offs_l_[j] = c_offs_l_[j - 1] + c_lens_[j - 1];
        }

        // for MPI communication
        block_counts_.resize(2);
        send_lens_.resize(2);

        for (std::size_t dim_idx = 0; dim_idx < 2; dim_idx++)
        {
            block_counts_[dim_idx].resize(dims_[dim_idx]);
            send_lens_[dim_idx].resize(dims_[dim_idx]);
            for (std::size_t i = 0; i < dims_[dim_idx]; i++)
            {
                std::tie(send_lens_[dim_idx][i], block_counts_[dim_idx][i]) =
                    numroc(N_, blocksize[dim_idx], i, isrc[dim_idx],
                           dims_[dim_idx]);
            }
        }

        block_displs_.resize(2);
        block_lens_.resize(2);
        std::size_t send[2];
        send[0] = irsrc_;
        send[1] = icsrc_;
        g_offsets_.resize(2);

        for (std::size_t dim_idx = 0; dim_idx < 2; dim_idx++)
        {
            block_lens_[dim_idx].resize(dims_[dim_idx]);
            block_displs_[dim_idx].resize(dims_[dim_idx]);

            int displs_cnt = 0;

            for (std::size_t i = 0; i < dims_[dim_idx]; i++)
            {
                block_lens_[dim_idx][i].resize(block_counts_[dim_idx][i]);
                block_displs_[dim_idx][i].resize(block_counts_[dim_idx][i]);
                nr = 0;
                cnt = 0;
                send[0] = irsrc_;
                send[1] = icsrc_;
                for (std::size_t r = 0; r < N_; r += blocksize[dim_idx],
                                 send[dim_idx] = (send[dim_idx] + 1) %
                                                 dims_[dim_idx])
                {
                    nr = blocksize[dim_idx];
                    if (N_ - r < blocksize[dim_idx])
                    {
                        nr = N_ - r;
                    }
                    if (i == send[dim_idx])
                    {
                        block_lens_[dim_idx][i][cnt] = nr;
                        block_displs_[dim_idx][i][cnt] = r;
                        g_offsets_[dim_idx].push_back(r);

                        cnt++;
                    }
                }
            }
        }
#if defined(HAS_SCALAPACK)
        int zero = 0;
        int ictxt;
        blacs_get_(&zero, &zero, &ictxt);
        colcomm_ctxt_ = ictxt;
        int userMap[dims_[0]];
        if (col_major == false)
        {
            for (int i = 0; i < dims_[0]; i++)
            {
                userMap[i] = (rank_ % dims_[1]) + dims_[1] * i;
            }
        }
        else
        {
            for (int i = 0; i < dims_[0]; i++)
            {
                userMap[i] = (rank_ / dims_[0]) * dims_[0] + i;
            }
        }
        int ONE = 1;
        blacs_gridmap_(&colcomm_ctxt_, userMap, &dims_[0], &dims_[0], &ONE);
        int info;
        std::size_t nevex_ = nev_ + nex_;
        std::size_t bsize = 32;
        std::size_t nx_b = std::min(bsize, nevex_);
        t_descinit(desc1D_Nxnevx_, &N_, &nevex_, &mb_, &nx_b, &irsrc_, &zero,
                   &colcomm_ctxt_, &m_, &info);
#endif

        comm_2 row_comm_dup;
        comm_2 col_comm_dup;
#if defined(HAS_NCCL)
        ncclUniqueId nccl_id, nccl_ids[nprocs_];
        ncclGetUniqueId(&nccl_id);
        MPI_Allgather(&nccl_id, sizeof(ncclUniqueId), MPI_UINT8_T, &nccl_ids[0],
                      sizeof(ncclUniqueId), MPI_UINT8_T, comm);

        for (auto i = 0; i < dims_[0]; i++)
        {
            if (coord_[0] == i)
            {
                ncclCommInitRank(&row_comm_dup, dims_[1], nccl_ids[i],
                                 coord_[1]);
            }
        }

        // col_comm
        ncclUniqueId nccl_id_2, nccl_ids_2[nprocs_];
        ncclGetUniqueId(&nccl_id_2);
        MPI_Allgather(&nccl_id_2, sizeof(ncclUniqueId), MPI_UINT8_T,
                      &nccl_ids_2[0], sizeof(ncclUniqueId), MPI_UINT8_T, comm);

        for (auto i = 0; i < dims_[1]; i++)
        {
            if (coord_[1] == i)
            {
                ncclCommInitRank(&col_comm_dup, dims_[0],
                                 nccl_ids_2[i * dims_[0]], coord_[0]);
            }
        }
#else
        MPI_Comm_dup(row_comm_, &row_comm_dup);
        MPI_Comm_dup(col_comm_, &col_comm_dup);
#endif
        mpi_wrapper_.add(row_comm_, row_comm_dup);
        mpi_wrapper_.add(col_comm_, col_comm_dup);

#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
    }

    //! A constructor of the class ChaseMpiProperties which distributes matrix
    //! `A` in `Block Distribution`.
    /*!
        It constructs a 2D grid of MPI ranks within the MPI communicator
       `comm_`.

        - The dimensions of this 2D grid is determined by the input arguments
       `npr` and `npc`. The 2D grid is `npr x npc`
        - It distributes the Hermitian matrix `A` in a **Block-Dsitribution**
       scheme.

        This constructor requires the explicit values for the initalization of
       the size `N` of the matrix *A*, the number of sought after extremal
        eigenvalues `nev`, and the number of extra eigenvalue `nex` which
        defines, together with `nev`, the search space, the dimension of local
       matrix `m` and `n`, the 2D MPI grid `npr` and `npc`, and the working MPI
       communicator `comm_`.

        All the private members are either initialized
        directly by these parameters, or setup within the construction of this
       constructor.

        \param N Size of the square matrix defining the eigenproblem.
        \param nev Number of desired extremal eigenvalues.
        \param nex Number of eigenvalues augmenting the search space. Usually a
       relatively small fraction of `nev`. \param m row number of local matrix
       on each MPI rank \param n column number of local matrix on each MPI rank
        \param npr row number of 2D MPI grid
        \param npc column number of 2D MPI grid
        \param grid_major the type
        of numbering of the MPI ranks within the constructed 2D grid.
          - Row major numbering if `grid_major=="C"`.
          - Column major numbering if `grid_major=="R"`.

        \param comm the working MPI communicator for ChASE.
     */
    ChaseMpiProperties(std::size_t N, std::size_t nev, std::size_t nex,
                       std::size_t m, std::size_t n, int npr, int npc,
                       char* grid_major, MPI_Comm comm)
        : N_(N), nev_(nev), nex_(nex), max_block_(nev + nex), m_(m), n_(n),
          comm_(comm)
    {
#ifdef USE_NSIGHT
        nvtxRangePushA("ChaseMpiProperties(Block-Block1): Init");
#endif
        data_layout = "Block-Block";

        MPI_Comm_size(comm, &nprocs_);
        MPI_Comm_rank(comm, &rank_);

        /* Set sharem memory communicator and ranks */
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm_);
        MPI_Comm_size(shm_comm_, &shm_nprocs_);
        MPI_Comm_rank(shm_comm_, &shm_rank_);
        
#ifdef HAS_CUDA
        int num_devices = -1;
        cudaGetDeviceCount(&num_devices);

        // Check the number of MPI ranks per node and the number of available GPUs
       if(num_devices < shm_nprocs_) {
            if (!rank_) {
                std::cerr << "Error! The number of MPI ranks per node ( " << shm_nprocs_ 
                        << " ) is larger than the number of available GPU devices ( " 
                        << num_devices << " )!" << std::endl;
                std::cerr << "The ChASE will terminate. Please, re-run ChASE with the number of " 
                        << "MPI ranks per node not larger than the number of available GPU devices."
                        << std::endl;
            }
        } else if (num_devices > shm_nprocs_) {
            cudaSetDevice(shm_rank_);
            if (!rank_) {
                std::cout << "Warning! The number of MPI ranks ( " << shm_nprocs_ 
                        << " ) is less than the number of available GPUs ( " << num_devices << " )! " << std::endl;
                std::cout << "The resources will not be used optimally. " 
                        << "The number of MPI ranks should be equal to the number of available GPU devices to exploit all computational resources!" 
                        << std::endl;
            }
        } else { // num_devices mpi_shm_rankmpi_shm_size
            cudaSetDevice(shm_rank_);
        }
#endif

        int tmp_dims_[2];

        dims_[0] = npr;
        dims_[1] = npc;

        bool col_major = false;

        if (strcmp(grid_major, "C") == 0)
        {
            col_major = true;
        }

        create2DGrid(dims_[0], dims_[1], col_major, comm, &row_comm_,
                     &col_comm_, &coord_[0], &coord_[1]);

        std::size_t len;
        len = m;
        off_[0] = coord_[0] * len;
        if (coord_[0] < dims_[0] - 1)
        {
            m_ = len;
        }
        else
        {
            m_ = N_ - (dims_[0] - 1) * len;
        }

        len = n;
        off_[1] = coord_[1] * len;
        ldh_ = m_;

        if (coord_[1] < dims_[1] - 1)
        {
            n_ = len;
        }
        else
        {
            n_ = N_ - (dims_[1] - 1) * len;
        }

        nb_ = n_;
        if (coord_[1] == dims_[1] - 1 && dims_[1] != 1)
        {
            nb_ = (N_ - n_) / (dims_[1] - 1);
        }

        mb_ = m_;
        if (coord_[0] == dims_[0] - 1 && dims_[0] != 1)
        {
            mb_ = (N_ - m_) / (dims_[0] - 1);
        }

        mblocks_ = 1;
        nblocks_ = 1;

        irsrc_ = 0;
        icsrc_ = 0;

        r_offs_.reset(new std::size_t[1]());
        r_lens_.reset(new std::size_t[1]());
        r_offs_l_.reset(new std::size_t[1]());
        c_offs_.reset(new std::size_t[1]());
        c_lens_.reset(new std::size_t[1]());
        c_offs_l_.reset(new std::size_t[1]());

        r_offs_[0] = off_[0];
        r_lens_[0] = m_;
        r_offs_l_[0] = 0;
        c_offs_[0] = off_[1];
        c_lens_[0] = n_;
        c_offs_l_[0] = 0;

        block_counts_.resize(2);
        for (std::size_t dim_idx = 0; dim_idx < 2; dim_idx++)
        {
            block_counts_[dim_idx].resize(dims_[dim_idx]);
            for (std::size_t i = 0; i < dims_[dim_idx]; i++)
            {
                block_counts_[dim_idx][i] = 1;
            }
        }

        block_displs_.resize(2);
        block_lens_.resize(2);
        send_lens_.resize(2);
        g_offsets_.resize(2);

        for (std::size_t dim_idx = 0; dim_idx < 2; dim_idx++)
        {
            block_lens_[dim_idx].resize(dims_[dim_idx]);
            block_displs_[dim_idx].resize(dims_[dim_idx]);
            send_lens_[dim_idx].resize(dims_[dim_idx]);
            for (std::size_t i = 0; i < dims_[dim_idx]; ++i)
            {
                block_lens_[dim_idx][i].resize(1);
                block_displs_[dim_idx][i].resize(1);
                if (dim_idx == 0)
                {
                    len = m;
                }
                else
                {
                    len = n;
                }
                block_lens_[dim_idx][i][0] = len;
                block_displs_[dim_idx][i][0] = i * block_lens_[dim_idx][0][0];
                send_lens_[dim_idx][i] = len;
                g_offsets_[dim_idx].push_back(block_displs_[dim_idx][i][0]);
            }
            block_lens_[dim_idx][dims_[dim_idx] - 1].resize(1);
            block_displs_[dim_idx][dims_[dim_idx] - 1].resize(1);
            block_lens_[dim_idx][dims_[dim_idx] - 1][0] =
                N_ - (dims_[dim_idx] - 1) * len;
            block_displs_[dim_idx][dims_[dim_idx] - 1][0] =
                (dims_[dim_idx] - 1) * block_lens_[dim_idx][0][0];
            send_lens_[dim_idx][dims_[dim_idx] - 1] =
                N_ - (dims_[dim_idx] - 1) * len;
            g_offsets_[dim_idx].push_back(
                block_displs_[dim_idx][dims_[dim_idx] - 1][0]);
        }

#if defined(HAS_SCALAPACK)
        int zero = 0;
        int ictxt;

        blacs_get_(&zero, &zero, &ictxt);
        colcomm_ctxt_ = ictxt;
        int userMap[dims_[0]];
        if (col_major == false)
        {
            for (int i = 0; i < dims_[0]; i++)
            {
                userMap[i] = (rank_ % dims_[1]) + dims_[1] * i;
            }
        }
        else
        {
            for (int i = 0; i < dims_[0]; i++)
            {
                userMap[i] = (rank_ / dims_[0]) * dims_[0] + i;
            }
        }
        int ONE = 1;
        blacs_gridmap_(&colcomm_ctxt_, userMap, &dims_[0], &dims_[0], &ONE);
        int info;
        std::size_t nevex_ = nev_ + nex_;
        std::size_t bsize = 32;
        std::size_t nx_b = std::min(bsize, nevex_);
        t_descinit(desc1D_Nxnevx_, &N_, &nevex_, &mb_, &nx_b, &zero, &zero,
                   &colcomm_ctxt_, &m_, &info);
#endif

        comm_2 row_comm_dup;
        comm_2 col_comm_dup;
#if defined(HAS_NCCL)
        ncclUniqueId nccl_id, nccl_ids[nprocs_];
        ncclGetUniqueId(&nccl_id);
        MPI_Allgather(&nccl_id, sizeof(ncclUniqueId), MPI_UINT8_T, &nccl_ids[0],
                      sizeof(ncclUniqueId), MPI_UINT8_T, comm);

        for (auto i = 0; i < dims_[0]; i++)
        {
            if (coord_[0] == i)
            {
                ncclCommInitRank(&row_comm_dup, dims_[1], nccl_ids[i],
                                 coord_[1]);
            }
        }

        // col_comm
        ncclUniqueId nccl_id_2, nccl_ids_2[nprocs_];
        ncclGetUniqueId(&nccl_id_2);
        MPI_Allgather(&nccl_id_2, sizeof(ncclUniqueId), MPI_UINT8_T,
                      &nccl_ids_2[0], sizeof(ncclUniqueId), MPI_UINT8_T, comm);

        for (auto i = 0; i < dims_[1]; i++)
        {
            if (coord_[1] == i)
            {
                ncclCommInitRank(&col_comm_dup, dims_[0],
                                 nccl_ids_2[i * dims_[0]], coord_[0]);
            }
        }
#else
        MPI_Comm_dup(row_comm_, &row_comm_dup);
        MPI_Comm_dup(col_comm_, &col_comm_dup);
#endif
        mpi_wrapper_.add(row_comm_, row_comm_dup);
        mpi_wrapper_.add(col_comm_, col_comm_dup);

#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
    }

    //! A constructor of the class ChaseMpiProperties which distributes matrix
    //! `A` in `Block Distribution`.
    /*!
        It constructs a 2D grid of MPI ranks within the MPI communicator
       `comm_`.

        - The dimensions of this 2D grid is determined by `MPI_Cart_create`
       using all the available MPI nodes within `comm_`.
        - It distributes the Hermitian matrix `A` in a **Block-Dsitribution**
       scheme.

        This constructor requires the explicit values for the initalization of
       the size `N` of the matrix *A*, the number of sought after extremal
        eigenvalues `nev`, and the number of extra eigenvalue `nex` which
        defines, together with `nev`, the search space, and the working MPI
        communicator `comm_`.

        All the private members are either initialized
        directly by these parameters, or setup within the construction of this
       constructor.

        \param N Size of the square matrix defining the eigenproblem.
        \param nev Number of desired extremal eigenvalues.
        \param nex Number of eigenvalues augmenting the search space. Usually a
       relatively small fraction of `nev`. \param comm the working MPI
       communicator for ChASE.
     */
    ChaseMpiProperties(std::size_t N, std::size_t nev, std::size_t nex,
                       MPI_Comm comm)
        : N_(N), nev_(nev), nex_(nex), max_block_(nev + nex), comm_(comm)
    {
#ifdef USE_NSIGHT
        nvtxRangePushA("ChaseMpiProperties(Block-Block2): Init");
#endif
        data_layout = "Block-Block";

        MPI_Comm_size(comm, &nprocs_);
        MPI_Comm_rank(comm, &rank_);
        dims_[0] = dims_[1] = 0;
        MPI_Dims_create(nprocs_, 2, dims_);

        /* Set sharem memory communicator and ranks */
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm_);
        MPI_Comm_size(shm_comm_, &shm_nprocs_);
        MPI_Comm_rank(shm_comm_, &shm_rank_);

#ifdef HAS_CUDA

        int num_devices = -1;
        cudaGetDeviceCount(&num_devices);

        // Check the number of MPI ranks per node and the number of available GPUs
        if(num_devices < shm_nprocs_) {
            if (!rank_) {
                std::cerr << "Error! The number of MPI ranks per node ( " << shm_nprocs_ 
                        << " ) is larger than the number of available GPU devices ( " 
                        << num_devices << " )!" << std::endl;
                std::cerr << "The ChASE will terminate. Please, re-run ChASE with the number of " 
                        << "MPI ranks per node not larger than the number of available GPU devices."
                        << std::endl;
            }
        } else if (num_devices > shm_nprocs_) {
            cudaSetDevice(shm_rank_);
            if (!rank_) {
                std::cout << "Warning! The number of MPI ranks ( " << shm_nprocs_ 
                        << " ) is less than the number of available GPUs ( " << num_devices << " )! " << std::endl;
                std::cout << "The resources will not be used optimally. " 
                        << "The number of MPI ranks should be equal to the number of available GPU devices to exploit all computational resources!" 
                        << std::endl;
            }
        } else { // num_devices mpi_shm_rankmpi_shm_size
            cudaSetDevice(shm_rank_);
        }
#endif

        create2DGrid(dims_[0], dims_[1], false, comm, &row_comm_, &col_comm_,
                     &coord_[0], &coord_[1]);

        // size of local part of H
        int len;

        if (N_ % dims_[0] == 0)
        {
            len = N_ / dims_[0];
        }
        else
        {
            len = std::min(N_, N_ / dims_[0] + 1);
        }
        off_[0] = coord_[0] * len;

        if (coord_[0] < dims_[0] - 1)
        {
            m_ = len;
        }
        else
        {
            m_ = N_ - (dims_[0] - 1) * len;
        }

        if (N_ % dims_[1] == 0)
        {
            len = N_ / dims_[1];
        }
        else
        {
            len = std::min(N_, N_ / dims_[1] + 1);
        }
        off_[1] = coord_[1] * len;

        if (coord_[1] < dims_[1] - 1)
        {
            n_ = len;
        }
        else
        {
            n_ = N_ - (dims_[1] - 1) * len;
        }
        ldh_ = m_;

        nb_ = n_;
        if (coord_[1] == dims_[1] - 1 && dims_[1] != 1)
        {
            nb_ = (N_ - n_) / (dims_[1] - 1);
        }

        mb_ = m_;
        if (coord_[0] == dims_[0] - 1 && dims_[0] != 1)
        {
            mb_ = (N_ - m_) / (dims_[0] - 1);
        }

        mblocks_ = 1;
        nblocks_ = 1;

        irsrc_ = 0;
        icsrc_ = 0;

        r_offs_.reset(new std::size_t[1]());
        r_lens_.reset(new std::size_t[1]());
        r_offs_l_.reset(new std::size_t[1]());
        c_offs_.reset(new std::size_t[1]());
        c_lens_.reset(new std::size_t[1]());
        c_offs_l_.reset(new std::size_t[1]());

        r_offs_[0] = off_[0];
        r_lens_[0] = m_;
        r_offs_l_[0] = 0;
        c_offs_[0] = off_[1];
        c_lens_[0] = n_;
        c_offs_l_[0] = 0;

        block_counts_.resize(2);
        for (std::size_t dim_idx = 0; dim_idx < 2; dim_idx++)
        {
            block_counts_[dim_idx].resize(dims_[dim_idx]);
            for (std::size_t i = 0; i < dims_[dim_idx]; i++)
            {
                block_counts_[dim_idx][i] = 1;
            }
        }

        block_displs_.resize(2);
        block_lens_.resize(2);
        send_lens_.resize(2);
        g_offsets_.resize(2);

        for (std::size_t dim_idx = 0; dim_idx < 2; dim_idx++)
        {
            block_lens_[dim_idx].resize(dims_[dim_idx]);
            block_displs_[dim_idx].resize(dims_[dim_idx]);
            send_lens_[dim_idx].resize(dims_[dim_idx]);
            for (std::size_t i = 0; i < dims_[dim_idx]; ++i)
            {
                block_lens_[dim_idx][i].resize(1);
                block_displs_[dim_idx][i].resize(1);
                if (N_ % dims_[dim_idx] == 0)
                {
                    len = N_ / dims_[dim_idx];
                }
                else
                {
                    len = std::min(N_, N_ / dims_[dim_idx] + 1);
                }
                block_lens_[dim_idx][i][0] = len;
                block_displs_[dim_idx][i][0] = i * block_lens_[dim_idx][0][0];
                send_lens_[dim_idx][i] = len;
                g_offsets_[dim_idx].push_back(block_displs_[dim_idx][i][0]);
            }
            block_lens_[dim_idx][dims_[dim_idx] - 1].resize(1);
            block_displs_[dim_idx][dims_[dim_idx] - 1].resize(1);
            block_lens_[dim_idx][dims_[dim_idx] - 1][0] =
                N_ - (dims_[dim_idx] - 1) * len;
            block_displs_[dim_idx][dims_[dim_idx] - 1][0] =
                (dims_[dim_idx] - 1) * block_lens_[dim_idx][0][0];
            send_lens_[dim_idx][dims_[dim_idx] - 1] =
                N_ - (dims_[dim_idx] - 1) * len;
            g_offsets_[dim_idx].push_back(
                block_displs_[dim_idx][dims_[dim_idx] - 1][0]);
        }
#if defined(HAS_SCALAPACK)
        int zero = 0;
        int ictxt;
        blacs_get_(&zero, &zero, &ictxt);
        colcomm_ctxt_ = ictxt;
        int userMap[dims_[0]];
        for (int i = 0; i < dims_[0]; i++)
        {
            userMap[i] = (rank_ % dims_[1]) + dims_[1] * i;
        }

        int ONE = 1;
        blacs_gridmap_(&colcomm_ctxt_, userMap, &dims_[0], &dims_[0], &ONE);

        int info;
        std::size_t nevex_ = nev_ + nex_;
        std::size_t bsize = 32;
        std::size_t nx_b = std::min(bsize, nevex_);
        t_descinit(desc1D_Nxnevx_, &N_, &nevex_, &mb_, &nx_b, &zero, &zero,
                   &colcomm_ctxt_, &m_, &info);
#endif

        comm_2 row_comm_dup;
        comm_2 col_comm_dup;
#if defined(HAS_NCCL)
        ncclUniqueId nccl_id, nccl_ids[nprocs_];
        ncclGetUniqueId(&nccl_id);
        MPI_Allgather(&nccl_id, sizeof(ncclUniqueId), MPI_UINT8_T, &nccl_ids[0],
                      sizeof(ncclUniqueId), MPI_UINT8_T, comm);

        for (auto i = 0; i < dims_[0]; i++)
        {
            if (coord_[0] == i)
            {
                ncclCommInitRank(&row_comm_dup, dims_[1], nccl_ids[i],
                                 coord_[1]);
            }
        }

        // col_comm
        ncclUniqueId nccl_id_2, nccl_ids_2[nprocs_];
        ncclGetUniqueId(&nccl_id_2);
        MPI_Allgather(&nccl_id_2, sizeof(ncclUniqueId), MPI_UINT8_T,
                      &nccl_ids_2[0], sizeof(ncclUniqueId), MPI_UINT8_T, comm);

        for (auto i = 0; i < dims_[1]; i++)
        {
            if (coord_[1] == i)
            {
                ncclCommInitRank(&col_comm_dup, dims_[0],
                                 nccl_ids_2[i * dims_[0]], coord_[0]);
            }
        }
#else
        MPI_Comm_dup(row_comm_, &row_comm_dup);
        MPI_Comm_dup(col_comm_, &col_comm_dup);
#endif
        mpi_wrapper_.add(row_comm_, row_comm_dup);
        mpi_wrapper_.add(col_comm_, col_comm_dup);

#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
    }

    Comm_t get_mpi_wrapper() { return mpi_wrapper_; }
#if defined(HAS_SCALAPACK)
    int get_colcomm_ctxt() { return colcomm_ctxt_; }

    std::size_t* get_desc1D_Nxnevx() { return desc1D_Nxnevx_; }
#endif
    //! Returns the rank of matrix `A` which is distributed within 2D MPI grid.
    /*!
        \return `N_`: the rank of matrix `A`.
     */
    std::size_t get_N() { return N_; };
    //! Return leading dimension of local part of Symmetric/Hermtian matrix on
    //! each MPI proc.
    /*! \return `ldh_`, a private member of this class.
     */
    std::size_t get_ldh() { return ldh_; };

    //! Returns column number of the local matrix on each MPI node.
    /*!
        \return `n_`: the column number of the local matrix on each MPI node.
     */
    std::size_t get_n() { return n_; };

    //! Returns row number of the local matrix.
    /*!
        \return `m_`: Row number of the local matrix on each MPI node.
     */
    std::size_t get_m() { return m_; };

    //! Returns the maximum column number of rectangular matrix `V`.
    /*!
        \return `max_block_`: Maximum column number of matrix `V`.
     */
    std::size_t get_max_block() { return max_block_; };

    //! Returns number of desired extremal eigenpairs, which was set by users.
    /*!
        \return `nev_`: Number of desired extremal eigenpairs.
     */
    std::size_t GetNev() { return nev_; };

    //! Returns the Increment of the search subspace so that its total size,
    //! which was set by users within chase::ChaseConfig.
    /*!
        \return `nex_`: Increment of the search subspace.
     */
    std::size_t GetNex() { return nex_; };

    //! Returns the column blocking factor, this function is useful for ChASE
    //! with *Block-Cyclic Distribution*.
    /*!
        \return `nb_`: Column blocking factor.
     */
    std::size_t get_nb() { return nb_; };

    //! Returns the row blocking factor, this function is useful for ChASE with
    //! *Block-Cyclic Distribution*.
    /*!
        \return `mb_`: Row blocking factor.
     */
    std::size_t get_mb() { return mb_; };

    /*!
        \return `row_comm_`: the row communicator within 2D MPI grid.
     */
    MPI_Comm get_row_comm() { return row_comm_; }

    /*!
        \return `col_comm_`: the column communicator within 2D MPI grid.
     */
    MPI_Comm get_col_comm() { return col_comm_; }

    //! Returns the dimension of cartesian communicator grid
    /*!
        \return `dims_`: an array of size 2 which store the dimensions of the
       constructed 2D MPI grid.
     */
    int* get_dims() { return dims_; }

    //! Returns offset of row and column of the local matrix on each MPI node
    //! regarding the index of row and column of global matrix `A`.
    /*! For example, for a global matrix `A`, `A_sub` is a submatrix of `A` of
      size `m * n`, which is extracted out of `A` starting from the row indexing
      `i` and column indexing `j`. This member function returns `i` and `j` as
      an array of size 2 for the submatrix on each MPI rank.
    */
    //! This member function is mostly used for ChASE with *Block-Distribution*.
    /*!
      \return `off_`: an array of size 2 which stores the offset of row and
      column of the local matrix on each MPI node regarding the index of row and
      column of global matrix `A`.
    */
    std::size_t* get_off() { return off_; }

    //! Return the offset and length along the two dimensions of the local
    //! matrix on each MPI node regarding the global index of matrix `A`.
    /*! For example, for a global matrix `A`, `A_sub` is a submatrix of `A` of
      size `m * n`, which is extracted out of `A` starting from the row indexing
      `i` and column indexing `j`. This member function outputs `xoff=i`,
      `yoff=j`, `xlen=m`, and `ylen=n` for the submatrix on each MPI rank.
    */
    //! This member function is mostly used for ChASE with *Block-Distribution*.
    /*!
      @param[in/out] `xoff` -> the offset of row of the local matrix on each MPI
      node regarding the row index of matrix `A`.
      @param[in/out] `yoff` -> the offset of column of the local matrix on each
      MPI node regarding the column index of matrix `A`.
      @param[in/out] `xlen` -> the number of row of the local matrix on each MPI
      node.
      @param[in/out] `ylen` -> the number of column of the local matrix on each
      MPI node.
    */
    void get_off(std::size_t* xoff, std::size_t* yoff, std::size_t* xlen,
                 std::size_t* ylen)
    {
        *xoff = off_[0];
        *yoff = off_[1];
        *xlen = m_;
        *ylen = n_;
    }

    //! This member function is mostly used for ChASE with *Block-Cyclic
    //! Distribution*. The number of submatrix on each MPI node is `mblocks_ *
    //! nblocks_`.
    /*!
      \return `mblocks_`: the number of submatrices in the local matrix on each
      MPI node along the row direction.
    */
    std::size_t get_mblocks() { return mblocks_; }

    //! This member function is mostly used for ChASE with *Block-Cyclic
    //! Distribution*. The number of submatrix on each MPI node is `mblocks_ *
    //! nblocks_`.
    /*!
      \return `nblocks_`: the number of submatrices in the local matrix on each
      MPI node along the column direction.
    */
    std::size_t get_nblocks() { return nblocks_; }

    //! This member function only matters for the *Block-Cyclic Distribution*.
    /*!
      \return `irsrc_`: the process row within 2D MPI grid over which the first
      row of the global matrix `A` is distributed.
    */
    int get_irsrc() { return irsrc_; }

    //! This member function only matters for the *Block-Cyclic Distribution*.
    /*!
      \return `icsrc_`: the process column within 2D MPI grid over which the
      first column of the global matrix `A` is distributed.
    */
    int get_icsrc() { return icsrc_; }

    //! This member function only matters for the *Block-Cyclic Distribution*.
    /*!
      \return `r_offs_.get()`: the pointer to store offset of each subblock in
      local matrix along the row direction regarding the row direction of global
      matrix `A`.
    */
    std::size_t* get_row_offs() { return r_offs_.get(); }

    //! This member function only matters for the *Block-Cyclic Distribution*.
    /*!
      \return `r_lens_.get()`: the pointer to store number of row of each
      subblock in local matrix along the row direction regarding the row
      direction of global matrix `A`.
    */
    std::size_t* get_row_lens() { return r_lens_.get(); }

    //! This member function only matters for the *Block-Cyclic Distribution*.
    /*!
      \return `r_offs_l_.get()`: the pointer to store offset of each subblock in
      local matrix along the row direction regarding the row direction of local
      matrix.
    */
    std::size_t* get_row_offs_loc() { return r_offs_l_.get(); }

    //! This member function only matters for the *Block-Cyclic Distribution*.
    /*!
      \return `c_offs_.get()`: the pointer to store offset of each subblock of
      local matrix along the column direction regarding the column direction of
      global matrix `A`.
    */
    std::size_t* get_col_offs() { return c_offs_.get(); }

    //! This member function only matters for the *Block-Cyclic Distribution*.
    /*!
      \return `c_lens_.get()`: the pointer to store length of each subblock of
      local matrix along the column direction regarding the column direction of
      global matrix `A`.
    */
    std::size_t* get_col_lens() { return c_lens_.get(); }

    //! This member function only matters for the *Block-Cyclic Distribution*.
    /*!
      \return `c_offs_l_.get()`: the pointer to store offset of each subblock in
      local matrix along the column direction regarding the column direction of
      local matrix.
    */
    std::size_t* get_col_offs_loc() { return c_offs_l_.get(); }

    //! Return the pointers to `r_offs_`, `r_lens_`, `r_offs_l_`, `c_offs_`,
    //! `c_lens_` and `c_offs_l_` in single member function.
    //! This member function only matters for the *Block-Cyclic Distribution*.
    /*!
      @param[in/out] `r_offs` -> the pointer to store offset of each subblock of
      local matrix along the row direction regarding the global indexing of
      matrix A. Its size is `mblocks_`, which can be obtained by get_mblocks().
      @param[in/out] `r_lens` -> the pointer to store length of each subblock of
      local matrix along the row direction regarding the global indexing of
      matrix A. Its size is `mblocks_`, which can be obtained by get_mblocks().
      @param[in/out] `r_offs_l` -> the pointer to store offset of each subblock
      of local matrix along the row direction regarding the indexing of local
      matrix. Its size is `mblocks_`, which can be obtained by get_mblocks().
      @param[in/out] `c_offs` -> the pointer to store offset of each subblock of
      local matrix along the column direction regarding the global indexing of
      matrix A. Its size is `nblocks_`, which can be obtained by get_nblocks().
      @param[in/out] `c_lens` -> the pointer to store length of each subblock of
      local matrix along the column direction regarding the global indexing of
      matrix A. Its size is `nblocks_`, which can be obtained by get_nblocks().
      @param[in/out] `c_offs_l` -> the pointer to store offset of each subblock
      of local matrix along the column direction regarding the indexing of local
      matrix. Its size is `nblocks_`, which can be obtained by get_nblocks().

    */
    void get_offs_lens(std::size_t*& r_offs, std::size_t*& r_lens,
                       std::size_t*& r_offs_l, std::size_t*& c_offs,
                       std::size_t*& c_lens, std::size_t*& c_offs_l)
    {
        r_offs = r_offs_.get();
        r_lens = r_lens_.get();
        r_offs_l = r_offs_l_.get();
        c_offs = c_offs_.get();
        c_lens = c_lens_.get();
        c_offs_l = c_offs_l_.get();
    }
    // get the distribution layout of Hermtian/symmetric matrix
    /*!
     *  - `Block-Block`
     *  - `Block-Cyclic`
     * */
    std::string get_dataLayout() { return data_layout; }

    // coordinates in the cartesian communicator grid
    /*!
      \return `coord_`: the coordinates of each MPI rank in the cartesian
      communicator grid.
    */
    int* get_coord() { return coord_; }

    // some nice utility functions
    /*!
      \return `n_`: the leading dimension of the local matrix of `A` in the row
      dimension.
    */
    std::size_t get_ldb() { return n_; };

    /*!
      \return `m_`: the leading dimension of the local matrix of `A` in the
      column dimension.
    */
    std::size_t get_ldc() { return m_; };

    /*!
      \return `block_counts_`: 2D array which stores the block number of local
      matrix on each MPI node in each dimension.
    */
    const std::vector<std::vector<int>>& get_blockcounts()
    {
        return block_counts_;
    }

    /*!
       \return `block_lens_`: 3D array which stores the size of each sublock on
       each MPI node in each dimension.
     */

    const std::vector<std::vector<std::vector<int>>>& get_blocklens()
    {
        return block_lens_;
    }

    /*!
      \return `block_displs_`: 3D array which stores the offset of each sublock
      on each MPI node in each dimension.
    */
    const std::vector<std::vector<std::vector<int>>>& get_blockdispls()
    {
        return block_displs_;
    }

    /*!
     \return `send_lens_`: 2D array which stores the size of local matrix on
     each MPI node in each dimension.
   */

    const std::vector<std::vector<int>>& get_sendlens() { return send_lens_; }

    /*!
     \return `g_offsets_`: 2D array which stores the offset of the first block
     (the most up and left one) on each MPI node in each dimension regarding the
     global indexing of matrix A.
   */
    const std::vector<std::vector<int>>& get_g_offsets() { return g_offsets_; }

    //! Returns the total number of MPI nodes within MPI communicator where
    //! ChASE is working on.
    /*!
        \return `nprocs_`: the total number of MPI nodes within the root MPI
       communicator.
     */
    int get_nprocs() { return nprocs_; }

    //! Returns the total number of MPI nodes within the shared-memory communicator.
    /*!
        \return `shm_nprocs_`: the total number of MPI nodes within the shared-memory communicator.
     */
    int get_shm_nprocs() { return shm_nprocs_; }

    //! Returns the rank of MPI node within 2D grid.
    /*!
        \return `rank_`: the rank of MPI node within 2D grid.
     */
    int get_my_rank() { return rank_; }

    //! Returns the rank of MPI node within the shared-memory communicator.
    /*!
        \return `shm_rank_`: the rank of MPI node within the shared-memory communicator.
     */
    int get_my_shm_rank() { return shm_rank_; }

    //! Create a ChaseMpiMatrices object which stores the operating matrices and
    //! vectors for ChASE-MPI.
    /*!
      @param H a `m_ * n_` rectangular matrix, with leading dimension `ldh`
      @param ldh leading dimension of `H`, which should be `>=m_`
      @param V1 a `m_ * max_block_` rectangular matrix.
      @param ritzv a `max_block_` vector which stores the computed Ritz values.
      @param V2 a `n_ * max_block_` rectangular matrix.
      @param resid a `max_block_` vector which stores the residual of each
      computed Ritz value.
    */
    ChaseMpiMatrices<T> create_matrices(int mode, T* H, std::size_t ldh,
                                        T* V1 = nullptr,
                                        Base<T>* ritzv = nullptr) const
    {
        return ChaseMpiMatrices<T>(mode, comm_, N_, m_, n_, max_block_, H, ldh,
                                   V1, ritzv);
    }

private:
    ///////////////////////////////////////////////////
    // General parameters of the eigenproblem
    //////////////////////////////////////////////////

    //! Global size of the matrix *A* defining the eigenproblem.
    /*!    This variable is initialized by the constructor using the value of
       the first of its input parameter `N`.

       This variable is private, it can be
       accessed by the member function get_N().
     */
    std::size_t N_;
    //! The leading dimension of local part of Symmetric/Hermtian matrix on each
    //! MPI proc. It should be `>= m_`.
    std::size_t ldh_;
    //! Number of desired extremal eigenpairs
    /*!
        This variable is initialized by the constructor using the value
        of its input parameter `nev`.

        This variable is private, it can be accessed by the member function
       GetNev().
     */
    std::size_t nev_;

    //! Increment of the search subspace so that its total size is `nev + nex`.
    /*!
        This variable is initialized by the constructor using the value
        of its input parameter `nex`.

        This variable is private, it can be accessed by the member function
       GetNex().
     */
    std::size_t nex_;

    /*!
        This variable is initialized by the constructor using the **sum**
        of its input parameter `nex` and `nev`. Thus we have
       `max_block_=nev_+nex_`. This variable is private, it can be accessed by
       the member function get_max_block().
     */
    std::size_t max_block_;

    //! Column number of the local matrix on each MPI node.
    /*!
          - For *Block Distribution*, this variable is initialized based on the
       global size of matrix `A` and dimension of the column of 2D MPI grid.
          - For *Block-Cyclic Distribution*, it is determined also by block size
          of submatrices `nb_`, etc.
          This variable is private, it can be accessed by the member function
       get_n().
     */
    std::size_t n_;

    //! Row number of the local matrix on each MPI node.
    /*!
          - For *Block Distribution*, this variable is initialized based on the
       global size of matrix `A` and dimension of the column of 2D MPI grid.
          - For *Block-Cyclic Distribution*, it is determined also by block size
          of submatrices `mb_`, etc.
          This variable is private, it can be accessed by the member function
       get_m().
     */
    std::size_t m_;

    //! Column blocking factor.
    /*!
          - For *Block Distribution*, this variable equals to the variable `n_`.
          - For *Block-Cyclic Distribution*, it is initialized by the parameter
       of the constructor `nb`, This variable is private, it can be accessed by
       the member function get_nb().
     */
    std::size_t nb_;

    //! Row blocking factor.
    /*!
          - For *Block Distribution*, this variable equals to the variable `m_`.
          - For *Block-Cyclic Distribution*, it is initialized by the parameter
       of the constructor `mb`, This variable is private, it can be accessed by
       the member function get_mb().
     */
    std::size_t mb_;

    //! Number of submatrices along the column direction in the local matrix.
    //! Thus each local matrix is constructed by `mblocks_ * nblocks_` blocks.
    /*!
          - For *Block Distribution*, this variable equals to 1.
          - For *Block-Cyclic Distribution*, it is initialized during the
       construction of Block-Cyclic scheme, This variable is private, it can be
       accessed by the member function get_nblocks().
     */
    std::size_t nblocks_;

    //! Number of submatrices along the row direction in the local matrix. Thus
    //! each local matrix is constructed by `mblocks_ * nblocks_` blocks.
    /*!
          - For *Block Distribution*, this variable equals to 1.
          - For *Block-Cyclic Distribution*, it is initialized during the
       construction of Block-Cyclic scheme, This variable is private, it can be
       accessed by the member function get_mblocks().
     */
    std::size_t mblocks_;

    //! Process row over which the first row of the global matrix A is
    //! distributed.
    /*! This variable matters only for the *Block-Cyclic Distribution*, it is
       initialized by the parameter irsrc of the constructor. This variable is
       private, it can be accessed by the member function get_irsrc().
    */
    int irsrc_;

    //! Process column over which the first column of the global matrix A is
    //! distributed.
    /*! This variable matters only for the *Block-Cyclic Distribution*, it is
       initialized by the parameter irsrc of the constructor. This variable is
       private, it can be accessed by the member function get_icsrc().
    */
    int icsrc_;

    //! Offset of each subblock (especially for *Block-Cyclic Distribution*)
    //! along the row direction regarding the global indexing of matrix `A`.
    /*! This variable matters only for the *Block-Cyclic Distribution*, it is
       initialized during the setup of the *Block-Cyclic Distribution* scheme.
       This variable is private, it can be accessed by the member function
       get_row_offs(). The size of this array is `mblocks_`.
    */
    std::unique_ptr<std::size_t[]> r_offs_;

    //! Length of each subblock (especially for *Block-Cyclic Distribution*)
    //! along the row direction regarding the global indexing of matrix `A`.
    /*! This variable matters only for the *Block-Cyclic Distribution*, it is
       initialized during the setup of the *Block-Cyclic Distribution* scheme.
       This variable is private, it can be accessed by the member function
       get_row_lens(). The size of this array is `mblocks_`.
    */
    std::unique_ptr<std::size_t[]> r_lens_;

    //! Offset of each subblock (especially for *Block-Cyclic Distribution*)
    //! along the row direction regarding the indexing of local matrix on each
    //! MPI node.
    /*! This variable matters only for the *Block-Cyclic Distribution*, it is
       initialized during the setup of the *Block-Cyclic Distribution* scheme.
       This variable is private, it can be accessed by the member function
       get_row_offs_loc(). The size of this array is
       `mblocks_`.
    */
    std::unique_ptr<std::size_t[]> r_offs_l_;

    //! Offset of each subblock (especially for *Block-Cyclic Distribution*)
    //! along the column direction regarding the global indexing of matrix `A`.
    /*! This variable matters only for the *Block-Cyclic Distribution*, it is
       initialized during the setup of the *Block-Cyclic Distribution* scheme.
       This variable is private, it can be accessed by the member function
       get_col_offs(). The size of this array is `nblocks_`.
    */
    std::unique_ptr<std::size_t[]> c_offs_;

    //! Length of each subblock (especially for *Block-Cyclic Distribution*)
    //! along the column direction regarding the global indexing of matrix `A`.
    /*! This variable matters only for the *Block-Cyclic Distribution*, it is
       initialized during the setup of the *Block-Cyclic Distribution* scheme.
       This variable is private, it can be accessed by the member function
       get_col_lens(). The size of this array is `nblocks_`.
    */
    std::unique_ptr<std::size_t[]> c_lens_;

    //! Offset of each subblock (especially for *Block-Cyclic Distribution*)
    //! along the column direction regarding the indexing of local matrix on
    //! each MPI node.
    /*! This variable matters only for the *Block-Cyclic Distribution*, it is
       initialized during the setup of the *Block-Cyclic Distribution* scheme.
       This variable is private, it can be accessed by the member function
       get_col_offs_loc(). The size of this array is
       `nblocks_`.
    */
    std::unique_ptr<std::size_t[]> c_offs_l_;

    //! 2D array which stores the block number of local matrix on each MPI node
    //! in each dimension.
    /*! - This variable is equally shared by all the MPI nodes, which make the
       *Block-Cyclic* scheme within each MPI node be visible to all other MPI
       nodes.
        - The variable is especically useful for
        the MPI communication in the case that global matrix cannot be equally
       distributed to each MPI node.
        - For example, the block number within rank `i` (this rank is within
       row_comm_) along the row dimension is `block_counts_[0][i]`, and the
       block number within rank `j` (this rank is within col_comm_) along the
       column dimension is `block_counts_[1][j]`. For *Block-Distribution*, all
       the values in this 2D array equal to **1**.

        This variable is private, it can be accessed by the member function
       get_blockcounts().
    */
    std::vector<std::vector<int>> block_counts_;

    //! 3D array which stores the length of each sublock on each MPI node in
    //! each dimension.
    /*! - This variable is equally shared by all the MPI nodes, which make the
       *Block-Cyclic* scheme within each MPI node be visible to all other MPI
       nodes.
        - The variable is especically useful for
        the MPI communication in the case that global matrix cannot be equally
       distributed to each MPI node.
        - For example, the length of the block numbering `k` *th* within rank
       `i` (this rank is within row_comm_) along the row dimension is
       `block_lens_[0][i][k]`, and the length of the block numbering `k` *th*
       the block number within rank `j` (this rank is within col_comm_) along
       the column dimension is `block_lens_[1][j][k]`.

        This variable is private, it can be accessed by the member function
       get_blocklens().
    */
    std::vector<std::vector<std::vector<int>>> block_lens_;

    //! 3D array which stores the offset of each sublock on each MPI node in
    //! each dimension.
    /*! - This variable is equally shared by all the MPI nodes, which make the
       *Block-Cyclic* scheme within each MPI node be visible to all other MPI
       nodes.
        - The variable is especically useful for
        the MPI communication in the case that global matrix cannot be equally
       distributed to each MPI node.
        - For example, the offset of the block numbering `k` *th* within rank
       `i` (this rank is within row_comm_) along the row dimension is
       `block_displs_[0][i][k]`, and the offset of the block numbering `k` *th*
       the block number within rank `j` (this rank is within col_comm_) along
       the column dimension is `block_displs_[1][j][k]`.

        This variable is private, it can be accessed by the member function
       get_blockdispls().
    */
    std::vector<std::vector<std::vector<int>>> block_displs_;

    //! 2D array which stores the length of local matrix on each MPI node in
    //! each dimension.
    /*! - This variable is equally shared by all the MPI nodes, which make the
       *Block-Cyclic* scheme within each MPI node be visible to all other MPI
       nodes.
        - The variable is especically useful for
        the MPI communication in the case that global matrix cannot be equally
       distributed to each MPI node.
        - For example, the length of local matrix within rank `i` (this rank is
       within row_comm_) along the row dimension is `send_lens_[0][i]`, and the
       block number within rank `j` (this rank is within col_comm_) along the
       column dimension is `send_lens_[1][j]`.

        This variable is private, it can be accessed by the member function
       get_sendlens().
    */
    std::vector<std::vector<int>> send_lens_;

    //! 2D array which stores the offset of the first block (the most up and
    //! left one) on each MPI node in each dimension regarding the global
    //! indexing of matrix `A`.
    /*! - This variable is equally shared by all the MPI nodes, which make the
       *Block-Cyclic* scheme within each MPI node be visible to all other MPI
       nodes.
        - The variable is especically useful for
        the MPI communication in the case that global matrix cannot be equally
       distributed to each MPI node.
        - For example, the offset of first block of local matrix within rank `i`
       (this rank is within row_comm_) along the row dimension is
       `g_offsets_[0][i]`, and the block number within rank `j` (this rank is
       within col_comm_) along the column dimension is `g_offsets_[1][j]`.

        This variable is private, it can be accessed by the member function
       get_g_offsets().
    */
    std::vector<std::vector<int>> g_offsets_;

    //! The MPI communicator which ChASE is working on.
    /*!
        This variable is initialized by the constructor using the value
        of its input parameters `comm`.
     */
    MPI_Comm comm_;

    //! The shared-memory MPI communicator which ChASE is working on.
    /*!
        This variable is initialized by the constructor using the value
        of its input parameters `shm_comm`.
     */
    MPI_Comm shm_comm_;

    //! Total number of MPI nodes in the MPI communicator which ChASE is working
    //! on.
    int nprocs_;

    //! Total number of MPI ranks in the shared-memory communicator (ie. #mpi ranks per node).
    int shm_nprocs_;

    //! The rank of each MPI node within the MPI communicator which ChASE is
    //! working on.
    int rank_;

    //! The rank of each MPI within the shared-memory MPI communicator.
    int shm_rank_;

    //! The row communicator of the constructed 2D grid of MPI codes.
    /*!
        This variable is initialized in the constructor, after the construction
       of MPI 2D grid. This variable is private, it can be accessed by the
       member function get_row_comm().
     */
    MPI_Comm row_comm_;

    //! The column communicator of the constructed 2D grid of MPI codes.
    /*!
        This variable is initialized in the constructor, after the construction
       of MPI 2D grid. This variable is private, it can be accessed by the
       member function get_col_comm().
     */
    MPI_Comm col_comm_;

    //! The array with two elements determines the dimension of 2D grid of MPI
    //! nodes.
    /*!
        - For *Block Distribution*,
            1. it can be initialized by `MPI_Dims_create` which creates a
       division of MPI ranks in a cartesian grid.
            2. it can be initialized by users with the input parameters,
       paramters `row_dim` and `col_dim`. More precise, we have `dims_[0] =
       row_dim` and `dims_[1] = col_dim`.
        - For *Block-Cyclic Distribution*, it is always initialized by the input
       paramters `row_dim` and `col_dim`. More precise, we have `dims_[0] =
       row_dim` and `dims_[1] = col_dim`.
     */
    int dims_[2];

    //! The array with two elements indicates the coordinates of each MPI node
    //! within the 2D grid of MPI nodes.
    /*!
        This variable is determined by the properties of 2D grid of MPI node.
     */
    int coord_[2];

    //! The array with two elements indicates the offset of the local matrix on
    //! each MPI node regarding the global index of matrix `A`.
    /*!
        This variable is determined by the properties of 2D grid of MPI node,
       the size of matrix and the scheme of distribution across the 2D grid.
     */
    std::size_t off_[2];

    // A std::string which indicates the data distribution scheme.
    /*!
     *  - `Block-Block`
     *  - `Block-Cyclic`
     * */
    std::string data_layout;
#if defined(HAS_SCALAPACK)
    //! ScaLAPACK context for each row communicator. This context is in 1D grid,
    //! with only 1 column
    int colcomm_ctxt_;
    std::size_t desc1D_Nxnevx_[9];
#endif

    Comm_t mpi_wrapper_;
};
} // namespace mpi
} // namespace chase
