/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2018, Simulation Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany
// and
// Copyright (c) 2016-2018, Aachen Institute for Advanced Study in Computational
//   Engineering Science, RWTH Aachen University, Germany All rights reserved.
// License is 3-clause BSD:
// https://github.com/SimLabQuantumMaterials/ChASE/

#pragma once

#include <mpi.h>
#include <memory>
#include <vector>
#include <tuple>

#include "algorithm/types.hpp"
#include "chase_mpi_matrices.hpp"

namespace chase {
namespace mpi {

/*
 * Logic for an MPI parallel HEMM
 */

template <typename T>
MPI_Datatype getMPI_Type();

template <>
MPI_Datatype getMPI_Type<float>() {
  return MPI_FLOAT;
}

template <>
MPI_Datatype getMPI_Type<double>() {
  return MPI_DOUBLE;
}

template <>
MPI_Datatype getMPI_Type<std::complex<float> >() {
  return MPI_COMPLEX;
}

template <>
MPI_Datatype getMPI_Type<std::complex<double> >() {
  return MPI_DOUBLE_COMPLEX;
}

std::pair<std::size_t, std::size_t> numroc(std::size_t n, std::size_t nb, int iproc, int isrcproc, int nprocs){

    std::size_t numroc;
    std::size_t extrablks, mydist, nblocks;
    mydist = (nprocs + iproc - isrcproc) % nprocs;
    nblocks = n / nb;
    numroc = (nblocks / nprocs) * nb;
    extrablks = nblocks % nprocs;

    if(mydist < extrablks)
        numroc = numroc + nb;
    else if(mydist == extrablks)
        numroc = numroc + n % nb;

    std::size_t nb_loc = numroc / nb;
    
    if(numroc % nb != 0){
        nb_loc += 1;
    }

    return std::make_pair(numroc, nb_loc);
}

template <class T>
class ChaseMpiProperties {
 public:

    //row_dim = number of row in the grid
    //col_dim = number of col in the grid
    ChaseMpiProperties(std::size_t N, std::size_t mb, std::size_t nb, std::size_t nev,
                  std::size_t nex, int row_dim, int col_dim, char *grid_major, int irsrc, int icsrc, MPI_Comm comm)
      : N_(N), mb_(mb), nb_(nb), nev_(nev), nex_(nex), max_block_(nev + nex), irsrc_(irsrc), icsrc_(icsrc), comm_(comm) {
   
	std::size_t blocknb[2];
        std::size_t N_loc[2];	
	std::size_t blocksize[2];
	blocksize[0] = mb_;
        blocksize[1] = nb_;	
    
	int tmp_dims_[2];
    	dims_[0] = row_dim;
	dims_[1] = col_dim;

        bool col_major = false;

    	if(strcmp (grid_major, "C") == 0){
    	    col_major = true;
	}
	
        if(col_major){
            tmp_dims_[1] = row_dim;
            tmp_dims_[0] = col_dim;
        }else{    
    	    tmp_dims_[0] = row_dim;
	    tmp_dims_[1] = col_dim;
        }

	int isrc[2];
	isrc[0] = irsrc;
        isrc[1] = icsrc;

        int periodic[] = {0, 0};
        int reorder = 0;
        int free_coords[2];
        int row_procs, col_procs;
        int tmp_coord[2];

	MPI_Comm cartComm;

        MPI_Cart_create(comm, 2, tmp_dims_, periodic, reorder, &cartComm);
    
	MPI_Comm_size(cartComm, &nprocs_);
       	MPI_Comm_rank(cartComm, &rank_);
    	MPI_Cart_coords(cartComm, rank_, 2, tmp_coord);

    	if(col_major){
            coord_[1] = tmp_coord[0];
            coord_[0] = tmp_coord[1];	
    	}else{
            coord_[1] = tmp_coord[1];
            coord_[0] = tmp_coord[0];    
        }

        if (nprocs_ > N_) throw std::exception();

        // row communicator
        if(col_major){
            free_coords[0] = 1;
            free_coords[1] = 0;
        }else{
            free_coords[0] = 0;
            free_coords[1] = 1;    
        }

        MPI_Cart_sub(cartComm, free_coords, &row_comm_);
        MPI_Comm_size(row_comm_, &row_procs);

        // column communicator
    	if(col_major){
            free_coords[0] = 0;
            free_coords[1] = 1;
        }else{
            free_coords[0] = 1;
            free_coords[1] = 0;
        }

        MPI_Cart_sub(cartComm, free_coords, &col_comm_);
        MPI_Comm_size(col_comm_, &col_procs);    

        int myrow = coord_[0];
        int mycol = coord_[1];

	for (std::size_t dim_idx = 0; dim_idx < 2; dim_idx++) {
	    std::tie(N_loc[dim_idx], blocknb[dim_idx]) = numroc( N_, blocksize[dim_idx], coord_[dim_idx], isrc[dim_idx], dims_[dim_idx]);
	}

	mblocks_ = blocknb[0];
        nblocks_ = blocknb[1];	
	m_ = N_loc[0];
	n_ = N_loc[1];

#ifdef CHASE_OUTPUT
	std::cout << grid_major << " " << dims_[0] <<"x"<< dims_[1] << ", " << "rank: " << rank_ << " (" << coord_[0] << ","
	    << coord_[1] << "), row_comm_size = " << row_procs << ", col_comm_size = " << col_procs << ", local matrix size = "
	    << m_ << "x" << n_ << ", num of blocks in local = " << mblocks_ << "x" << nblocks_ << std::endl;
#endif

	r_offs_.reset(new std::size_t[mblocks_]());
        r_lens_.reset(new std::size_t[mblocks_]());
        r_offs_l_.reset(new std::size_t[mblocks_]());
        c_offs_.reset(new std::size_t[nblocks_]());
        c_lens_.reset(new std::size_t[nblocks_]());
        c_offs_l_.reset(new std::size_t[nblocks_]());

	//get offsets
	std::size_t sendr = irsrc_;
        std::size_t sendc = icsrc_;	

	int cnt = 0;
        std::size_t nr, nc;

	//for the first dimension of grid
        for(std::size_t r = 0; r < N_; r += mb_, sendr = (sendr + 1) % dims_[0]){
            nr = mb_;
            if (N_ - r < mb_){
                nr = N_ - r;
            }

  	    if(coord_[0] == sendr){
                r_offs_[cnt] = r;
                r_lens_[cnt] = nr;
                cnt++;
            }
        }

        cnt = 0;
        //for the second dimension of grid
        for(std::size_t c = 0; c < N_; c += nb_, sendc = (sendc + 1) % dims_[1]){
	    nc = nb_;
	    if(N_ - c < nb_){
	        nc = N_ - c;
	    }
	    if(coord_[1] == sendc){
                c_offs_[cnt] = c;
                c_lens_[cnt] = nc;
	        cnt++;	
	    }	    
        }	

       cnt = 0;
       r_offs_l_[0] = 0;
       c_offs_l_[0] = 0;

       for(std::size_t i = 1; i < mblocks_; i++){
           r_offs_l_[i] = r_offs_l_[i - 1] + r_lens_[i - 1];
       }

       for(std::size_t j = 1; j < nblocks_; j++){
           c_offs_l_[j] = c_offs_l_[j - 1] + c_lens_[j - 1];
       } 

       H_.reset(new T[n_ * m_]());
       B_.reset(new T[n_ * max_block_]());
       C_.reset(new T[m_ * max_block_]());
       IMT_.reset(new T[std::max(n_, m_) * max_block_]());

       //for MPI communication
       block_counts_.resize(2);
       send_lens_.resize(2);

       for (std::size_t dim_idx = 0; dim_idx < 2; dim_idx++) {
           block_counts_[dim_idx].resize(dims_[dim_idx]);
           send_lens_[dim_idx].resize(dims_[dim_idx]);	   
	   for(std::size_t i = 0; i < dims_[dim_idx]; i++){
		   std::tie(send_lens_[dim_idx][i], block_counts_[dim_idx][i]) = numroc( N_, blocksize[dim_idx], i, isrc[dim_idx], dims_[dim_idx]);
           }
       }

       block_displs_.resize(2);
       block_lens_.resize(2);
       std::size_t send[2];
       send[0] = irsrc_;
       send[1] = icsrc_;
       g_offsets_.resize(2);

       for (std::size_t dim_idx = 0; dim_idx < 2; dim_idx++) {
           block_lens_[dim_idx].resize(dims_[dim_idx]);
           block_displs_[dim_idx].resize(dims_[dim_idx]);
           
           int displs_cnt = 0;
  
	   for(std::size_t i = 0; i < dims_[dim_idx]; i++){
	       block_lens_[dim_idx][i].resize(block_counts_[dim_idx][i]);
               block_displs_[dim_idx][i].resize(block_counts_[dim_idx][i]);
               nr = 0; cnt = 0; send[0] = irsrc_; send[1] = icsrc_;
	       for(std::size_t r = 0; r < N_; r += blocksize[dim_idx], send[dim_idx] = (send[dim_idx] + 1) % dims_[dim_idx]){
	           nr = blocksize[dim_idx];
		   if(N_ - r < blocksize[dim_idx]){
		       nr = N_ - r;
		   }
		   if(i == send[dim_idx]){
		       block_lens_[dim_idx][i][cnt] = nr;
                       block_displs_[dim_idx][i][cnt] = r;
                       g_offsets_[dim_idx].push_back(r);

		       cnt++;		       
		   }
	       }
	   }
       }
       
    }


    ChaseMpiProperties(std::size_t N, std::size_t nev, std::size_t nex,
                     MPI_Comm comm = MPI_COMM_WORLD)
      : N_(N), nev_(nev), nex_(nex), max_block_(nev + nex), comm_(comm) {
    int periodic[] = {0, 0};
    int reorder = 0;
    int free_coords[2];
    MPI_Comm cartComm;

    // create cartesian communicator
    MPI_Comm_size(comm, &nprocs_);
    dims_[0] = dims_[1] = 0;
    MPI_Dims_create(nprocs_, 2, dims_);
    MPI_Cart_create(comm, 2, dims_, periodic, reorder, &cartComm);
    MPI_Comm_size(cartComm, &nprocs_);
    MPI_Comm_rank(cartComm, &rank_);
    MPI_Cart_coords(cartComm, rank_, 2, coord_);

    if (nprocs_ > N_) throw std::exception();

    // row communicator
    free_coords[0] = 0;
    free_coords[1] = 1;
    MPI_Cart_sub(cartComm, free_coords, &row_comm_);

    // column communicator
    free_coords[0] = 1;
    free_coords[1] = 0;
    MPI_Cart_sub(cartComm, free_coords, &col_comm_);

    // size of local part of H
    int len;

    len = std::min(N_, N_ / dims_[0] + 1);
    off_[0] = coord_[0] * len;

    if (coord_[0] < dims_[0] - 1) {
      m_ = len;
    } else {
      m_ = N_ - (dims_[0] - 1) * len;
    }

    len = std::min(N_, N_ / dims_[1] + 1);
    off_[1] = coord_[1] * len;

    if (coord_[1] < dims_[1] - 1) {
      n_ = len;
    } else {
      n_ = N_ - (dims_[1] - 1) * len;
    }

    mb_ = m_;
    nb_ = n_;
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

    H_.reset(new T[n_ * m_]());
    B_.reset(new T[n_ * max_block_]());
    C_.reset(new T[m_ * max_block_]());
    IMT_.reset(new T[std::max(n_, m_) * max_block_]());

    block_counts_.resize(2);
    for (std::size_t dim_idx = 0; dim_idx < 2; dim_idx++) {
        block_counts_[dim_idx].resize(dims_[dim_idx]);
	for(std::size_t i = 0; i < dims_[dim_idx]; i++){
	    block_counts_[dim_idx][i] = 1; 
        }
    }

    block_displs_.resize(2);
    block_lens_.resize(2);
    send_lens_.resize(2);
    g_offsets_.resize(2);

    for (std::size_t dim_idx = 0; dim_idx < 2; dim_idx++) {
        block_lens_[dim_idx].resize(dims_[dim_idx]);
        block_displs_[dim_idx].resize(dims_[dim_idx]);	   
        send_lens_[dim_idx].resize(dims_[dim_idx]);
	for(std::size_t i = 0; i < dims_[dim_idx]; ++i){
	    block_lens_[dim_idx][i].resize(1);
            block_displs_[dim_idx][i].resize(1);
            len = std::min(N_, N_ / dims_[dim_idx] + 1);
	    block_lens_[dim_idx][i][0] = len;
	    block_displs_[dim_idx][i][0] = i * block_lens_[dim_idx][0][0];
	    send_lens_[dim_idx][i] = len;
      g_offsets_[dim_idx].push_back(block_displs_[dim_idx][i][0]);
	}
	block_lens_[dim_idx][dims_[dim_idx] - 1].resize(1);
        block_displs_[dim_idx][dims_[dim_idx] - 1].resize(1);
	block_lens_[dim_idx][dims_[dim_idx] - 1][0] = N_ - (dims_[dim_idx] - 1) * len;
        block_displs_[dim_idx][dims_[dim_idx] - 1][0] = (dims_[dim_idx] - 1) * block_lens_[dim_idx][0][0];
        send_lens_[dim_idx][dims_[dim_idx] - 1] = N_ - (dims_[dim_idx] - 1) * len;
        g_offsets_[dim_idx].push_back(block_displs_[dim_idx][dims_[dim_idx] - 1][0]);
    }

  }

  std::size_t get_N() { return N_; };
  std::size_t get_n() { return n_; };
  std::size_t get_m() { return m_; };
  std::size_t get_max_block() { return max_block_; };
  std::size_t GetNev() { return nev_; };
  std::size_t GetNex() { return nex_; };

  std::size_t get_nb() { return nb_; };
  std::size_t get_mb() { return mb_; };

  MPI_Comm get_row_comm() { return row_comm_; }
  MPI_Comm get_col_comm() { return col_comm_; }

  // dimensions of cartesian communicator grid
  int* get_dims() { return dims_; }
  // offsets of this rank
  std::size_t* get_off() { return off_; }
  void get_off(std::size_t* xoff, std::size_t* yoff, std::size_t* xlen,
               std::size_t* ylen) {
    *xoff = off_[0];
    *yoff = off_[1];
    *xlen = m_;
    *ylen = n_;
  }


  std::size_t get_mblocks(){
      return mblocks_;
  }

  std::size_t get_nblocks(){
      return nblocks_;
  }

  int get_irsrc(){
      return irsrc_;
  }

  int get_icsrc(){
      return icsrc_;
  }

  std::size_t *get_row_offs(){ return r_offs_.get();}
  std::size_t *get_row_lens(){ return r_lens_.get();}
  std::size_t *get_row_offs_loc(){ return r_offs_l_.get();}
  std::size_t *get_col_offs(){ return c_offs_.get();}
  std::size_t *get_col_lens(){ return c_lens_.get();}
  std::size_t *get_col_offs_loc(){ return c_offs_l_.get();}

  void get_offs_lens(std::size_t* &r_offs, std::size_t* &r_lens, std::size_t* &r_offs_l, std::size_t* &c_offs, std::size_t* &c_lens, std::size_t* &c_offs_l){
      r_offs = r_offs_.get();
      r_lens = r_lens_.get();
      r_offs_l = r_offs_l_.get();      
      c_offs = c_offs_.get();
      c_lens = c_lens_.get();
      c_offs_l = c_offs_l_.get();
  }


  // coordinates in the cartesian communicator grid
  int* get_coord() { return coord_; }

  // some nice utility functions
  std::size_t get_ldb() { return n_; };
  std::size_t get_ldc() { return m_; };

  T* get_H() { return H_.get(); }
  T* get_B() { return B_.get(); }
  T* get_C() { return C_.get(); }
  T* get_IMT() { return IMT_.get(); }

  const std::vector<std::vector<int> >& get_blockcounts() { return block_counts_; }
  const std::vector<std::vector<std::vector<int>>>& get_blocklens() { return block_lens_; }
  const std::vector<std::vector<std::vector<int>>>& get_blockdispls() { return block_displs_; }
  const std::vector<std::vector<int>>& get_sendlens() { return send_lens_; }
  const std::vector<std::vector<int>>& get_g_offsets() { return g_offsets_; }

  int get_nprocs() { return nprocs_; }
  // TODO this should take a dimIdx and return the rank of the col or row
  // communicator
  int get_my_rank() { return rank_; }

  ChaseMpiMatrices<T> create_matrices(T* V1 = nullptr, Base<T>* ritzv = nullptr,
                                      T* V2 = nullptr,
                                      Base<T>* resid = nullptr) const {
    return ChaseMpiMatrices<T>(comm_, N_, max_block_, V1, ritzv, V2, resid);
  }

 private:
  std::size_t N_;
  std::size_t nev_;
  std::size_t nex_;
  std::size_t max_block_;

  std::size_t n_;
  std::size_t m_;

  //block size for block-cylic data layout
  //if nb_ = n_ and mb = m_, block-cylic data layout
  //equals to default block layout of ChASE
  std::size_t nb_;
  std::size_t mb_;
  std::size_t nblocks_;
  std::size_t mblocks_;
  int irsrc_;
  int icsrc_;
  std::unique_ptr<std::size_t[]> r_offs_;
  std::unique_ptr<std::size_t[]> r_lens_;
  std::unique_ptr<std::size_t[]> r_offs_l_;
  std::unique_ptr<std::size_t[]> c_offs_;
  std::unique_ptr<std::size_t[]> c_lens_;
  std::unique_ptr<std::size_t[]> c_offs_l_;

  // TODO this should be std::array<std::vector<>,2>
  std::vector<std::vector<int>> block_counts_;
  std::vector<std::vector<std::vector<int>>> block_lens_;
  std::vector<std::vector<std::vector<int>>> block_displs_;
  std::vector<std::vector<int>> send_lens_;
  std::vector<std::vector<int>> g_offsets_;

  MPI_Comm comm_;

  int nprocs_;
  int rank_;

  std::unique_ptr<T[]> H_;
  std::unique_ptr<T[]> B_;
  std::unique_ptr<T[]> C_;
  std::unique_ptr<T[]> IMT_;

  MPI_Comm row_comm_, col_comm_;
  int dims_[2];
  int coord_[2];
  std::size_t off_[2];
};
}  // namespace mpi
}  // namespace chase
