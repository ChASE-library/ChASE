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
#include <iterator>

#if USE_TIMER
#include <chrono>
#endif

#include "ChASE-MPI/chase_mpi_properties.hpp"
#include "ChASE-MPI/chase_mpihemm_interface.hpp"

#if USE_TIMER
using namespace std::chrono;
#endif

namespace chase {
namespace mpi {

template <class T>
class ChaseMpiHemm : public ChaseMpiHemmInterface<T> {
 public:
  ChaseMpiHemm(ChaseMpiProperties<T>* matrix_properties,
               ChaseMpiHemmInterface<T>* gemm)
      : gemm_(gemm) {
    ldc_ = matrix_properties->get_ldc();
    ldb_ = matrix_properties->get_ldb();

    N_ = matrix_properties->get_N();
    n_ = matrix_properties->get_n();
    m_ = matrix_properties->get_m();

    H_ = matrix_properties->get_H();
    B_ = matrix_properties->get_B();
    C_ = matrix_properties->get_C();
    IMT_ = matrix_properties->get_IMT();

    matrix_properties_ = matrix_properties;

    row_comm_ = matrix_properties->get_row_comm();
    col_comm_ = matrix_properties->get_col_comm();

    dims_ = matrix_properties->get_dims();
    coord_ = matrix_properties->get_coord();
    off_ = matrix_properties->get_off();

    matrix_properties->get_offs_lens(r_offs_, r_lens_, r_offs_l_, c_offs_, c_lens_, c_offs_l_);
    mb_ = matrix_properties->get_mb();
    nb_ = matrix_properties->get_nb();    
    mblocks_ = matrix_properties->get_mblocks();
    nblocks_ = matrix_properties->get_nblocks();

#if USE_TIMER
	time_preApp = std::chrono::milliseconds::zero(); 
	time_apply = std::chrono::milliseconds::zero();
	time_allreduce = std::chrono::milliseconds::zero();
	time_scal = std::chrono::milliseconds::zero();
	time_axpy = std::chrono::milliseconds::zero();
	time_postApp = std::chrono::milliseconds::zero();
	time_shift = std::chrono::milliseconds::zero();
#endif
  }

  ~ChaseMpiHemm() {

#if USE_TIMER
	std::cout << "CHASE_MPIHEMM timings: " << std::endl;
	std::cout << "preApplication  = " << time_preApp.count()/1000 << " sec" << std::endl;
	std::cout << "apply           = " << time_apply.count()/1000 << " sec" << std::endl;
	std::cout << "allReduce       = " << time_allreduce.count()/1000 << " sec" << std::endl;
	std::cout << "scale	 		  = " << time_scal.count()/1000 << " sec"   << std::endl;
	std::cout << "axpy     		  = " << time_axpy.count()/1000 << " sec"  << std::endl;
	std::cout << "postApplication = " << time_postApp.count()/1000 << " sec"  << std::endl;
	std::cout << "shift           = " << time_shift.count()/1000 << " sec"  << std::endl;
	std::cout << std::endl;
#endif
  }

  void preApplication(T* V, std::size_t locked, std::size_t block) {
    next_ = NextOp::bAc;
    locked_ = locked;

#if USE_TIMER
	auto start = high_resolution_clock::now();
#endif

    for (auto j = 0; j < block; j++){
   	for(auto i = 0; i < mblocks_; i++){
	    std::memcpy(C_ + j * m_ + r_offs_l_[i], V + j * N_ + locked * N_ + r_offs_[i], r_lens_[i] * sizeof(T));
	} 
    }

    gemm_->preApplication(V, locked, block);
#if USE_TIMER
	auto stop = high_resolution_clock::now();
	time_preApp += stop - start;
#endif
  }

  void preApplication(T* V1, T* V2, std::size_t locked, std::size_t block) {

#if USE_TIMER
	auto start = high_resolution_clock::now();
#endif

    for (auto j = 0; j < block; j++) {
	for(auto i = 0; i < nblocks_; i++){
            std::memcpy(B_ + j * n_ + c_offs_l_[i], V2 + j * N_ + locked * N_ + c_offs_[i], c_lens_[i] * sizeof(T));	    
	}	
    }

    gemm_->preApplication(V1, V2, locked, block);

#if USE_TIMER
	auto stop = high_resolution_clock::now();
	time_preApp += stop - start;
#endif

    this->preApplication(V1, locked, block);
  }

  void apply(T alpha, T beta, std::size_t offset, std::size_t block) {
    T One = T(1.0);
    T Zero = T(0.0);

    std::size_t dim;
    if (next_ == NextOp::bAc) {
      dim = n_ * block;

#if USE_TIMER
 	  auto start = high_resolution_clock::now();
#endif

      gemm_->apply(One, Zero, offset, block);

#if USE_TIMER
	  auto stop = high_resolution_clock::now();
	  time_apply += stop - start;

	  start = high_resolution_clock::now();	  
#endif

      MPI_Allreduce(MPI_IN_PLACE, IMT_ + offset * n_, dim, getMPI_Type<T>(),
                    MPI_SUM, col_comm_);
	

#if USE_TIMER
	  stop = high_resolution_clock::now();
	  time_allreduce += stop -start;

	  start = high_resolution_clock::now();	  
#endif

      t_scal(dim, &beta, B_ + offset * n_, 1);

#if USE_TIMER
	  stop = high_resolution_clock::now();
	  time_scal += stop -start;
	  start = high_resolution_clock::now();	  
#endif

      t_axpy(dim, &alpha, IMT_ + offset * n_, 1, B_ + offset * n_, 1);

#if USE_TIMER
	  stop = high_resolution_clock::now();
	  time_axpy += stop -start;
#endif

      next_ = NextOp::cAb;
    } else {  // cAb

      dim = m_ * block;

#if USE_TIMER
 	  auto start = high_resolution_clock::now();
#endif

      gemm_->apply(One, Zero, offset, block);

#if USE_TIMER
	  auto stop = high_resolution_clock::now();
	  time_apply += stop - start;

	  start = high_resolution_clock::now();	  
#endif

      MPI_Allreduce(MPI_IN_PLACE, IMT_ + offset * m_, dim, getMPI_Type<T>(),
                    MPI_SUM, row_comm_);
#if USE_TIMER
	  stop = high_resolution_clock::now();
	  time_allreduce += stop -start;

	  start = high_resolution_clock::now();	  
#endif

      t_scal(dim, &beta, C_ + offset * m_, 1);

#if USE_TIMER
	  stop = high_resolution_clock::now();
	  time_scal += stop -start;
	  start = high_resolution_clock::now();	  
#endif

      t_axpy(dim, &alpha, IMT_ + offset * m_, 1, C_ + offset * m_, 1);

#if USE_TIMER
	  stop = high_resolution_clock::now();
	  time_axpy += stop -start;
#endif

      next_ = NextOp::bAc;
    }

  }
  bool postApplication(T* V, std::size_t block) {
    gemm_->postApplication(V, block);

    std::size_t N = N_;
    std::size_t dimsIdx;
    std::size_t subsize;
    std::size_t blocksize;
    std::size_t nbblocks;

    T* buff;
    MPI_Comm comm;
    std::size_t *offs, *lens, *offs_l;  

    T* targetBuf = V + locked_ * N;

    if (next_ == NextOp::bAc) {
      subsize = m_;
      blocksize = mb_;
      buff = C_;
      comm = col_comm_;
      dimsIdx = 0;
      offs = r_offs_;
      lens = r_lens_;
      offs_l = r_offs_l_;
      nbblocks = mblocks_;
    } else {
      subsize = n_;
      blocksize = nb_;
      buff = B_;
      comm = row_comm_;
      dimsIdx = 1;
      offs = c_offs_;
      lens = c_lens_;
      offs_l = c_offs_l_; 
      nbblocks = nblocks_;      
    }

    int gsize, rank;
    MPI_Comm_size(comm, &gsize);
    MPI_Comm_rank(comm, &rank);

    auto& blockcounts = matrix_properties_->get_blockcounts()[dimsIdx];
    auto& blocklens = matrix_properties_->get_blocklens()[dimsIdx];
    auto& blockdispls = matrix_properties_->get_blockdispls()[dimsIdx];
    auto& sendlens = matrix_properties_->get_sendlens()[dimsIdx];

    std::vector<MPI_Request> reqs(gsize);
    std::vector<MPI_Datatype> newType(gsize);


    for(auto j = 0; j < gsize; j++){
        MPI_Type_indexed(blockcounts[j], blocklens[j].data(), blockdispls[j].data(), getMPI_Type<T>(), &(newType[j]));
        MPI_Type_commit(&(newType[j]));
    }

#if USE_TIMER
    auto start = high_resolution_clock::now();
#endif
    for(auto j = 0; j < block; j++){
        for(auto i = 0; i < nbblocks; i++){
	    std::memcpy(targetBuf + j*N + offs[i], buff + j*subsize + offs_l[i], lens[i] * sizeof(T));
	}    
    }	

    for(auto i = 0; i < block; i++){
        for (auto j = 0; j < gsize; j++) {
            MPI_Ibcast(targetBuf + i * N , 1, newType[j], j, comm, &reqs[j]);
	}
        MPI_Waitall(gsize, reqs.data(), MPI_STATUSES_IGNORE);
    }

    for (auto j = 0; j < gsize; j++) {
      MPI_Type_free(&newType[j]);
    }

#if USE_TIMER
	auto stop = high_resolution_clock::now();
	time_postApp += stop - start;
#endif

  }

  void shiftMatrix(T c, bool isunshift = false) {

#if USE_TIMER
 	auto start = high_resolution_clock::now();
#endif
	
    for(std::size_t j = 0; j < nblocks_; j++){
        for(std::size_t i = 0; i < mblocks_; i++){
            for(std::size_t q = 0; q < c_lens_[j]; q++){
                for(std::size_t p = 0; p < r_lens_[i]; p++){
		    if(q + c_offs_[j] == p + r_offs_[i]){
		        H_[(q + c_offs_l_[j]) * m_ + p + r_offs_l_[i]] += c;
		    }
		}
            }
        }
    }

    gemm_->shiftMatrix(c);

#if USE_TIMER
 	auto stop = high_resolution_clock::now();
	time_shift += stop - start;
#endif
  }

  void applyVec(T* B, T* C) {
    // TODO

    T One = T(1.0);
    T Zero = T(0.0);

    this->preApplication(B, 0, 1);
    this->apply(One, Zero, 0, 1);
    this->postApplication(C, 1);

    // gemm_->applyVec(B, C);
  }

  void get_off(std::size_t* xoff, std::size_t* yoff, std::size_t* xlen,
               std::size_t* ylen) const override {
    *xoff = off_[0];
    *yoff = off_[1];
    *xlen = m_;
    *ylen = n_;
  }

  T* get_H() const override { return matrix_properties_->get_H(); }
  std::size_t get_mblocks() const override {return mblocks_;}
  std::size_t get_nblocks() const override {return nblocks_;}
  std::size_t get_n() const override {return n_;}
  std::size_t get_m() const override {return m_;}  
  int *get_coord() const override {return matrix_properties_->get_coord();}
  void get_offs_lens(std::size_t* &r_offs, std::size_t* &r_lens, std::size_t* &r_offs_l,
                  std::size_t* &c_offs, std::size_t* &c_lens, std::size_t* &c_offs_l) const override{
     matrix_properties_->get_offs_lens(r_offs, r_lens, r_offs_l, c_offs, c_lens, c_offs_l); 
  }

  void Start() override { gemm_->Start(); }

 private:
  enum NextOp { cAb, bAc };

  std::size_t locked_;
  std::size_t ldc_;
  std::size_t ldb_;

  std::size_t n_;
  std::size_t m_;
  std::size_t N_;

  T* H_;
  T* B_;
  T* C_;
  T* IMT_;

  NextOp next_;
  MPI_Comm row_comm_, col_comm_;
  int* dims_;
  int* coord_;
  std::size_t* off_;

  std::size_t *r_offs_;
  std::size_t *r_lens_;
  std::size_t *r_offs_l_;
  std::size_t *c_offs_;
  std::size_t *c_lens_;
  std::size_t *c_offs_l_;
  std::size_t nb_;
  std::size_t mb_;
  std::size_t nblocks_;
  std::size_t mblocks_;

  std::unique_ptr<ChaseMpiHemmInterface<T>> gemm_;
  ChaseMpiProperties<T>* matrix_properties_;

#if USE_TIMER
  /// Timing variables
  std::chrono::duration<double, std::milli> time_preApp;
  std::chrono::duration<double, std::milli> time_apply;
  std::chrono::duration<double, std::milli> time_allreduce;
  std::chrono::duration<double, std::milli> time_scal;
  std::chrono::duration<double, std::milli> time_axpy;
  std::chrono::duration<double, std::milli> time_postApp;
  std::chrono::duration<double, std::milli> time_shift;
#endif
};
}  // namespace mpi
}  // namespace chase
