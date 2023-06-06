/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2023, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <cstddef>
#include <memory>
#if defined(HAS_CUDA)
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif
#include "algorithm/types.hpp"

namespace chase
{
namespace mpi
{

template<class T>
class CpuMem
{
    public:

      CpuMem():size_(0), ptr_(nullptr), allocated_(false){}	      
      CpuMem(std::size_t size)
	: size_(size), allocated_(true), type_("CPU") {
#if defined(HAS_CUDA)
	  cudaMallocHost(&ptr_, size_ * sizeof(T));
#else		
          ptr_=std::allocator<T>().allocate(size_);
#endif
      }

      CpuMem(T* ptr, std::size_t size) : size_(size), ptr_(ptr), allocated_(false), type_("CPU") {}

      ~CpuMem() {
	if (allocated_) {
#if defined(HAS_CUDA)
	    cudaFreeHost(ptr_);
#else		
	    std::allocator<T>().deallocate(ptr_,size_);
#endif
	}
      }
      
      T *ptr () 
      {
          return ptr_;
      }

      bool isAlloc () 
      {
          return allocated_;
      }

      std::string type() 
      {
          return type_;
      }
    private:
      std::size_t size_;
      T* ptr_;
      bool allocated_;
      std::string type_;	    
};

#if defined(HAS_CUDA)
template<class T>
class GpuMem
{
    public:
      GpuMem():size_(0), ptr_(nullptr), allocated_(false){}

      GpuMem(std::size_t size)
        : size_(size), allocated_(true), type_("GPU") {
        cudaMalloc(&ptr_, size_ * sizeof(T));
      }

      GpuMem(T* ptr, std::size_t size) : size_(size), ptr_(ptr), allocated_(false), type_("GPU") {}

      ~GpuMem() {
        if (allocated_) {
          cudaFree(ptr_);
        }
      }

      T *ptr () 
      {
          return ptr_;
      }

      bool isAlloc ()
      {
          return allocated_;
      }

      std::string type()
      {
          return type_;
      }

    private:
      std::size_t size_;
      T* ptr_;
      bool allocated_;
      std::string type_;

};
#endif

template<class T>
class Matrix
{
    public:
      using ElementType = T;

      Matrix():m_(0), n_(0), ld_(0){}
      //mode: 0: CPU, 1: traditional GPU, 2: CUDA-Aware
      Matrix(int mode, std::size_t m, std::size_t n)
      :m_(m), n_(n), ld_(m), mode_(mode)
      {
	switch(mode)
	{
	    case 0:
		Host_ = std::make_shared<CpuMem<T>>(m * n);
		isHostAlloc_ = true;
		isDeviceAlloc_ = false;
		break;
#if defined(HAS_CUDA)		
	    case 1:
                Host_ = std::make_shared<CpuMem<T>>(m * n);
		Device_ = std::make_shared<GpuMem<T>>(m * n);
                isHostAlloc_ = true;
                isDeviceAlloc_ = true;		
		break;
	    case 2:
    		Device_ = std::make_shared<GpuMem<T>>(m * n);		
                isHostAlloc_ = false;
                isDeviceAlloc_ = true;		
		break;
#endif	
	}
      }     
  
      Matrix(int mode, std::size_t m, std::size_t n, T *ptr, std::size_t ld)
      :m_(m), n_(n), ld_(ld), mode_(mode)
      {
        switch(mode)
        {
            case 0:
                Host_ = std::make_shared<CpuMem<T>>(ptr, ld * n);
                isHostAlloc_ = false;
                isDeviceAlloc_ = false;
                break;
#if defined(HAS_CUDA)
            case 1:
                Host_ = std::make_shared<CpuMem<T>>(ptr, ld * n);
                Device_ = std::make_shared<GpuMem<T>>(m * n);
                isHostAlloc_ = false;
                isDeviceAlloc_ = true;
                break;
            case 2:
                Device_ = std::make_shared<GpuMem<T>>(m * n);
                Host_ = std::make_shared<CpuMem<T>>(ptr, ld * n);		
                isHostAlloc_ = false;
                isDeviceAlloc_ = true;
                break;
#endif
		
        }
      }

     T *host(){
         return Host_.get()->ptr();
     }  

     T *ptr(){
         return Host_.get()->ptr();
     }
     
#if defined(HAS_CUDA)
     T *device(){
         return Device_.get()->ptr();
     }
#endif
     std::size_t ld(){
         return ld_;
     }

     std::size_t h_ld(){
         return ld_;
     }
#if defined(HAS_CUDA)     
     std::size_t d_ld(){
         return m_;
     }
#endif    

#if defined(HAS_CUDA)
    void H2D()
    {
       cublasSetMatrix(m_, n_, sizeof(T), this->host(), this->h_ld(), this->device(), this->d_ld());
    }	    

    void H2D(std::size_t nrows, std::size_t ncols, std::size_t offset = 0)
    {
       cublasSetMatrix(nrows, ncols, sizeof(T), this->host() + offset * this->h_ld(),
		       this->h_ld(), this->device() + offset * this->d_ld(), this->d_ld());
    }
#endif     
    
    private:
      std::size_t m_;
      std::size_t n_;
      std::size_t ld_;
      std::shared_ptr<CpuMem<T>> Host_;
#if defined(HAS_CUDA)
      std::shared_ptr<GpuMem<T>> Device_;
#endif      
      bool isHostAlloc_;
      bool isDeviceAlloc_;
      bool mode_;
};
/*
 *  Utility class for Buffers
 */
//! @brief A class to setup the buffers of matrices and vectors which will be
//! used by ChaseMpi.
/*!
  This class provides three constructors:
  - Allocating the buffers for ChaseMpi without MPI support.
  - Allocating the buffers for ChaseMpi with MPI support.
  -  Allocating the buffers for ChaseMpi with MPI support in which the buffer
  to store the matrix to be diagonalised is externally allocated and provided by
  users.
  @tparam T: the scalar type used for the application. ChASE is templated
    for real and complex scalar with both Single Precision and Double Precision,
    thus `T` can be one of `float`, `double`, `std::complex<float>` and
    `std::complex<double>`.
*/
template <class T>
class ChaseMpiMatrices
{
public:
    //! A constructor of ChaseMpiMatrices for **Non-MPI case** which allocates
    //! everything required.
    /*!
      The **private members** of this class are initialized by the parameters of
      this constructor.
      - For `H__` and `H_`, they are of size `ldh * N`.
      - For `V1__`, `V1_`, `V2__` and `V1_`, they are of size `N * max_block`.
      - For `ritzv__`, `ritzv_`, `resid__` and `resid_`, they are of size
      `max_block`.
      @param N: size of the square matrix defining the eigenproblem.
      @param max_block: Maximum column number of matrix `V1_` and `V2_`. It
      equals to `nev_ + nex_`.
      @param H: a pointer to the buffer `H_`.
      @param ldh: the leading dimension of `H_`.
      @param V1: a pointer to the buffer `V1_`.
      @param ritz: a pointer to the buffer `ritz_`.
      @param V2: a pointer to the buffer `V2_`.
      @param resid: a pointer to the buffer `resid_`.
    */
    ChaseMpiMatrices(std::size_t N, std::size_t max_block, T *H, std::size_t ldh, 
                     T* V1, Base<T>* ritzv, T* V2 = nullptr,
                     Base<T>* resid = nullptr)
        // if value is null then allocate otherwise don't
        : H__(nullptr),
          V1__(V1 == nullptr ? new T[N * max_block] : nullptr),
          V2__(V2 == nullptr ? new T[N * max_block] : nullptr),
          ritzv__(ritzv == nullptr ? new Base<T>[max_block] : nullptr),
          resid__(resid == nullptr ? new Base<T>[max_block] : nullptr), 
          ldh_(ldh),
          // if value is null we take allocated
          H_(H),
          V1_(V1 == nullptr ? V1__.get() : V1),
          V2_(V2 == nullptr ? V2__.get() : V2),
          ritzv_(ritzv == nullptr ? ritzv__.get() : ritzv),
          resid_(resid == nullptr ? resid__.get() : resid)
    {
    }    


    //! A constructor of ChaseMpiMatrices for **MPI case** which allocates
    //! everything necessary except `H_`.
    /*!
      The **private members** of this class are initialized by the parameters of
      this constructor.
      - For `V1__` and `V1_`, they are of size `m_ * max_block`.
      - For `V2__` and `V2_`, they are of size `n_ * max_block`.
      - For `ritzv__`, `ritzv_`, `resid__` and `resid_`, they are of size
      `max_block`.
      - `H_` is allocated externally based the users, it is of size `ldh_ * n_`
      with `ldh_>=m_`.
      - `m` and `n` can be obtained through ChaseMpiProperties::get_m() and
      ChaseMpiProperties::get_n(), respecitvely.
      @param comm: the working MPI communicator of ChASE.
      @param N: size of the square matrix defining the eigenproblem.
      @param m: row number of `H_`.`m` can be obtained through
       ChaseMpiProperties::get_m()
      @param n: column number of `H_`.`n` can be obtained through
       ChaseMpiProperties::get_n()
      @param max_block: Maximum column number of matrix `V1_` and `V2_`. It
      equals to `nev_ + nex_`.
      @param H: the pointer to the user-provided buffer of matrix to be
      diagonalised.
      @param ldh: The leading dimension of local part of Symmetric/Hermtian
      matrix on each MPI proc.
      @param V1: a pointer to the buffer `V1_`.
      @param ritz: a pointer to the buffer `ritz_`.
      @param V2: a pointer to the buffer `V2_`.
      @param resid: a pointer to the buffer `resid_`.
    */
    ChaseMpiMatrices(int mode, MPI_Comm comm, std::size_t N, std::size_t m, std::size_t n,
                     std::size_t max_block, T* H, std::size_t ldh,
                     T* V1,  Base<T>* ritzv)
        // if value is null then allocate otherwise don't
        : ritzv__(ritzv == nullptr ? new Base<T>[max_block] : nullptr),
          resid__(new Base<T>[max_block] ),
          ldh_(ldh),
          // if value is null we take allocated
          H_(H), V1_(V1 == nullptr ? V1__.get() : V1),
          V2_(V2__.get()),
          ritzv_(ritzv == nullptr ? ritzv__.get() : ritzv),
          resid_(resid__.get())
    {    
	int isGPU;
    	int isCUDA_Aware;
	if(mode == 0){
	    isGPU = 0;
	    isCUDA_Aware = 0;
	}else if(mode == 1){
	    isGPU = 1;
	    isCUDA_Aware = 1;
	}else{
            isGPU = 1;
            isCUDA_Aware = 2;	
	}	
	H___ = std::make_unique<Matrix<T>>(isGPU, m, n, H, ldh);
	C___ = std::make_unique<Matrix<T>>(isCUDA_Aware, m, max_block, V1, m);
        C2___ = std::make_unique<Matrix<T>>(isCUDA_Aware, m, max_block);	
    }

    //! Return buffer stores the (local part if applicable) matrix A.
    /*! \return `H_`, a private member of this class.
     */
    T* get_H() { return H_; }
    //! Return the buffer `V1_`  which will be right-multiplied to `A` during
    //! the process of ChASE.
    /*! \return `V1_`, a private member of this class.
     */
    T* get_V1() { return V1_; }
    //! Return the buffer `V2_` whose conjugate transpose will be
    //! left-multiplied to `A` during the process of ChASE.
    /*! \return `V2_`, a private member of this class.
     */
    T* get_V2() { return V2_; }
    //! Return the buffer which stores the computed Ritz values.
    /*! \return `ritzv_`, a private member of this class.
     */
    Base<T>* get_Ritzv() { return ritzv_; }
    //! Return the buffer which stores the residual of computed Ritz pairs.
    /*! \return `resid_`, a private member of this class.
     */
    Base<T>* get_Resid() { return resid_; }
    //! Return leading dimension of local part of Symmetric/Hermtian matrix on
    //! each MPI proc.
    /*! \return `ldh_`, a private member of this class.
     */
    std::size_t get_ldh() { return ldh_; }

    Matrix<T> H() {return *H___.get();}
    Matrix<T> C() {return *C___.get();}
    Matrix<T> C2() {return *C2___.get();}

private:
    //! A smart pointer which manages the buffer of `H_` which stores (local
    //! part if applicable) matrix `A`.
    std::unique_ptr<T[]> H__;
    //! A smart pointer which manages the buffer of `V1_` which will be
    //! right-multiplied to `A` during the process of ChASE. The eigenvectors
    //! obtained will also stored in `V1_`.
    std::unique_ptr<T[]> V1__;
    //! A smart pointer which manages the buffer of `V2_` whose conjugate
    //! transpose will be left-multiplied to `A` during the process of ChASE.
    std::unique_ptr<T[]> V2__;
    //! A smart pointer which manages the buffer which stores the computed Ritz
    //! values.
    std::unique_ptr<Base<T>[]> ritzv__;
    //! A smart pointer which manages the buffer which stores the residual of
    //! computed Ritz pairs.
    std::unique_ptr<Base<T>[]> resid__;

    //! The buffer stores (local part if applicable) matrix `A`.
    T* H_;
    //! The buffer of `V1_`  which will be right-multiplied to `A` during the
    //! process of ChASE. The eigenvectors obtained will also be stored in `V_`.
    T* V1_;
    //! The buffer of `V2_` whose conjugate transpose will be left-multiplied to
    //! `A` during the process of ChASE.
    T* V2_;
    //! The buffer which stores the computed Ritz values.
    Base<T>* ritzv_;
    //! The buffer which stores the residual of computed Ritz pairs.
    Base<T>* resid_;
    //! The leading dimension of local part of Symmetric/Hermtian matrix on each
    //! MPI proc.
    std::size_t ldh_;

    std::unique_ptr<Matrix<T>> H___;
    std::unique_ptr<Matrix<T>> C___;
    std::unique_ptr<Matrix<T>> C2___;
};
} // namespace mpi
} // namespace chase

