/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials, 
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <cstddef>
#include <memory>

#include "algorithm/types.hpp"

namespace chase {
namespace mpi {

/*
 *  Utility class for Buffers
 */
//! A class to setup the buffers of matrices and vectors which will be used by ChASE-MPI.
/*!
  This class provides two constructors:
  - Allocating the buffers for ChaseMpi without MPI support.
  - Allocating the buffers for ChaseMpi with MPI support.
  @tparam T: the scalar type used for the application. ChASE is templated
    for real and complex scalar with both Single Precision and Double Precision,
    thus `T` can be one of `float`, `double`, `std::complex<float>` and 
    `std::complex<double>`.
*/  
template <class T>
class ChaseMpiMatrices {
 public:
  // Non-MPI case: Allocate everything
  //! A constructor of ChaseMpiMatrices for **Non-MPI case** which allocates everything required.
  /*!
    The **private members** of this class are initialized by the parameters of this constructor.
    - For `H__` and `H_`, they are of size `N * N`.
    - For `V1__`, `V1_`, `V2__` and `V1_`, they are of size `N * max_block`.
    - For `ritzv__`, `ritzv_`, `resid__` and `resid_`, they are of size `max_block`.
    @param N: size of the square matrix defining the eigenproblem.
    @param max_block: Maximum column number of matrix `V1_` and `V2_`. It equals to `nev_ + nex_`.
    @param V1: a pointer to the buffer `V1_`.
    @param ritz: a pointer to the buffer `ritz_`.
    @param H: a pointer to the buffer `H_`.
    @param V2: a pointer to the buffer `V2_`.
    @param resid: a pointer to the buffer `resid_`. 
  */
  ChaseMpiMatrices(std::size_t N, std::size_t max_block, T* V1 = nullptr,
                   Base<T>* ritzv = nullptr, T* H = nullptr, T* V2 = nullptr,
                   Base<T>* resid = nullptr)
      // if value is null then allocate otherwise don't
      : H__(H == nullptr ? new T[N * N] : nullptr),
        V1__(V1 == nullptr ? new T[N * max_block] : nullptr),
        V2__(V2 == nullptr ? new T[N * max_block] : nullptr),
        ritzv__(ritzv == nullptr ? new Base<T>[max_block] : nullptr),
        resid__(resid == nullptr ? new Base<T>[max_block] : nullptr),
        ldh_(N),
        // if value is null we take allocated
        H_(H == nullptr ? H__.get() : H),
        V1_(V1 == nullptr ? V1__.get() : V1),
        V2_(V2 == nullptr ? V2__.get() : V2),
        ritzv_(ritzv == nullptr ? ritzv__.get() : ritzv),
        resid_(resid == nullptr ? resid__.get() : resid) {}

  // MPI case: we don't allocate H here
  //! A constructor of ChaseMpiMatrices for **MPI case** which allocates everything required except `H_`.
  /*!
    The **private members** of this class are initialized by the parameters of this constructor.
    - For `V1__`, `V1_`, `V2__` and `V1_`, they are of size `N * max_block`.
    - For `ritzv__`, `ritzv_`, `resid__` and `resid_`, they are of size `N * max_block`.
    - `H_` is not allocated here, but in ChaseMpiProperties class which takes the MPI environment and data distribution scheme into account.
    @param comm: the working MPI communicator of ChASE.
    @param N: size of the square matrix defining the eigenproblem.
    @param max_block: Maximum column number of matrix `V1_` and `V2_`. It equals to `nev_ + nex_`.
    @param V1: a pointer to the buffer `V1_`.
    @param ritz: a pointer to the buffer `ritz_`.
    @param V2: a pointer to the buffer `V2_`.
    @param resid: a pointer to the buffer `resid_`. 
  */        
  ChaseMpiMatrices(MPI_Comm comm, std::size_t N, std::size_t m, std::size_t n, std::size_t max_block,
                   T* V1 = nullptr, std::size_t ldv1 = 0, Base<T>* ritzv = nullptr, T* V2 = nullptr,
                   Base<T>* resid = nullptr)
      // if value is null then allocate otherwise don't
      : H__(nullptr),
        ldv1_(ldv1 == 0 ? m  : ldv1),
        V1__(V1 == nullptr ? new T[std::max(ldv1, m) * max_block] : nullptr),
#if !defined(HAS_SCALAPACK)        
        V2__(V2 == nullptr ? new T[N * max_block] : nullptr),
#endif
        ritzv__(ritzv == nullptr ? new Base<T>[max_block] : nullptr),
        resid__(resid == nullptr ? new Base<T>[max_block] : nullptr),
        //for this case, ldh_ should define in chasempiproperties
        ldh_(0),
        // if value is null we take allocated
        H_(nullptr),
        V1_(V1 == nullptr ? V1__.get() : V1),
        V2_(V2 == nullptr ? V2__.get() : V2),
        ritzv_(ritzv == nullptr ? ritzv__.get() : ritzv),
        resid_(resid == nullptr ? resid__.get() : resid) {}

  ChaseMpiMatrices(MPI_Comm comm, std::size_t N, std::size_t m, std::size_t n, std::size_t max_block, 
                   T* V1 = nullptr,  std::size_t ldv1 = 0, Base<T>* ritzv = nullptr, T* H = nullptr, 
                   std::size_t ldh = 0, T* V2 = nullptr, Base<T>* resid = nullptr)
      // if value is null then allocate otherwise don't
      : H__(H == nullptr ? new T[m * n] : nullptr),
        ldv1_(ldv1 == 0 ? m  : ldv1),      
        V1__(V1 == nullptr ? new T[std::max(ldv1, m) * max_block] : nullptr),
#if !defined(HAS_SCALAPACK)                
        V2__(V2 == nullptr ? new T[N * max_block] : nullptr),
#endif        
        ritzv__(ritzv == nullptr ? new Base<T>[max_block] : nullptr),
        resid__(resid == nullptr ? new Base<T>[max_block] : nullptr),
        ldh_(ldh == 0 ? m : ldh),
        // if value is null we take allocated
        H_(H == nullptr ? H__.get() : H),
        V1_(V1 == nullptr ? V1__.get() : V1),
        V2_(V2 == nullptr ? V2__.get() : V2),
        ritzv_(ritzv == nullptr ? ritzv__.get() : ritzv),
        resid_(resid == nullptr ? resid__.get() : resid) {
        }

  //! Return buffer stores the (local part if applicable) matrix A.
  /*! \return `H_`, a private member of this class.
  */
  T* get_H() { return H_; }
  //! Return the buffer `V1_`  which will be right-multiplied to `A` during the process of ChASE.  
  /*! \return `V1_`, a private member of this class.  
  */
  T* get_V1() { return V1_; }
  //! Return the buffer `V2_` whose conjugate transpose will be left-multiplied to `A` during the process of ChASE.
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

  std::size_t get_ldh() { return ldh_; }

 private:
  //! A smart pointer which manages the buffer of `H_` which stores (local part if applicable) matrix `A`. 
  std::unique_ptr<T[]> H__;
  //! A smart pointer which manages the buffer of `V1_` which will be right-multiplied to `A` during the process of ChASE. The eigenvectors obtained will also stored in `V1_`.  
  std::unique_ptr<T[]> V1__;
  //! A smart pointer which manages the buffer of `V2_` whose conjugate transpose will be left-multiplied to `A` during the process of ChASE.
  std::unique_ptr<T[]> V2__;
  //! A smart pointer which manages the buffer which stores the computed Ritz values.
  std::unique_ptr<Base<T>[]> ritzv__;
  //! A smart pointer which manages the buffer which stores the residual of computed Ritz pairs.  
  std::unique_ptr<Base<T>[]> resid__;

  //! The buffer stores (local part if applicable) matrix `A`.
  T* H_;
  //! The buffer of `V1_`  which will be right-multiplied to `A` during the process of ChASE. The eigenvectors obtained will also be stored in `V_`.
  T* V1_;
  //! The buffer of `V2_` whose conjugate transpose will be left-multiplied to `A` during the process of ChASE.
  T* V2_;
  //! The buffer which stores the computed Ritz values.
  Base<T>* ritzv_;
  //! The buffer which stores the residual of computed Ritz pairs.  
  Base<T>* resid_;

  std::size_t ldh_;
  std::size_t ldv1_;
};
}  // namespace mpi
}  // namespace chase
