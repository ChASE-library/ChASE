/* -*- Mode: C++; -*- */
#ifndef _CHASE_GENERA_BLAS_CHASE_BLAS_MATRICES_
#define _CHASE_GENERA_BLAS_CHASE_BLAS_MATRICES_

#include <cstddef>
#include <memory>

#include "algorithm/types.h"

namespace chase {
namespace matrixfree {

/*
 *  Utility class for Buffers
 */

template <class T>
class ChASE_Blas_Matrices {
 public:
  // Non-MPI case: Allocate everything
  ChASE_Blas_Matrices(std::size_t N, std::size_t max_block,
                      T* V1 = nullptr, Base<T>* ritzv = nullptr,
                      T* H = nullptr, T* V2 = nullptr,
                      Base<T>* resid = nullptr)
      // if value is null then allocate otherwise don't
      : H__(H == nullptr ? new T[N * N] : nullptr),
        V1__(V1 == nullptr ? new T[N * max_block] : nullptr),
        V2__(V2 == nullptr ? new T[N * max_block] : nullptr),
        ritzv__(ritzv == nullptr ? new Base<T>[max_block] : nullptr),
        resid__(resid == nullptr ? new Base<T>[max_block] : nullptr),
        // if value is null we take allocated
        H_(H == nullptr ? H__.get() : H),
        V1_(V1 == nullptr ? V1__.get() : V1),
        V2_(V2 == nullptr ? V2__.get() : V2),
        ritzv_(ritzv == nullptr ? ritzv__.get() : ritzv),
        resid_(resid == nullptr ? resid__.get() : resid) {}

  // MPI case: we don't allocate H here
  ChASE_Blas_Matrices(MPI_Comm comm, std::size_t N, std::size_t max_block,
                      T* V1 = nullptr, Base<T>* ritzv = nullptr,
                      T* V2 = nullptr, Base<T>* resid = nullptr)
      // if value is null then allocate otherwise don't
      : H__(nullptr),
        V1__(V1 == nullptr ? new T[N * max_block] : nullptr),
        V2__(V2 == nullptr ? new T[N * max_block] : nullptr),
        ritzv__(ritzv == nullptr ? new Base<T>[max_block] : nullptr),
        resid__(resid == nullptr ? new Base<T>[max_block] : nullptr),
        // if value is null we take allocated
        H_(nullptr),
        V1_(V1 == nullptr ? V1__.get() : V1),
        V2_(V2 == nullptr ? V2__.get() : V2),
        ritzv_(ritzv == nullptr ? ritzv__.get() : ritzv),
        resid_(resid == nullptr ? resid__.get() : resid) {}

  /*
  // we don't alloc H, since skewed classes use the buffers in skewedproperties
  ChASE_Blas_Matrices(std::size_t N, std::size_t max_block, T* V1,
                      Base<T>* ritzv,
                      std::shared_ptr<SkewedMatrixProperties<T>>& properties)
      : H__(nullptr),
        V1__(nullptr),
        V2__(new T[N * max_block]),
        ritzv__(nullptr),
        resid__(new Base<T>[max_block]),
        H_(nullptr),
        V1_(V1),
        V2_(V2__.get()),
        ritzv_(ritzv),
        resid_(resid__.get()) {}
  */
  T* get_H() { return H_; }
  T* get_V1() { return V1_; }
  T* get_V2() { return V2_; }
  Base<T>* get_Ritzv() { return ritzv_; }
  Base<T>* get_Resid() { return resid_; }

 private:
  std::unique_ptr<T[]> H__;
  std::unique_ptr<T[]> V1__;
  std::unique_ptr<T[]> V2__;
  std::unique_ptr<Base<T>[]> ritzv__;
  std::unique_ptr<Base<T>[]> resid__;

  T* H_;
  T* V1_;
  T* V2_;
  Base<T>* ritzv_;
  Base<T>* resid_;
};
}  // namespace matrixfree
}  // namespace chase
#endif  //  _CHASE_GENERA_BLAS_CHASE_BLAS_MATRICES_
