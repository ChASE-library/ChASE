/* -*- Mode: C++; -*- */
#ifndef _CHASE_GENERA_BLAS_CHASE_BLAS_MATRICES_
#define _CHASE_GENERA_BLAS_CHASE_BLAS_MATRICES_

#include <memory>

#include "algorithm/types.h"
#include "matrixfree_properties.h"

namespace chase {

template <class T>
class ChASE_Blas_Matrices {
 public:

  ChASE_Blas_Matrices(std::size_t N, std::size_t max_block, T* V1,
                      Base<T>* ritzv)
      : H__(new T[N * N]),
        V1__(nullptr),
        V2__(new T[N * max_block]),
        ritzv__(nullptr),
        resid__(new Base<T>[max_block]),
        H_(H__.get()),
        V1_(V1),
        V2_(V2__.get()),
        ritzv_(ritzv),
        resid_(resid__.get()) {}

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

  T* get_H() { return H_; }
  T* get_V1() { return V1_; }
  T* get_V2() { return V2_; }
  Base<T>* get_Ritzv() { return ritzv_; }
  Base<T>* get_Resid() { return resid_; }
  /*
  ChASE_Blas_Matrices(ChASE_Blas_Matrices&& othr) {
    std::swap(H__, othr.H__);
    std::swap(V1__, othr.V1__);
    std::swap(V2__, othr.V2__);
    std::swap(ritzv__, othr.ritzv__);

    std::swap(H_, othr.H_);
    std::swap(V1_, othr.V1_);
    std::swap(V2_, othr.V2_);
    std::swap(ritzv_, othr.ritzv_);
    std::swap(resid_, othr.resid_);
  }
  */
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
}  // namespace chase
#endif  //  _CHASE_GENERA_BLAS_CHASE_BLAS_MATRICES_
