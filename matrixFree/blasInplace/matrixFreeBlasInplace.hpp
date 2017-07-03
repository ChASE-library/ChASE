#pragma once

#include "../matrixFreeInterface/matrixFreeInterface.hpp"
#include "../matrixFreeInterface/template_wrapper.hpp"

template <class T>
class MatrixFreeBlasInplace : public MatrixFreeInterface<T> {
 public:
  MatrixFreeBlasInplace(T* H, T* V1, T* V2, std::size_t n, std::size_t maxBlock)
      : n_(n), V1_(V1), V2_(V2), H_(H) {}

  ~MatrixFreeBlasInplace() {}

  void preApplication(T* V, std::size_t locked, std::size_t block) {
    locked_ = locked;

    if (V != V1_) std::swap(V1_, V2_);
  }

  void preApplication(T* V1, T* V2, std::size_t locked, std::size_t block) {
    this->preApplication(V1, locked, block);
  }

  void apply(T alpha, T beta, std::size_t offset, std::size_t block) {
    t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,  //
           n_, block, n_,                              //
           &alpha,                                     // V2_ <-
           H_, n_,                                     //   H * V1_
           V1_ + (locked_ + offset) * n_, n_,          //   + V2_
           &beta,                                      //
           V2_ + (locked_ + offset) * n_, n_);         //

    std::swap(V1_, V2_);
  }

  bool postApplication(T* V, std::size_t block) {
    // this is somewhat a hack, but causes the approxV in the next
    // preApplication to be the same pointer content as V1_
    std::swap(V1_, V2_);
    return false;
  }

  void shiftMatrix(T c) {
    for (std::size_t i = 0; i < n_; ++i) {
      H_[i + i * n_] += c;
    }
  }

  void applyVec(T* B, T* C) override {
    T alpha = T(1.0);
    T beta = T(0.0);

    t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,  //
           n_, 1, n_,                                  //
           &alpha,                                     //
           H_, n_,                                     //
           B, n_,                                      //
           &beta,                                      //
           C, n_);                                     //
  }

  T* get_H() const override { return H_; }

  void get_off(CHASE_INT* xoff, CHASE_INT* yoff, CHASE_INT* xlen,
               CHASE_INT* ylen) const override {
    *xoff = 0;
    *yoff = 0;
    *xlen = n_;
    *ylen = n_;
  }

 private:
  std::size_t n_;
  std::size_t locked_;

  T* H_;
  T* V1_;
  T* V2_;
};
