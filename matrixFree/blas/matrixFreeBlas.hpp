#pragma once

#include <memory>

#include "../matrixFreeInterface/matrixFreeInterface.hpp"
#include "../matrixFreeInterface/template_wrapper.hpp"

//
// A very simple implementation of MatrixFreeInterface
// We duplicate the two vector sets from ChASE_Blas and copy
// into the duplicates before each GEMM call.

template <class T>
class MatrixFreeBlas : public MatrixFreeInterface<T> {
 public:
  explicit MatrixFreeBlas(T* H, std::size_t n, std::size_t maxBlock)
      : N_(n), H_(H), V1_(new T[N_ * maxBlock]), V2_(new T[N_ * maxBlock]) {}

  MatrixFreeBlas() = delete;
  MatrixFreeBlas(MatrixFreeBlas const& rhs) = delete;

  ~MatrixFreeBlas() {}

  void preApplication(T* V, std::size_t const locked,
                      std::size_t const block) override {
    locked_ = locked;
    std::memcpy(get_V1(), V + locked * N_, N_ * block * sizeof(T));
  }

  void preApplication(T* V1, T* V2, std::size_t const locked,
                      std::size_t const block) override {
    std::memcpy(get_V2(), V2 + locked * N_, N_ * block * sizeof(T));
    this->preApplication(V1, locked, block);
  }

  void apply(T const alpha, T const beta, const std::size_t offset,
             const std::size_t block) override {
    t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,  //
           N_, block, N_,                              //
           &alpha,                                     // V2 <-
           H_, N_,                                     //      alpha * H*V1
           get_V1() + offset * N_, N_,                 //      + beta * V2
           &beta,                                      //
           get_V2() + offset * N_, N_);
    std::swap(V1_, V2_);
  }

  bool postApplication(T* V, std::size_t const block) override {
    std::memcpy(V + locked_ * N_, get_V1(), N_ * block * sizeof(T));
    return false;
  }

  void shiftMatrix(T const c) override {
    for (std::size_t i = 0; i < N_; ++i) {
      H_[i + i * N_] += c;
    }
  }

  void applyVec(T* B, T* C) override {
    T const alpha = T(1.0);
    T const beta = T(0.0);

    t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,  //
           N_, 1, N_,                                  //
           &alpha,                                     // C <-
           H_, N_,                                     //     1.0 * H*B
           B, N_,                                      //     + 0.0 * C
           &beta,                                      //
           C, N_);
  }

  void get_off(CHASE_INT* xoff, CHASE_INT* yoff, CHASE_INT* xlen,
               CHASE_INT* ylen) const override {
    *xoff = 0;
    *yoff = 0;
    *xlen = N_;
    *ylen = N_;
  }

  T* get_H() const { return H_; }

  T* get_V1() const { return V1_.get(); }
  T* get_V2() const { return V2_.get(); }

 private:
  std::size_t N_;
  std::size_t locked_;
  T* H_;
  std::unique_ptr<T> V1_;
  std::unique_ptr<T> V2_;
};
