/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
#pragma once

//#include <El.hpp>
#include <elemental.hpp>

#include "algorithm/chase.h"
#include "omp.h"

using namespace chase;
namespace El = elem;

template <class T>
class ElementalChase : public Chase<T> {
 public:
  ElementalChase(std::size_t N, std::size_t nev, std::size_t nex,
                 El::DistMatrix<T>& H)
      : N_(N),
        nev_(nev),
        nex_(nex),
        locked_(0),
        config_(N, nev, nex),
        H_(H),
        V_(N_, nev_ + nex_, H_.Grid()),
        W_(N_, nev_ + nex_, H_.Grid()),
        // ritzv_(H_.Grid()),
        ritzv_(nev_ + nex_),
        resid_(nev_ + nex_),
        approxV_(&V_),
        workspace_(&W_) {
    El::MakeGaussian(V_);
    El::MakeGaussian(W_);
  };

  void Shift(T c, bool isunshift = false) override {
    // El::ShiftDiagonal(H_, c);
    El::UpdateDiagonal(H_, c);
  }

  void HEMM(CHASE_INT block, T alpha, T beta, CHASE_INT offset) override {
    // auto approxV = El::LockedView(*approxV_, El::IR(0, N_),
    // GetActive(offset));
    // auto workspace = El::View(*workspace_, El::IR(0, N_), GetActive(offset));
    auto approxV = El::LockedView(*approxV_, 0, locked_ + offset, N_,
                                  nex_ + nev_ - locked_ - offset);
    auto workspace = El::View(*workspace_, 0, locked_ + offset, N_,
                              nex_ + nev_ - locked_ - offset);

    El::Hemm(El::LEFT, El::LOWER,  //
             alpha,                //
             H_,                   //
             approxV,              //
             beta,                 //
             workspace             //
    );                             //

    std::swap(approxV_, workspace_);
  }

  void QR(CHASE_INT fixednev) override {
    auto locked_vectors = El::LockedView(*approxV_, 0, 0, N_, locked_);
    auto saved_locked_vectors = El::LockedView(*workspace_, 0, 0, N_, locked_);

    // auto locked_vectors =
    //     El::View(*approxV_, El::IR(0, N_), El::IR(0, locked_));
    // auto saved_locked_vectors =
    //     El::View(*workspace_, El::IR(0, N_), El::IR(0, locked_));

    saved_locked_vectors = locked_vectors;

    // El::qr::ExplicitUnitary(*approxV_);
    El::qr::Explicit(*approxV_);

    locked_vectors = saved_locked_vectors;
  }

  void RR(Base<T>* ritzv, CHASE_INT block) override {
    //*
    El::Grid const& grid = H_.Grid();
    El::DistMatrix<T> H_reduced{grid};
    El::DistMatrix<T> V_reduced{grid};
    El::DistMatrix<Base<T>, El::VR, El::STAR> ritzv_tmp{block, 1, grid};

    auto approxV =
        El::LockedView(*approxV_, 0, locked_, N_, nex_ + nev_ - locked_);
    auto workspace =
        El::View(*workspace_, 0, locked_, N_, nex_ + nev_ - locked_);

    H_reduced.AlignWith(approxV);
    H_reduced.Resize(block, block);

    El::Hemm(El::LEFT, El::LOWER,  //
             T(1.0),               //
             H_,                   //
             approxV,              //
             T(0.0),               //
             workspace             //
    );

    El::Gemm(El::ADJOINT, El::NORMAL,  //
             T(1.0),                   //
             approxV,                  //
             workspace,                //
             T(0.0),                   //
             H_reduced                 //
    );                                 //

    El::HermitianEig(El::LOWER, H_reduced, ritzv_tmp, V_reduced, El::ASCENDING);

    for (std::size_t i = 0; i < ritzv_tmp.Height(); ++i)
      ritzv[i] = ritzv_tmp.Get(i, 0);

    // Back transformation.
    H_reduced.AlignWith(approxV);
    H_reduced = V_reduced;

    El::Gemm(El::NORMAL, El::NORMAL,  //
             T(1.0),                  //
             approxV,                 //
             H_reduced,               //
             T(0.0),                  //
             workspace                //
    );

    std::swap(approxV_, workspace_);

    /*/
    // version w/o locking
    block = nev_ + nex_;
    El::Grid const& grid = H_.Grid();
    El::DistMatrix<T> H_reduced{block, block, grid};
    El::DistMatrix<T> V_reduced{block, block, grid};
    El::DistMatrix<Base<T>, El::VR, El::STAR> ritzv_tmp{block, 1, grid};

    auto approxV = El::View(*approxV_, 0, 0, N_, nev_ + nex_);
    auto workspace = El::View(*workspace_, 0, 0, N_, nev_ + nex_);

    El::Hemm(El::LEFT, El::LOWER,  //
             T(1.0),               //
             H_,                   //
             approxV,              //
             T(0.0),               //
             workspace             //
             );

    El::Gemm(El::ADJOINT, El::NORMAL,  //
             T(1.0),                   //
             approxV,                  //
             workspace,                //
             T(0.0),                   //
             H_reduced                 //
             );                        //

    El::HermitianEig<T>(El::LOWER, H_reduced, ritzv_tmp, V_reduced,
                        El::ASCENDING);
    for (std::size_t i = locked_; i < ritzv_tmp.Height(); ++i)
      ritzv[i - locked_] = ritzv_tmp.Get(i, 0);

    El::Gemm(El::NORMAL, El::NORMAL, T(1.0), approxV, V_reduced, T(0.0),
             workspace);
    // std::swap(approxV_, workspace_);
    El::Copy(workspace, approxV);
    //*/
  }

  void Resd(Base<T>* ritzv, Base<T>* resid, CHASE_INT fixednev) override {
    //*
    // auto approxV = El::LockedView(*approxV_, El::IR(0, N_), GetActive());
    // auto workspace = El::View(*workspace_, El::IR(0, N_), GetActive());
    // auto approxV = El::LockedView(*approxV_, 0, locked_, N_,
    // nev_+nex_-locked_);
    // auto workspace = El::View(*workspace_, 0, locked_, N_,
    // nev_+nex_-locked_);
    // auto ritzv_reduced = El::View(ritzv_, El::IR(0,1), GetActive());

    auto approxV =
        El::LockedView(*approxV_, 0, locked_, N_, nex_ + nev_ - locked_);
    auto workspace =
        El::View(*workspace_, 0, locked_, N_, nex_ + nev_ - locked_);

    El::DistMatrix<Base<T>, El::STAR, El::STAR> ritzv_reduced{
        nev_ + nex_ - locked_, 1, H_.Grid()};

    for (std::size_t i = 0; i < nev_ + nex_ - locked_; ++i)
      ritzv_reduced.Set(i, 0, ritzv[i]);

    El::Copy(approxV, workspace);
    El::DiagonalScale(El::RIGHT, El::NORMAL, ritzv_reduced, workspace);
    El::Hemm(El::LEFT, El::LOWER, T(1.0), H_, approxV, T(-1.0), workspace);

    // El::Hemm(El::LEFT, El::LOWER, T(1.0), H_, approxV, T(0.0), workspace);

    Base<T> norm_2;
    for (std::size_t i = 0; i < workspace.Width(); ++i) {
      // auto wi = El::View(workspace, El::IR(0, N_), El::IR(i, i + 1));
      auto wi = El::View(workspace, 0, i, N_, 1);
      // auto vi = El::View(approxV, El::IR(0, N_), El::IR(i, i + 1));

      // El::Axpy(-ritzv[i], vi, wi);

      // || H x - lambda x || / ( max( ||H||, |lambda| ) )
      // norm_2 <- || H x - lambda x ||
      norm_2 = El::Nrm2(wi);
      resid[i] = norm_2;
    }

    /*/

    El::DistMatrix<Base<T>, El::STAR, El::STAR> el_ritzv{nev_ + nex_, 1,
                                                         H_.Grid()};
    for (std::size_t i = 0; i < nev_ + nex_; ++i) el_ritzv.Set(i, 0, ritzv_[i]);

    // Version without locking
    El::Copy(*approxV_, *workspace_);
    El::DiagonalScale(El::RIGHT, El::NORMAL, el_ritzv, *workspace_);
    El::Hemm(El::LEFT, El::LOWER, T(1.0), H_, *approxV_, T(-1.0), *workspace_);

    for (std::size_t i = locked_; i < workspace_->Width(); ++i) {
      auto wi = El::View(*workspace_, 0, i, N_, 1);

      // || H x - lambda x || / ( max( ||H||, |lambda| ) )
      // norm_2 <- || H x - lambda x ||
      Base<T> norm_2 = El::Nrm2(wi);

      resid[i - locked_] = norm_2 / (std::max(this->GetNorm(), 1.0));
    }
    //*/
  }

  void Swap(CHASE_INT i, CHASE_INT j) override {
    El::ColSwap(*approxV_, i, j);
    El::ColSwap(*workspace_, i, j);
  }

  void Lanczos(CHASE_INT m, Base<T>* upperb) override {
    T alpha;
    T beta = 0;

    El::DistMatrix<Base<T>> d{m, 1, H_.Grid()};
    El::DistMatrix<Base<T>> e{m, 1, H_.Grid()};

    El::DistMatrix<T> v1{N_, 1, H_.Grid()};
    El::DistMatrix<T> v0{N_, 1, H_.Grid()};
    El::DistMatrix<T> w{N_, 1, H_.Grid()};

    Zeros(v0, N_, 1);
    Zeros(w, N_, 1);
    El::MakeGaussian(v1);
    El::Scale(1 / El::Nrm2(v1), v1);

    for (std::size_t k = 0; k < m; ++k) {
      El::Hemm(El::LEFT, El::LOWER,  //
               T(1.0),               //
               H_,                   //
               v1,                   //
               T(0.0),               //
               w                     //
      );                             //

      alpha = El::Dot(v1, w);

      El::Axpy(-alpha, v1, w);

      d.Set(k, 0, std::real(alpha));
      if (k == m - 1) break;

      El::Axpy(-beta, v0, w);
      beta = El::Nrm2(w);

      El::Scale(1.0 / beta, w);
      e.Set(k, 0, std::real(beta));

      std::swap(v1, v0);
      std::swap(v1, w);
    }

    El::DistMatrix<Base<T>, El::STAR, El::STAR> lambda{m, 1, H_.Grid()};
    El::DistMatrix<Base<T>, El::STAR, El::STAR> ElRitzV{m, m, H_.Grid()};

    El::HermitianTridiagEig(d, e, lambda, ElRitzV, El::ASCENDING);

    *upperb =
        std::max(std::abs(lambda.Get(0, 0)), std::abs(lambda.Get(m - 1, 0))) +
        std::abs(std::real(beta));
  };

  void Lanczos(CHASE_INT m, CHASE_INT idx, Base<T>* upperb, Base<T>* ritzv,
               Base<T>* Tau, Base<T>* ritzV) override {
    T alpha;
    T beta = 0;

    El::DistMatrix<Base<T>, El::STAR, El::STAR> d{m, 1, H_.Grid()};
    El::DistMatrix<Base<T>, El::STAR, El::STAR> e{m, 1, H_.Grid()};

    El::DistMatrix<T> v1{N_, 1, H_.Grid()};
    El::DistMatrix<T> v0{N_, 1, H_.Grid()};
    El::DistMatrix<T> w{N_, 1, H_.Grid()};

    Zeros(v0, N_, 1);
    Zeros(w, N_, 1);
    El::MakeGaussian(v1);
    El::Scale(1 / El::Nrm2(v1), v1);

    for (std::size_t k = 0; k < m; ++k) {
      // save vector LanczosDos
      // auto ww = El::View(*workspace_, El::IR(0, N_), El::IR(k, k + 1));
      auto ww = El::View(*workspace_, 0, k, N_, 1);
      ww = v1;
      //      El::Copy(v1, ww);

      El::Hemm(El::LEFT, El::LOWER,  //
               T(1.0),               //
               H_,                   //
               v1,                   //
               T(0.0),               //
               w                     //
      );                             //

      alpha = El::Dot(v1, w);

      El::Axpy(-alpha, v1, w);

      d.Set(k, 0, std::real(alpha));
      if (k == m - 1) break;

      El::Axpy(-beta, v0, w);
      beta = El::Nrm2(w);

      El::Scale(1.0 / beta, w);
      e.Set(k, 0, std::real(beta));

      std::swap(v1, v0);
      std::swap(v1, w);
    }

    El::DistMatrix<Base<T>, El::STAR, El::STAR> lambda{m, 1, H_.Grid()};
    El::DistMatrix<Base<T>, El::STAR, El::STAR> ElRitzV{m, m, H_.Grid()};

    El::Display(e, "e");
    El::Display(d, "d");

    El::HermitianTridiagEig(d, e, lambda, ElRitzV, El::ASCENDING);

    for (std::size_t k = 0; k < m; ++k) {
      ritzv[k] = lambda.Get(k, 0);
    }

    *upperb = std::max(std::abs(ritzv[0]), std::abs(ritzv[m - 1])) +
              std::abs(std::real(beta));

    for (std::size_t k = 1; k < m; ++k) {
      Tau[k] = std::abs(ElRitzV.Get(0, k)) * std::abs(ElRitzV.Get(0, k));
      // std::cout << Tau[k] << "\n";
    }

    El::Display(lambda, "lamdba");
  }

  void Lock(CHASE_INT new_converged) override {
    // auto workspace_new = El::View(*workspace_, El::IR(0, N_),
    //                               El::IR(locked_, locked_ + new_converged));
    // workspace_new = El::View(*approxV_, El::IR(0, N_),
    //                          El::IR(locked_, locked_ + new_converged));

    auto workspace_new = El::View(*workspace_, 0, locked_, N_, new_converged);
    workspace_new = El::View(*approxV_, 0, locked_, N_, new_converged);

    locked_ += new_converged;
  }

  void LanczosDos(CHASE_INT idx, CHASE_INT m, T* ritzVc) override {
    // we saved the lanczos vectors in workspace
    // auto lanczos_vectors = El::View(*workspace_, El::IR(0, N_), El::IR(0,
    // m));
    // auto dos_vectors = El::View(*approxV_, El::IR(0, N_), El::IR(0, idx));

    auto lanczos_vectors = El::View(*workspace_, 0, 0, N_, m);
    auto dos_vectors = El::View(*approxV_, 0, 0, N_, idx);

    El::DistMatrix<T> ritz_vectors{m, idx, H_.Grid()};
    for (std::size_t i = 0; i < m; ++i)
      for (std::size_t j = 0; j < idx; ++j)
        ritz_vectors.Set(i, j, ritzVc[j * m + i]);

    El::Gemm(El::NORMAL, El::NORMAL,         //
             T(0.0),                         //
             lanczos_vectors, ritz_vectors,  //
             T(1.0),                         //
             dos_vectors                     //
    );                                       //
  }

  std::size_t GetN() const override { return H_.Width(); }

  CHASE_INT GetNev() override { return nev_; }
  CHASE_INT GetNex() override { return nex_; }

  Base<T>* GetRitzv() override { return ritzv_.data(); }
  Base<T>* GetResid() { return resid_.data(); }

  ChaseConfig<T>& GetConfig() { return config_; }

  void Reset() { locked_ = 0; }

  void GetOff(CHASE_INT* xoff, CHASE_INT* yoff, CHASE_INT* xlen,
              CHASE_INT* ylen) {
    Output("not implemented\n");
  }

#ifdef OUTPUT
  void Output(std::string str) override {
    MPI_Barrier(MPI_COMM_WORLD);

    if (H_.Grid().Rank() == 0) std::cout << str;
  }
#else
  void Output(std::string str) {
    MPI_Barrier(MPI_COMM_WORLD);

    if (H_.Grid().Rank() == 0) std::cout << str;
  }
#endif

  El::DistMatrix<T>& GetV() { return *approxV_; }
  // El::DistMatrix<Base<T>, El::VR, El::STAR>& GetRitzvEl() { return ritzv_; }

  ChasePerfData GetPerfData() { return perf_; }

 private:
  std::size_t const N_;
  std::size_t const nev_;
  std::size_t const nex_;
  std::size_t locked_;

  ChaseConfig<T> config_;
  ChasePerfData perf_;

  El::DistMatrix<T>& H_;
  El::DistMatrix<T> V_;
  El::DistMatrix<T> W_;

  //  El::DistMatrix<Base<T>, El::VR, El::STAR> ritzv_;
  std::vector<Base<T>> ritzv_;
  std::vector<Base<T>> resid_;

  El::DistMatrix<T>* approxV_;
  El::DistMatrix<T>* workspace_;
};
