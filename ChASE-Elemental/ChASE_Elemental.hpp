/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials, 
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#ifndef CHASE_CHASE_ELEMENTAL_CHASE_ELEMENTAL_HPP
#define CHASE_CHASE_ELEMENTAL_CHASE_ELEMENTAL_HPP

//#include <El.hpp>
#include <elemental.hpp>

#include "algorithm/chase.hpp"

namespace chase {
namespace elemental {

using namespace chase;
namespace El = elem;

template <class T>
class ChaseElemental : public Chase<T> {
 public:
  ChaseElemental(std::size_t N, std::size_t nev, std::size_t nex,
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
    El::UpdateDiagonal(H_, c);
  }

  void HEMM(std::size_t block, T alpha, T beta, std::size_t offset) override {
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

  void QR(std::size_t fixednev) override {
    auto locked_vectors = El::LockedView(*approxV_, 0, 0, N_, locked_);
    auto saved_locked_vectors = El::LockedView(*workspace_, 0, 0, N_, locked_);

    saved_locked_vectors = locked_vectors;
    El::qr::Explicit(*approxV_);
    locked_vectors = saved_locked_vectors;
  }

  void RR(Base<T>* ritzv, std::size_t block) override {
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
  }

  void Resd(Base<T>* ritzv, Base<T>* resid, std::size_t fixednev) override {
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

    Base<T> norm_2;
    for (std::size_t i = 0; i < workspace.Width(); ++i) {
      auto wi = El::View(workspace, 0, i, N_, 1);
      // || H x - lambda x || / ( max( ||H||, |lambda| ) )
      // norm_2 <- || H x - lambda x ||
      norm_2 = El::Nrm2(wi);
      resid[i] = norm_2;
    }
  }

  void Swap(std::size_t i, std::size_t j) override {
    El::ColSwap(*approxV_, i, j);
    El::ColSwap(*workspace_, i, j);
  }

  void Lanczos(std::size_t m, Base<T>* upperb) override {
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

  void Lanczos(std::size_t m, std::size_t idx, Base<T>* upperb, Base<T>* ritzv,
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

    // El::Display(e, "e");
    // El::Display(d, "d");

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

    // El::Display(lambda, "lamdba");
  }

  void Lock(std::size_t new_converged) override {
    // auto workspace_new = El::View(*workspace_, El::IR(0, N_),
    //                               El::IR(locked_, locked_ + new_converged));
    // workspace_new = El::View(*approxV_, El::IR(0, N_),
    //                          El::IR(locked_, locked_ + new_converged));

    auto workspace_new = El::View(*workspace_, 0, locked_, N_, new_converged);
    workspace_new = El::View(*approxV_, 0, locked_, N_, new_converged);

    locked_ += new_converged;
  }

  void LanczosDos(std::size_t idx, std::size_t m, T* ritzVc) override {
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

  std::size_t GetNev() override { return nev_; }
  std::size_t GetNex() override { return nex_; }

  Base<T>* GetRitzv() override { return ritzv_.data(); }
  Base<T>* GetResid() { return resid_.data(); }

  ChaseConfig<T>& GetConfig() { return config_; }

  void Start() { locked_ = 0; }
  void End() {}

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

 private:
  std::size_t const N_;
  std::size_t const nev_;
  std::size_t const nex_;
  std::size_t locked_;

  ChaseConfig<T> config_;

  El::DistMatrix<T>& H_;
  El::DistMatrix<T> V_;
  El::DistMatrix<T> W_;

  //  El::DistMatrix<Base<T>, El::VR, El::STAR> ritzv_;
  std::vector<Base<T>> ritzv_;
  std::vector<Base<T>> resid_;

  El::DistMatrix<T>* approxV_;
  El::DistMatrix<T>* workspace_;
};
}  // namespace elemental
}  // namespace chase
#endif
