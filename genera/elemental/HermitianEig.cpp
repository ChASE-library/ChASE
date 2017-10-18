#include <El.hpp>

using CHASE_INT = int;

// Base< std::complex< Base<T> > > -> Base<T>
// Base<               Base<T>   > -> Base<T>
template <class Q>
struct Base_Class {
  typedef Q type;
};

template <class Q>
struct Base_Class<std::complex<Q>> {
  typedef Q type;
};

template <typename Q>
using Base = typename Base_Class<Q>::type;

template <class T>
class ChASE_Elemental {
 public:
  ChASE_Elemental(El::Matrix<T> H, El::Matrix<T> V, std::size_t nex)
      : H_(H), approxV_(V), workspace_(V), nex_(nex){};

  void shift(T c, bool isunshift = false) { El::ShiftDiagonal(H, c); }
  void threeTerms(CHASE_INT block, T alpha, T beta, CHASE_INT offset) {
    Hemm(El::Left, El::Lower, alpha, H_, approxV_, beta, workspace_);
    El::swap(approxV_, workspace_);
  }

  void QR(CHASE_INT fixednev) {
    std::size = approxV_.Width();
    // auto myView = El::View(Matrix<T> & B, int i, int j, int height, int
    // width)
    void qr::ExplicitUnitary(approxV_)
  }

  void RR(Base<T>* ritzv, CHASE_INT block) {
    DistMatrix<F> H_reduced(grid), V_reduced(grid);
    DistMatrix<Real, VR, STAR> Lambda_reduced(grid);

    Hemm(El::LEFT, El::LOWER, 1.0, H_, approxV_, 0.0, workspace_);
    Gemm(El::ADJOINT, El::NORMAL, 1.0, approxV_, workspace_, 0.0, H_reduced);

    // void HermitianEig(UpperOrLower uplo, AbstractDistMatrix<F> &A,
    // AbstractDistMatrix<Base<F>> &w, SortType sort = ASCENDING, const
    // HermitianEigSubset<Base<F>> subset = HermitianEigSubset<Base<F>>(), const
    // HermitianEigCtrl<Base<F>> ctrl = HermitianEigCtrl<Base<F>>())

    El::HermitianEig(El::LOWER, H_reduced, Lambda_reduced, V_reduced,
                     El::ASCENDING);

    for (i = 0; i < Lambda_reduced.Width(); ++i) {
      ritzv[i] = Lambda_reduced.Get(i, 0);
    }

    Gemm(El::NORMAL, El::NORMAL, 1.0, approxV_, V_reduced, 0.0, workspace_);
    El::swap(approxV_, workspace_);
  }

  void resd(Base<T>* ritzv, Base<T>* resid, CHASE_INT fixednev) {
    DistMatrix<Base<T>, VR, STAR> Lambda_reduced(grid);
    DistMatrix<Base<T>, VR, STAR> residuals(grid);

    for (i = 0; i < Lambda_reduced.Width(); ++i) {
      Lambda_reduced.Set(i, 0, ritzv[i]);
    }

    workspace_ = approxV_;
    El::DiagonalScale(El::RIGHT, El::NORMAL, Lambda_reduced, workspace_);
    Hemm(El::LEFT, El::LOWER, 1.0, H_, approxV_, -1.0, workspace_);

    for (int i = 0; i < approxV_.Width(); ++i) {
      auto wi = workspace_(ALL, IR(i));

      // || H x - lambda x || / ( max( ||H||, |lambda| ) )
      // norm_2 <- || H x - lambda x ||
      norm_2 = Norm(wi, ONE_NORM);
      resid[i] = norm_2 / (max(norm_H, 1.0));
    }

    for (i = 0; i < Lambda_reduced.Width(); ++i) {
      residuals.Set(i, 0, resid[i]);
    }
  }

  void swap(CHASE_INT i, CHASE_INT j) {
    ColSwap(approxV_, i, j);
    ColSwap(workspace_, i, j);
  }

  void lanczos(CHASE_INT m, Base<T>* upperb) {
    std::cout << "not implemented\n";
  };

  void lanczos(CHASE_INT M, CHASE_INT idx, Base<T>* upperb, Base<T>* ritzv,
               Base<T>* Tau, Base<T>* ritzV) {
    std::cout << "not implemented\n";
    /*
    CHASE_INT m = M;
    CHASE_INT n = N_;

    // assert( m >= 1 );

    // The first m*N part is reserved for the lanczos vectors
    DistMatrix<Real, VR, STAR> d(m, 1, grid);
    DistMatrix<Real, VR, STAR> e(m, 1, grid);
    // Base<T> *d = new Base<T>[m]();
    // Base<T> *e = new Base<T>[m]();

    DistMatrix<Real, VR, STAR> v0(n, 1, grid);
    DistMatrix<Real, VR, STAR> w_(n, 1, grid);
    // SO C++03 5.3.4[expr.new]/15
    // T* v0_ = new T[n]();
    // T* w_ = new T[n]();

    auto v0(v0_);
    auto w(w_);
    // T* w = w_;
    // T* v0 = v0_;

    T alpha = T(1.0);
    T beta = T(0.0);
    T One = T(1.0);
    T Zero = T(0.0);

    // V is filled with randomness
    // T* v1 = workspace_;
    auto v1(workspace_);
    // for (std::size_t k = 0; k < N_; ++k) {
    //   v1[k] = V_[k + idx * N_];
    // }
    v1(ALL, IR(1)) = approxV_(ALL, IR(1));

    // ENSURE that v1 has one norm
    //    Base<T> real_alpha = t_nrm2(n, v1, 1);
    Base<T> real_alpha El::Norm(v1);
    alpha = T(1 / real_alpha);

    // t_scal(n, &alpha, v1, 1);
    El::Scale(alpha, v1);

    Base<T> real_beta = 0;

    real_beta = static_cast<Base<T>>(0);

    for (std::size_t k = 0; k < m; ++k) {
      //
      // if (workspace_ + k * n != v1) {
      //   memcpy(workspace_ + k * n, v1, n * sizeof(T));
      // }

      if (workspace_(0, k).Buffer() != v1.Buffer()) {
        workspace(All, IR(k)) = v1;
      }

      // gemm_->applyVec(v1, w);
      El::Hemm(El::LEFT, El::LOWER, 1.0, H_, v1, T(0.0), w);

      // alpha = t_dot(n, v1, 1, w, 1);
      alpha = El::dot(v1, w1);

      alpha = -alpha;
      // t_axpy(n, &alpha, v1, 1, w, 1);
      El::axpy(alpha, v1, w);
      alpha = -alpha;

      d[k] = std::real(alpha);
      if (k == m - 1) break;

      beta = T(-real_beta);
      // t_axpy(n, &beta, v0, 1, w, 1);
      El::axpy(beta, v0, w);
      beta = -beta;

      // real_beta = t_nrm2(n, w, 1);
      real_beta = Norm(wi);
      beta = T(1.0 / real_beta);

      // t_scal(n, &beta, w, 1);
      Scale(beta, w);

      e[k] = real_beta;

      std::swap(v1, v0);
      std::swap(v1, w);
    }

    delete[] w_;
    delete[] v0_;

    CHASE_INT notneeded_m;
    CHASE_INT vl, vu;
    Base<T> ul, ll;
    CHASE_INT tryrac = 0;
    CHASE_INT* isuppz = new CHASE_INT[2 * m];

    t_stemr(LAPACK_COL_MAJOR, 'V', 'A', m, d, e, ul, ll, vl, vu, &notneeded_m,
            ritzv, ritzV, m, m, isuppz, &tryrac);

    *upperb = std::max(std::abs(ritzv[0]), std::abs(ritzv[m - 1])) +
              std::abs(real_beta);

    std::cout << "upperb: " << *upperb << "\n";

    for (std::size_t k = 1; k < m; ++k) {
      Tau[k] = std::abs(ritzV[k * m]) * std::abs(ritzV[k * m]);
      // std::cout << Tau[k] << "\n";
    }

    delete[] isuppz;
    delete[] d;
    delete[] e;
    */
  }

  void lock(CHASE_INT new_converged) {
    W(ALL, IR(locked_, locked_ + new_converged)) =
        V(ALL, IR(locked_, locked_ + new_converged));
    locked_ += new_converged;
  }

 private:
  std::size_t nex_, locked_;
  El::Matrix<T> H_;
  El::Matrix<T> approxV_;
  El::Matrix<T> workspace_;
};
