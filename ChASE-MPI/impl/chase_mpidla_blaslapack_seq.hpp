/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2023, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <cstring>
#include <memory>

#include "ChASE-MPI/blas_templates.hpp"
#include "ChASE-MPI/chase_mpi_matrices.hpp"

namespace chase
{
namespace mpi
{


//! @brief A derived class of ChaseMpiDLAInterface which implements ChASE targeting
//! shared-memory architectures with only CPUs available.
template <class T>
class ChaseMpiDLABlaslapackSeq : public ChaseMpiDLAInterface<T>
{
public:
    //! A constructor of ChaseMpiDLABlaslapackSeq.
    /*! @param matrices: it is an object of ChaseMpiMatrices, which allocates
       the required buffer
        @param n: size of matrix defining the eigenproblem
        @param nev: number of eigenpairs to be computed
        @param nex: size of extral searching space
    */
    explicit ChaseMpiDLABlaslapackSeq(ChaseMpiMatrices<T>& matrices,
                                      std::size_t n, std::size_t nex,
                                      std::size_t nev)
        : N_(n), nev_(nev), nex_(nex), maxBlock_(nex + nex),
          H_(matrices.get_H()), V1_(new T[N_ * maxBlock_]),
          V2_(new T[N_ * maxBlock_]), A_(new T[maxBlock_ * maxBlock_])
    {

        V12_ = matrices.get_V1();
        V22_ = matrices.get_V2();
        v0_.resize(N_);
        v1_.resize(N_);
        w_.resize(N_);
    }

    ChaseMpiDLABlaslapackSeq() = delete;
    //! Destructor
    ChaseMpiDLABlaslapackSeq(ChaseMpiDLABlaslapackSeq const& rhs) = delete;

    ~ChaseMpiDLABlaslapackSeq() {}
    void initVecs() override
    {
        next_ = NextOp::bAc;
        t_lacpy('A', N_, nev_ + nex_, V12_, N_, get_V1(), N_);
    }
    void initRndVecs() override
    {
        std::mt19937 gen(1337.0);
        std::normal_distribution<> d;
        for (auto j = 0; j < (nev_ + nex_); j++)
        {
            for (auto i = 0; i < N_; i++)
            {
                V12_[i + j * N_] = getRandomT<T>([&]() { return d(gen); });
            }
        }
    }

    // v1->v2
    void V2C(T* v1, std::size_t off1, T* v2, std::size_t off2,
             std::size_t block) override
    {
        std::memcpy(v2 + off2 * N_, v1 + off1 * N_, N_ * block * sizeof(T));
    }

    // v1->v2;
    void C2V(T* v1, std::size_t off1, T* v2, std::size_t off2,
             std::size_t block) override
    {
        std::memcpy(v2 + off2 * N_, v1 + off1 * N_, N_ * block * sizeof(T));
    }


    void preApplication(T* V, std::size_t const locked,
                        std::size_t const block) override
    {
        next_ = NextOp::bAc;
        locked_ = locked;
        std::memcpy(V12_, V + locked * N_, N_ * block * sizeof(T));
    }

    void apply(T alpha, T beta, std::size_t offset, std::size_t block,
               std::size_t locked) override
    {

        if (next_ == NextOp::bAc)
        {
            t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, N_,
                   static_cast<std::size_t>(block), N_, &alpha, H_, N_,
                   V12_ + offset * N_ + locked * N_, N_, &beta,
                   V22_ + locked * N_ + offset * N_, N_);

            next_ = NextOp::cAb;
        }
        else
        {
            t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N_,
                   static_cast<std::size_t>(block), N_, &alpha, H_, N_,
                   V22_ + offset * N_ + locked * N_, N_, &beta,
                   V12_ + offset * N_ + locked * N_, N_);

            next_ = NextOp::bAc;
        }
    }

    bool postApplication(T* V, std::size_t const block,
                         std::size_t locked) override
    {
        if (next_ == NextOp::bAc)
        {
            std::memcpy(V + locked_ * N_, V12_, N_ * block * sizeof(T));
        }
        else
        {
            std::memcpy(V + locked_ * N_, V22_, N_ * block * sizeof(T));
        }
        return true;
    }

    void shiftMatrix(T const c, bool isunshift = false) override
    {
        for (std::size_t i = 0; i < N_; ++i)
        {
            H_[i + i * N_] += c;
        }
    }

    void asynCxHGatherC(std::size_t locked, std::size_t block,
                        bool isCcopied = false) override
    {
        T const alpha = T(1.0);
        T const beta = T(0.0);

        t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, N_, block, N_,
               &alpha, H_, N_, V12_ + locked * N_, N_, &beta,
               V22_ + locked * N_, N_);

        std::memcpy(get_V2() + locked * N_, get_V1() + locked * N_,
                    N_ * block * sizeof(T));
    }


    void applyVec(T* B, T* C) override
    {
        T One = T(1.0);
        T Zero = T(0.0);

        this->preApplication(B, 0, 1);
        this->apply(One, Zero, 0, 1, 0);
        this->postApplication(C, 1, 0);
    }

    //! return of a pointer to a matrix of size `N_*(nev_+nex_)` allocated in this class
    T* get_V1() const { return V1_.get(); }

    //! return of a pointer to a matrix of size `N_*(nev_+nex_)` allocated in this class
    T* get_V2() const { return V2_.get(); }

    int get_nprocs() const override { return 1; }

    void Start() override {}
    void End() override {}


    void axpy(std::size_t N, T* alpha, T* x, std::size_t incx, T* y,
              std::size_t incy) override
    {
        t_axpy(N, alpha, x, incx, y, incy);
    }

    void scal(std::size_t N, T* a, T* x, std::size_t incx) override
    {
        t_scal(N, a, x, incx);
    }

    Base<T> nrm2(std::size_t n, T* x, std::size_t incx) override
    {
        return t_nrm2(n, x, incx);
    }

    T dot(std::size_t n, T* x, std::size_t incx, T* y,
          std::size_t incy) override
    {
        return t_dot(n, x, incx, y, incy);
    }

    void RR(std::size_t block, std::size_t locked, Base<T>* ritzv) override
    {
        T One = T(1.0);
        T Zero = T(0.0);

        this->asynCxHGatherC(locked, block);

        auto A = std::unique_ptr<T[]>{new T[block * block]};

        // A <- W' * V
        t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, block, block, N_,
               &One, get_V2() + locked * N_, N_, V22_ + locked * N_, N_, &Zero,
               A.get(), block);

        t_heevd(LAPACK_COL_MAJOR, 'V', 'L', block, A.get(), block, ritzv);

        t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N_, block, block,
               &One, get_V1() + locked * N_, N_, A.get(), block, &Zero,
               V12_ + locked * N_, N_);

        std::memcpy(get_V1() + locked * N_, V12_ + locked * N_,
                    N_ * block * sizeof(T));
    }

    void syherk(char uplo, char trans, std::size_t n, std::size_t k, T* alpha,
                T* a, std::size_t lda, T* beta, T* c, std::size_t ldci,
                bool first = true) override
    {
    }

    int potrf(char uplo, std::size_t n, T* a, std::size_t lda) override
    {
        return 0;
    }

    void trsm(char side, char uplo, char trans, char diag, std::size_t m,
              std::size_t n, T* alpha, T* a, std::size_t lda, T* b,
              std::size_t ldb, bool first = false) override
    {
    }

    void heevd(int matrix_layout, char jobz, char uplo, std::size_t n, T* a,
               std::size_t lda, Base<T>* w) override
    {
    }

    void Resd(Base<T>* ritzv, Base<T>* resid, std::size_t locked,
              std::size_t unconverged) override
    {
        T alpha = T(1.0);
        T beta = T(0.0);

        this->asynCxHGatherC(locked, unconverged);

        for (std::size_t i = 0; i < unconverged; ++i)
        {
            beta = T(-ritzv[i]);
            t_axpy(N_, &beta, (get_V2() + locked * N_) + N_ * i, 1,
                   (V22_ + locked * N_) + N_ * i, 1);

            resid[i] = nrm2(N_, (V22_ + locked * N_) + N_ * i, 1);
        }
    }

    void hhQR(std::size_t locked) override
    {
        auto nevex = nev_ + nex_;

        std::unique_ptr<T[]> tau(new T[nevex]);

        t_geqrf(LAPACK_COL_MAJOR, N_, nevex, V12_, N_, tau.get());
        t_gqr(LAPACK_COL_MAJOR, N_, nevex, nevex, V12_, N_, tau.get());

        std::memcpy(V12_, get_V1(), locked * N_ * sizeof(T));
        std::memcpy(get_V1() + locked * N_, V12_ + locked * N_,
                    (nevex - locked) * N_ * sizeof(T));
    }

    void cholQR(std::size_t locked, Base<T> cond) override
    {
        auto nevex = nev_ + nex_;

        auto A_ = std::unique_ptr<T[]>{new T[nevex * nevex]};
        T one = T(1.0);
        T zero = T(0.0);
        int info = -1;

        t_syherk('U', 'C', nevex, N_, &one, V12_, N_, &zero, A_.get(), nevex);
        info = t_potrf('U', nevex, A_.get(), nevex);

        if (info == 0)
        {
            t_trsm('R', 'U', 'N', 'N', N_, nevex, &one, A_.get(), nevex, V12_,
                   N_);

            int choldeg = 2;
            char* choldegenv;
            choldegenv = getenv("CHASE_CHOLQR_DEGREE");
            if (choldegenv)
            {
                choldeg = std::atoi(choldegenv);
            }
#ifdef CHASE_OUTPUT
            std::cout << "choldegee: " << choldeg << std::endl;
#endif
            for (auto i = 0; i < choldeg - 1; i++)
            {
                t_syherk('U', 'C', nevex, N_, &one, V12_, N_, &zero, A_.get(),
                         nevex);
                t_potrf('U', nevex, A_.get(), nevex);
                t_trsm('R', 'U', 'N', 'N', N_, nevex, &one, A_.get(), nevex,
                       V12_, N_);
            }
            std::memcpy(V12_, get_V1(), locked * N_ * sizeof(T));
            std::memcpy(get_V1() + locked * N_, V12_ + locked * N_,
                        (nevex - locked) * N_ * sizeof(T));
        }
        else
        {
#ifdef CHASE_OUTPUT
            std::cout << "cholQR failed because of ill-conditioned vector, use "
                         "Householder QR instead"
                      << std::endl;
#endif
            this->hhQR(locked);
        }
    }

    void Swap(std::size_t i, std::size_t j) override
    {
        T* tmp = new T[N_];

        memcpy(tmp, V12_ + N_ * i, N_ * sizeof(T));
        memcpy(V12_ + N_ * i, V12_ + N_ * j, N_ * sizeof(T));
        memcpy(V12_ + N_ * j, tmp, N_ * sizeof(T));

        memcpy(tmp, get_V1() + N_ * i, N_ * sizeof(T));
        memcpy(get_V1() + N_ * i, get_V1() + N_ * j, N_ * sizeof(T));
        memcpy(get_V1() + N_ * j, tmp, N_ * sizeof(T));

        delete[] tmp;
    }

    void getLanczosBuffer(T** V1, T** V2, std::size_t* ld, T** v0, T** v1,
                          T** w) override
    {
        *V1 = V12_;
        *V2 = get_V1();
        *ld = N_;

        std::fill(v1_.begin(), v1_.end(), T(0));
        std::fill(v0_.begin(), v0_.end(), T(0));
        std::fill(w_.begin(), w_.end(), T(0));

        *v0 = v0_.data();
        *v1 = v1_.data();
        *w = w_.data();
    }

    void getLanczosBuffer2(T** v0, T** v1, T** w) override
    {
        std::fill(v0_.begin(), v0_.end(), T(0));
        std::fill(w_.begin(), w_.end(), T(0));
        std::mt19937 gen(2342.0);
        std::normal_distribution<> normal_distribution;

        for (std::size_t k = 0; k < N_; ++k)
        {
            v1_[k] = getRandomT<T>([&]() { return normal_distribution(gen); });
        }
        *v0 = v0_.data();
        *v1 = v1_.data();
        *w = w_.data();
    }

    void LanczosDos(std::size_t idx, std::size_t m, T* ritzVc) override
    {
        T alpha = T(1.0);
        T beta = T(0.0);

        t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N_, idx, m, &alpha,
               V12_, N_, ritzVc, m, &beta, get_V1(), N_);
        std::memcpy(V12_, get_V1(), m * N_ * sizeof(T));
    }

private:
    enum NextOp
    {
        cAb,
        bAc
    };
    NextOp next_; //!< it is to manage the switch of operation from `V2=H*V1` to `V1=H'*V2` in filter

    std::size_t N_; //!< global dimension of the symmetric/Hermtian matrix
    std::size_t locked_;  //!< number of converged eigenpairs
    std::size_t maxBlock_; //!< `maxBlock_=nev_ + nex_`
    std::size_t nex_; //!< number of extral searching space
    std::size_t nev_; //!< number of required eigenpairs
    T* H_; //!< a pointer to the Symmetric/Hermtian matrix
    std::unique_ptr<T> V1_; //!< a matrix of size `N_*(nev_+nex_)`
    std::unique_ptr<T> V2_; //!< a matrix of size `N_*(nev_+nex_)`
    std::unique_ptr<T> A_; //!< a matrix of size `(nev_+nex_)*(nev_+nex_)`
    std::vector<T> v0_; //!< a vector of size `N_`, which is allocated in this class for Lanczos
    std::vector<T> v1_; //!< a vector of size `N_`, which is allocated in this class for Lanczos
    std::vector<T> w_; //!< a vector of size `N_`, which is allocated in this class for Lanczos
    T* V12_; //!< a pointer to a matrix of size `N_*(nev_+nex_)`
    T* V22_; //!< a pointer to a matrix of size `N_*(nev_+nex_)`
};

template <typename T>
struct is_skewed_matrixfree<ChaseMpiDLABlaslapackSeq<T>>
{
    static const bool value = false;
};

} // namespace mpi
} // namespace chase
