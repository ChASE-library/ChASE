/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2023, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "ChASE-MPI/blas_templates.hpp"
#include "ChASE-MPI/chase_mpi_properties.hpp"
#include "ChASE-MPI/chase_mpidla_interface.hpp"

namespace chase
{
namespace mpi
{
//
//
//! @brief A derived class of ChaseMpiDLAInterface which implements the
//! inter-node computation for a pure-CPU MPI-based implementation of ChASE.
template <class T>
class ChaseMpiDLABlaslapack : public ChaseMpiDLAInterface<T>
{
public:
    //! A constructor of ChaseMpiDLABlaslapack.
    //! @param matrix_properties: it is an object of ChaseMpiProperties, which
    //! defines the MPI environment and data distribution scheme in ChASE-MPI.
    //! @param matrices: it is an instance of ChaseMpiMatrices, which
    //!  allocates the required buffers in ChASE-MPI.
    ChaseMpiDLABlaslapack(ChaseMpiProperties<T>* matrix_properties, T* H,
                          std::size_t ldh, T* V1, Base<T>* ritzv)
        : matrices_(std::move(
              matrix_properties->create_matrices(0, H, ldh, V1, ritzv)))
    {
        // TODO
        n_ = matrix_properties->get_n();
        m_ = matrix_properties->get_m();
        N_ = matrix_properties->get_N();
        nev_ = matrix_properties->GetNev();
        nex_ = matrix_properties->GetNex();
        ldh_ = matrices_.get_ldh();
        H_ = matrices_.H().ptr();
        C_ = matrices_.C().ptr();
        C2_ = matrices_.C2().ptr();
        B_ = matrices_.B().ptr();
        B2_ = matrices_.B2().ptr();
        A_ = matrices_.A().ptr();
        resid_ = matrices_.Resid().ptr();
        ritzv_ = matrices_.Ritzv().ptr();
        off_ = matrix_properties->get_off();

        matrix_properties->get_offs_lens(r_offs_, r_lens_, r_offs_l_, c_offs_,
                                         c_lens_, c_offs_l_);
        mb_ = matrix_properties->get_mb();
        nb_ = matrix_properties->get_nb();

        mblocks_ = matrix_properties->get_mblocks();
        nblocks_ = matrix_properties->get_nblocks();

        matrix_properties_ = matrix_properties;

        MPI_Comm row_comm = matrix_properties_->get_row_comm();
        MPI_Comm col_comm = matrix_properties_->get_col_comm();

        MPI_Comm_rank(row_comm, &mpi_row_rank);
        MPI_Comm_rank(col_comm, &mpi_col_rank);

        vv_ = matrices_.vv().ptr();
    }

    ~ChaseMpiDLABlaslapack() {}
    //! This function set initially the operation for apply() used in
    //! ChaseMpi::Lanczos()
    void initVecs() override
    {
        t_lacpy('A', m_, nev_ + nex_, C_, m_, C2_, m_);
        next_ = NextOp::bAc;
    }
    //! This function generates the random values for each MPI proc using C++
    //! STL
    //!     - each MPI proc with a same MPI rank among different column
    //!     communicator
    //!       same a same seed of RNG
    void initRndVecs() override
    {
        std::mt19937 gen(1337.0 + mpi_col_rank);
        std::normal_distribution<> d;

        for (auto j = 0; j < m_ * (nev_ + nex_); j++)
        {
            auto rnd = getRandomT<T>([&]() { return d(gen); });
            C_[j] = rnd;
        }
    }
    //! This function set initially the operation for apply() in filter
    void preApplication(T* V, std::size_t locked, std::size_t block) override
    {
        next_ = NextOp::bAc;
    }

    //! - This function performs the local computation of `GEMM` for
    //! ChaseMpiDLA::apply()
    //! - It is implemented based on `BLAS`'s `xgemm`.
    void apply(T alpha, T beta, std::size_t offset, std::size_t block,
               std::size_t locked) override
    {

        T Zero = T(0.0);

        if (next_ == NextOp::bAc)
        {

            if (mpi_col_rank != 0)
            {
                beta = Zero;
            }
            t_gemm<T>(CblasColMajor, CblasConjTrans, CblasNoTrans, n_,
                      static_cast<std::size_t>(block), m_, &alpha, H_, ldh_,
                      C_ + offset * m_ + locked * m_, m_, &beta,
                      B_ + locked * n_ + offset * n_, n_);
            next_ = NextOp::cAb;
        }
        else
        {

            if (mpi_row_rank != 0)
            {
                beta = Zero;
            }
            t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_,
                   static_cast<std::size_t>(block), n_, &alpha, H_, ldh_,
                   B_ + offset * n_ + locked * n_, n_, &beta,
                   C_ + offset * m_ + locked * m_, m_);
            next_ = NextOp::bAc;
        }
    }

    //! This function performs the shift of diagonal of a global matrix
    //! - This global is already distributed, so the shifting operation takes
    //! place on the local
    //!   block of global matrix on each MPI proc.
    //! - This function is naturally in parallel among all MPI procs.
    void shiftMatrix(T c, bool isunshift = false) override
    {

        for (std::size_t j = 0; j < nblocks_; j++)
        {
            for (std::size_t i = 0; i < mblocks_; i++)
            {
                for (std::size_t q = 0; q < c_lens_[j]; q++)
                {
                    for (std::size_t p = 0; p < r_lens_[i]; p++)
                    {
                        if (q + c_offs_[j] == p + r_offs_[i])
                        {
                            H_[(q + c_offs_l_[j]) * ldh_ + p + r_offs_l_[i]] +=
                                c;
                        }
                    }
                }
            }
        }
    }
    //! - This function performs the local computation of `GEMM` for
    //! ChaseMpiDLA::asynCxHGatherC()
    //! - It is implemented based on `BLAS`'s `xgemm`.
    void asynCxHGatherC(std::size_t locked, std::size_t block,
                        bool isCcopied = false) override
    {
        T alpha = T(1.0);
        T beta = T(0.0);

        t_gemm<T>(CblasColMajor, CblasConjTrans, CblasNoTrans, n_,
                  static_cast<std::size_t>(block), m_, &alpha, H_, ldh_,
                  C_ + locked * m_, m_, &beta, B_ + locked * n_, n_);
    }

    //! - All required operations for this function has been done in for
    //! ChaseMpiDLA::applyVec().
    //! - This function contains nothing in this class.
    void applyVec(T* v, T* w) override
    {
        T alpha = T(1.0);
        T beta = T(0.0);
        std::size_t k = 1;
        // t_gemm<T>(CblasColMajor, CblasConjTrans, CblasNoTrans, n_,
        //          k, m_, &alpha, H_, ldh_,
        //          v, m_, &beta, w, n_);
        t_gemv<T>(CblasColMajor, CblasConjTrans, m_, n_, &alpha, H_, ldh_, v, 1,
                  &beta, w, 1);
    }

    int get_nprocs() const override { return matrix_properties_->get_nprocs(); }
    void Start() override {}
    void End() override {}
    Base<T>* get_Resids() override { return resid_; }
    Base<T>* get_Ritzv() override { return ritzv_; }
    //! It is an interface to BLAS `?axpy`.
    void axpy(std::size_t N, T* alpha, T* x, std::size_t incx, T* y,
              std::size_t incy) override
    {
        t_axpy(N, alpha, x, incx, y, incy);
    }

    //! It is an interface to BLAS `?scal`.
    void scal(std::size_t N, T* a, T* x, std::size_t incx) override
    {
        t_scal(N, a, x, incx);
    }

    //! It is an interface to BLAS `?nrm2`.
    Base<T> nrm2(std::size_t n, T* x, std::size_t incx) override
    {
        return t_nrm2(n, x, incx);
    }

    //! It is an interface to BLAS `?dot`.
    T dot(std::size_t n, T* x, std::size_t incx, T* y,
          std::size_t incy) override
    {
        return t_dot(n, x, incx, y, incy);
    }
    //! - This function performs the local computation of `GEMM` for
    //! ChaseMpiDLA::RR()
    //! - It is implemented based on `BLAS`'s `?gemm`.
    void RR(std::size_t block, std::size_t locked, Base<T>* ritzv) override
    {
        T One = T(1.0);
        T Zero = T(0.0);

        t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, block, block, n_,
               &One, B2_ + locked * n_, n_, B_ + locked * n_, n_, &Zero, A_,
               nev_ + nex_);
    }

    //! It is an interface to BLAS `?sy(he)rk`.
    void syherk(char uplo, char trans, std::size_t n, std::size_t k, T* alpha,
                T* a, std::size_t lda, T* beta, T* c, std::size_t ldc,
                bool first = true) override
    {
        t_syherk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
    //! It is an interface to LAPACK `?potrf`.
    int potrf(char uplo, std::size_t n, T* a, std::size_t lda, bool isinfo = true) override    
    {
        return t_potrf(uplo, n, a, lda);
    }
    //! It is an interface to BLAS `?trsm`.
    void trsm(char side, char uplo, char trans, char diag, std::size_t m,
              std::size_t n, T* alpha, T* a, std::size_t lda, T* b,
              std::size_t ldb, bool first = false) override
    {
        t_trsm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
    }
    //! - This function performs the local computation of residuals for
    //! ChaseMpiDLA::Resd()
    //! - It is implemented based on `BLAS`'s `?axpy` and `?nrm2`.
    //! - This function computes only the residuals of local part of vectors on
    //! each MPI proc.
    //! - The final results are obtained in ChaseMpiDLA::Resd() with an
    //! MPI_Allreduce operation
    //!      within the row communicator.
    void Resd(Base<T>* ritzv, Base<T>* resid, std::size_t locked,
              std::size_t unconverged) override
    {
        for (auto i = 0; i < unconverged; i++)
        {
            T alpha = -ritzv[i];
            t_axpy(n_, &alpha, B2_ + locked * n_ + i * n_, 1,
                   B_ + locked * n_ + i * n_, 1);

            resid[i] = t_norm_p2(n_, B_ + locked * n_ + i * n_);
        }
    }

    //! - This function performs the local computation for ChaseMpiDLA::heevd()
    //! - It is implemented based on `BLAS`'s `?gemm` and LAPACK's `?sy(he)evd`.
    void heevd(int matrix_layout, char jobz, char uplo, std::size_t n, T* a,
               std::size_t lda, Base<T>* w) override
    {
        T One = T(1.0);
        T Zero = T(0.0);
        std::size_t locked = nev_ + nex_ - n;

        t_heevd(matrix_layout, jobz, uplo, n, a, nev_ + nex_, w);
        t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_, n, n, &One,
               C2_ + locked * m_, m_, A_, nev_ + nex_, &Zero, C_ + locked * m_,
               m_);
    }
    //! - All required operations for this function has been done in for
    //! ChaseMpiDLA::hhQR().
    //! - This function contains nothing in this class.
    void hhQR(std::size_t locked) override {}
    //! - All required operations for this function has been done in for
    //! ChaseMpiDLA::cholQR().
    //! - This function contains nothing in this class.
    void cholQR(std::size_t locked, Base<T> cond) override {}
    //! - All required operations for this function has been done in for
    //! ChaseMpiDLA::Swap().
    //! - This function contains nothing in this class.
    void Swap(std::size_t i, std::size_t j) override {}
    //! - All required operations for this function has been done in for
    //! ChaseMpiDLA::LanczosDos().
    //! - This function contains nothing in this class.
    void LanczosDos(std::size_t idx, std::size_t m, T* ritzVc) override
    {
        T alpha = T(1.0);
        T beta = T(0.0);
        t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_, idx, m, &alpha,
               C_, m_, ritzVc, m, &beta, C2_, m_);
        std::memcpy(C_, C2_, m * m_ * sizeof(T));

    }
    void Lanczos(std::size_t M, int idx, Base<T>* d, Base<T>* e,
                 Base<T>* r_beta) override
    {
    }
    void B2C(T* B, std::size_t off1, T* C, std::size_t off2,
             std::size_t block) override
    {
    }

    void lacpy(char uplo, std::size_t m, std::size_t n, T* a, std::size_t lda,
               T* b, std::size_t ldb) override
    {
        t_lacpy(uplo, m, n, a, lda, b, ldb);
    }

    void shiftMatrixForQR(T* A, std::size_t n, T shift) override
    {
        for (auto i = 0; i < n; i++)
        {
            A[i * n + i] += (T)shift;
        }
    }

    ChaseMpiMatrices<T>* getChaseMatrices() override { return &matrices_; }

private:
    enum NextOp
    {
        cAb,
        bAc
    };

    NextOp next_; //!< it is to manage the switch of operation from `V2=H*V1` to
                  //!< `V1=H'*V2` in filter
    std::size_t N_; //!< global dimension of the symmetric/Hermtian matrix

    std::size_t n_; //!< number of columns of local matrix of the
                    //!< symmetric/Hermtian matrix
    std::size_t
        m_; //!< number of rows of local matrix of the symmetric/Hermtian matrix
    std::size_t ldh_; //!< leading dimension of local matrix on each MPI proc
    T* H_;            //!< a pointer to the local matrix on each MPI proc
    T* B_;  //!< a matrix of size `n_*(nev_+nex_)`, which is allocated in
            //!< ChaseMpiMatrices
    T* B2_; //!< a matrix of size `n_*(nev_+nex_)`, which is allocated in
            //!< ChaseMpiProperties
    T* C_;  //!< a matrix of size `m_*(nev_+nex_)`, which is allocated in
            //!< ChaseMpiMatrices
    T* C2_; //!< a matrix of size `m_*(nev_+nex_)`, which is allocated in
            //!< ChaseMpiProperties
    T* A_;  //!< a matrix of size `(nev_+nex_)*(nev_+nex_)`, which is allocated
            //!< in ChaseMpiProperties
    T* w_;  //!< a vector of size `N_`, which is allocated in this
            //!< class for Lanczos

    T* vv_;

    Base<T>* resid_;

    Base<T>* ritzv_;

    std::size_t* off_;      //!< identical to ChaseMpiProperties::off_
    std::size_t* r_offs_;   //!< identical to ChaseMpiProperties::r_offs_
    std::size_t* r_lens_;   //!< identical to ChaseMpiProperties::r_lens_
    std::size_t* r_offs_l_; //!< identical to ChaseMpiProperties::r_offs_l_
    std::size_t* c_offs_;   //!< identical to ChaseMpiProperties::c_offs_
    std::size_t* c_lens_;   //!< identical to ChaseMpiProperties::c_lens_
    std::size_t* c_offs_l_; //!< identical to ChaseMpiProperties::c_offs_l_
    std::size_t nb_;        //!< identical to ChaseMpiProperties::nb_
    std::size_t mb_;        //!< identical to ChaseMpiProperties::mb_
    std::size_t nblocks_;   //!< identical to ChaseMpiProperties::nblocks_
    std::size_t mblocks_;   //!< identical to ChaseMpiProperties::mblocks_
    std::size_t nev_;       //!< number of required eigenpairs
    std::size_t nex_;       //!< number of extral searching space
    int mpi_row_rank;       //!< rank within each row communicator
    int mpi_col_rank;       //!< rank within each column communicator

    ChaseMpiProperties<T>*
        matrix_properties_; //!< an object of class ChaseMpiProperties
    ChaseMpiMatrices<T> matrices_;
};

template <typename T>
struct is_skewed_matrixfree<ChaseMpiDLABlaslapack<T>>
{
    static const bool value = true;
};

} // namespace mpi
} // namespace chase
