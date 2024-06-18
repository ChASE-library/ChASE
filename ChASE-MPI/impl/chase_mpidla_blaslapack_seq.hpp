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

//! @brief A derived class of ChaseMpiDLAInterface which implements ChASE
//! targeting shared-memory architectures with only CPUs available.
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
    explicit ChaseMpiDLABlaslapackSeq(T* H, std::size_t ldh, T* V1,
                                      Base<T>* ritzv, std::size_t n,
                                      std::size_t nex, std::size_t nev)
        : N_(n), nev_(nev), nex_(nex), maxBlock_(nex + nex), ldh_(ldh),
         matrices_(0, N_, nev_ + nex_, H, ldh, V1, ritzv)
    {
        C2vec = std::make_unique<Matrix<T>>(0, N_, maxBlock_);
        B2vec = std::make_unique<Matrix<T>>(0, N_, maxBlock_);        
        H_ =  matrices_.H().ptr();
        C_ =  matrices_.C().ptr();
        B_ = matrices_.B().ptr();
        C2_ = C2vec.get()->ptr();
        B2_ = B2vec.get()->ptr();
        A_ = matrices_.A().ptr();
    }

    ChaseMpiDLABlaslapackSeq() = delete;
    //! Destructor
    ChaseMpiDLABlaslapackSeq(ChaseMpiDLABlaslapackSeq const& rhs) = delete;

    ~ChaseMpiDLABlaslapackSeq()
    {
    }
    void initVecs() override
    {
        next_ = NextOp::bAc;
        t_lacpy('A', N_, nev_ + nex_, C_, N_, C2_, N_);
    }
    void initRndVecs() override
    {
        std::mt19937 gen(1337.0);
        std::normal_distribution<> d;
        for (auto j = 0; j < (nev_ + nex_); j++)
        {
            for (auto i = 0; i < N_; i++)
            {
                C_[i + j * N_] = getRandomT<T>([&]() { return d(gen); });
            }
        }
    }

    void preApplication(T* V, std::size_t const locked,
                        std::size_t const block) override
    {
        next_ = NextOp::bAc;
        locked_ = locked;
        std::memcpy(C_, V + locked * N_, N_ * block * sizeof(T));
    }

    void apply(T alpha, T beta, std::size_t offset, std::size_t block,
               std::size_t locked) override
    {

        if (next_ == NextOp::bAc)
        {
            t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, N_,
                   static_cast<std::size_t>(block), N_, &alpha, H_, ldh_,
                   C_ + offset * N_ + locked * N_, N_, &beta,
                   B_ + locked * N_ + offset * N_, N_);

            next_ = NextOp::cAb;
        }
        else
        {
            t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N_,
                   static_cast<std::size_t>(block), N_, &alpha, H_, ldh_,
                   B_ + offset * N_ + locked * N_, N_, &beta,
                   C_ + offset * N_ + locked * N_, N_);

            next_ = NextOp::bAc;
        }
    }

    void shiftMatrix(T const c, bool isunshift = false) override
    {
        for (std::size_t i = 0; i < N_; ++i)
        {
            H_[i + i * ldh_] += c;
        }
    }

    void asynCxHGatherC(std::size_t locked, std::size_t block,
                        bool isCcopied = false) override
    {
        T const alpha = T(1.0);
        T const beta = T(0.0);

        t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, N_, block, N_,
               &alpha, H_, ldh_, C_ + locked * N_, N_, &beta, B_ + locked * N_,
               N_);

        std::memcpy(B2_ + locked * N_, C2_ + locked * N_,
                    N_ * block * sizeof(T));
    }

    void applyVec(T* v, T* w, std::size_t n) override
    {
        T One = T(1.0);
        T Zero = T(0.0);

        t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, //
               N_, n, N_,                                 //
               &One,                                      //
               H_, ldh_,                                  //
               v, N_,                                     //
               &Zero,                                     //
               w, N_);
    }

    bool checkSymmetryEasy() override 
    {
        std::vector<T> v(N_);
        std::vector<T> u(N_);
        std::vector<T> uT(N_);

        std::mt19937 gen(1337.0);
        std::normal_distribution<> d;
        for (auto i = 0; i < N_; i++)
        {
            v[i] = getRandomT<T>([&]() { return d(gen); });
        }
        
        T One = T(1.0);
        T Zero = T(0.0);

        t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, //
               N_, 1, N_,                                 //
               &One,                                      //
               H_, ldh_,                                  //
               v.data(), N_,                             //
               &Zero,                                     //
               u.data(), N_);

        t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, //
               N_, 1, N_,                                 //
               &One,                                      //
               H_, ldh_,                                  //
               v.data(), N_,                             //
               &Zero,                                     //
               uT.data(), N_);

        bool is_sym = true;
        for(auto i = 0; i < N_; i++)
        {
            if(!(u[i] == uT[i]))
            {
                is_sym = false;
                return is_sym;
            }
        }

        return is_sym;
    }

    void symOrHermMatrix(char uplo) override 
    {
        if(uplo == 'U')
        {
            for(auto j = 0; j < N_; j++)
            {
                for(auto i = 0; i < j; i++)
                {
                    H_[j + i * ldh_]= conjugate(H_[i + j * ldh_]);
                }
            }
        }else
        {
            for(auto i = 0; i < N_; i++)
            {
                for(auto j = 0; j < i; j++)
                {
                    H_[j + i * ldh_]= conjugate(H_[i + j * ldh_]);
                }
            }
        }
    }

    int get_nprocs() const override { return 1; }

    void Start() override {}
    void End() override {}
    Base<T>* get_Resids() override { return matrices_.Resid().ptr(); }
    Base<T>* get_Ritzv() override { return matrices_.Ritzv().ptr(); }

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
               &One, B2_ + locked * N_, N_, B_ + locked * N_, N_, &Zero,
               A.get(), block);

        t_heevd(LAPACK_COL_MAJOR, 'V', 'L', block, A.get(), block, ritzv);

        t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N_, block, block,
               &One, C2_ + locked * N_, N_, A.get(), block, &Zero,
               C_ + locked * N_, N_);

        std::memcpy(C2_ + locked * N_, C_ + locked * N_,
                    N_ * block * sizeof(T));
    }

    void syherk(char uplo, char trans, std::size_t n, std::size_t k, T* alpha,
                T* a, std::size_t lda, T* beta, T* c, std::size_t ldci,
                bool first = true) override
    {
    }

    int potrf(char uplo, std::size_t n, T* a, std::size_t lda, bool isinfo = true) override
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
            t_axpy(N_, &beta, (B2_ + locked * N_) + N_ * i, 1,
                   (B_ + locked * N_) + N_ * i, 1);

            resid[i] = nrm2(N_, (B_ + locked * N_) + N_ * i, 1);
        }
    }

    void hhQR(std::size_t locked) override
    {
        auto nevex = nev_ + nex_;

        std::unique_ptr<T[]> tau(new T[nevex]);

        t_geqrf(LAPACK_COL_MAJOR, N_, nevex, C_, N_, tau.get());
        t_gqr(LAPACK_COL_MAJOR, N_, nevex, nevex, C_, N_, tau.get());
    }

    int cholQR1(std::size_t locked) override
    {
        auto nevex = nev_ + nex_;
        T one = T(1.0);
        T zero = T(0.0);
        int info = 1;

        t_syherk('U', 'C', nevex, N_, &one, C_, N_, &zero, A_, nevex);
        info = t_potrf('U', nevex, A_, nevex); 

        if(info != 0)
        {
            return info;
        }else
        {
            t_trsm('R', 'U', 'N', 'N', N_, nevex, &one, A_, nevex, C_, N_);
#ifdef CHASE_OUTPUT
            std::cout << "choldegree: 1" << std::endl;
#endif                    
            return info;  
        }
    }

    int cholQR2(std::size_t locked) override
    {
        auto nevex = nev_ + nex_;
        T one = T(1.0);
        T zero = T(0.0);
        int info = 1;

        t_syherk('U', 'C', nevex, N_, &one, C_, N_, &zero, A_, nevex);

        info = t_potrf('U', nevex, A_, nevex); 

        if(info != 0)
        {
            return info;
        }else
        {
            t_trsm('R', 'U', 'N', 'N', N_, nevex, &one, A_, nevex, C_, N_);
            t_syherk('U', 'C', nevex, N_, &one, C_, N_, &zero, A_, nevex);
            info = t_potrf('U', nevex, A_, nevex);
            t_trsm('R', 'U', 'N', 'N', N_, nevex, &one, A_, nevex, C_, N_); 

#ifdef CHASE_OUTPUT
            std::cout << std::setprecision(2) << "choldegree: 2" << std::endl;
#endif                    
            return info;  
        }    
    }

    int shiftedcholQR2(std::size_t locked) override
    {
        Base<T> shift;
        auto nevex = nev_ + nex_; 
        T one = T(1.0);
        T zero = T(0.0);
        int info = 1;
        t_syherk('U', 'C', nevex, N_, &one, C_, N_, &zero, A_, nevex);
        
        Base<T> nrmf = 0.0;
        this->computeDiagonalAbsSum(A_, &nrmf, nevex, nevex);

        shift = std::sqrt(N_) * nrmf * std::numeric_limits<Base<T>>::epsilon();
        
        this->shiftMatrixForQR(A_, nevex, (T)shift);
        info = t_potrf('U', nevex, A_, nevex); 

        if(info != 0)
	    {
	        return info;
	    }

        t_trsm('R', 'U', 'N', 'N', N_, nevex, &one, A_, nevex, C_, N_);
        t_syherk('U', 'C', nevex, N_, &one, C_, N_, &zero, A_, nevex);
        info = t_potrf('U', nevex, A_, nevex);
        t_trsm('R', 'U', 'N', 'N', N_, nevex, &one, A_, nevex, C_, N_); 
        t_syherk('U', 'C', nevex, N_, &one, C_, N_, &zero, A_, nevex);
        info = t_potrf('U', nevex, A_, nevex);
        t_trsm('R', 'U', 'N', 'N', N_, nevex, &one, A_, nevex, C_, N_); 

#ifdef CHASE_OUTPUT
        std::cout << "choldegree: 2, shift = " << shift << std::endl;
#endif 
        return info;
    }

    void estimated_cond_evaluator(std::size_t locked, Base<T> cond)
    {
        auto nevex = nev_ + nex_;
        std::vector<Base<T>> S(nevex - locked);
        std::vector<Base<T>> norms(nevex - locked);
        std::vector<T> V2(N_ * (nevex));
        V2.assign(C_, C_ + N_ * nevex);
        T* U;
        std::size_t ld = 1;
        T* Vt;
        t_gesvd('N', 'N', N_, nevex - locked, V2.data() + N_ * locked,
                N_, S.data(), U, ld, Vt, ld);
        
        for (auto i = 0; i < nevex - locked; i++)
        {
            norms[i] = std::sqrt(t_sqrt_norm(S[i]));
        }
        
        std::sort(norms.begin(), norms.end());

        std::cout << "estimate: " << cond << ", rcond: "
                    << norms[nev_ + nex_ - locked - 1] / norms[0]
                    << ", ratio: "
                    << cond * norms[0] / norms[nev_ + nex_ - locked - 1]
                    << std::endl;        
    }
    
    void lockVectorCopyAndOrthoConcatswap(std::size_t locked, bool isHHqr)
    {
        std::memcpy(C_, C2_, locked * N_ * sizeof(T));
        std::memcpy(C2_ + locked * N_, C_ + locked * N_,
                    (nev_ + nex_ - locked) * N_ * sizeof(T));        
    } 

    void Swap(std::size_t i, std::size_t j) override
    {
        T* tmp = new T[N_];

        memcpy(tmp, C_ + N_ * i, N_ * sizeof(T));
        memcpy(C_ + N_ * i, C_ + N_ * j, N_ * sizeof(T));
        memcpy(C_ + N_ * j, tmp, N_ * sizeof(T));

        memcpy(tmp, C2_ + N_ * i, N_ * sizeof(T));
        memcpy(C2_ + N_ * i, C2_ + N_ * j, N_ * sizeof(T));
        memcpy(C2_ + N_ * j, tmp, N_ * sizeof(T));

        delete[] tmp;
    }

    void LanczosDos(std::size_t idx, std::size_t m, T* ritzVc) override
    {
        T alpha = T(1.0);
        T beta = T(0.0);

        t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N_, idx, m, &alpha,
               C_, N_, ritzVc, m, &beta, C2_, N_);
        std::memcpy(C_, C2_, m * N_ * sizeof(T));
    }
    void Lanczos(std::size_t M,
                 Base<T>* r_beta) override
    {
    }

    void mLanczos(std::size_t M, int numvec, Base<T>* d, Base<T>* e,
                 Base<T>* r_beta) override
    {
        bool is_second_system = false;

        if(numvec == -1)
        {
            numvec = 1;
            is_second_system = true;
        }

        std::vector<Base<T>> real_alpha(numvec);
        std::vector<T> alpha(numvec, T(1.0));
        std::vector<T> beta(numvec, T(0.0));

        v_0 = new Matrix<T>(0, N_, numvec);
        v_1 = new Matrix<T>(0, N_, numvec);
        v_2 = new Matrix<T>(0, N_, numvec);

        std::memcpy(v_1->ptr(), C_, N_ * numvec * sizeof(T));
        this->nrm2_batch(N_, v_1, 1, numvec, real_alpha.data());
        
        for(auto i = 0; i < numvec; i++)
        {
            alpha[i] = T(1 / real_alpha[i]);
        }

        this->scal_batch(N_, alpha.data(), v_1, 1, numvec);

        for (std::size_t k = 0; k < M; k = k + 1)
        {
            if(!is_second_system)
            {
                for(auto i = 0; i < numvec; i++){
                    std::memcpy(C_ + k * N_, v_1->ptr() + i * N_, N_ * sizeof(T));
                }
            }

            this->applyVec(v_1, v_2, numvec);

            this->dot_batch(N_, v_1, 1, v_2, 1, alpha.data(), numvec);
            for(auto i = 0; i < numvec; i++)
            {
                alpha[i] = -alpha[i];
            }

            this->axpy_batch(N_, alpha.data(), v_1, 1, v_2, 1, numvec);
            for(auto i = 0; i < numvec; i++)
            {
                alpha[i] = -alpha[i];
            }

            for(auto i = 0; i < numvec; i++)
            {
                d[k + M * i] = std::real(alpha[i]);
            }

            if(k > 0){
                for(auto i = 0; i < numvec; i++)
                {
                    beta[i] = T(-r_beta[i]);
                }
                this->axpy_batch(N_, beta.data(), v_0, 1, v_2, 1, numvec);
            }

            for(auto i = 0; i < numvec; i++)
            {
                beta[i] = -beta[i];
            }

            this->nrm2_batch(N_, v_2, 1, numvec, r_beta);


            for(auto i = 0; i < numvec; i++)
            {
                beta[i] = T(1 / r_beta[i]);
            }

            if (k == M - 1)
                break;
            
            this->scal_batch(N_, beta.data(), v_2, 1, numvec);

            for(auto i = 0; i < numvec; i++)
            {
                e[k + M * i] = r_beta[i];
            }
            v_1->swap(*v_0);
            v_1->swap(*v_2);
        }

        if(!is_second_system)
        {
            std::memcpy(C_, v_1->ptr(), N_ * numvec * sizeof(T));  
        }
    }

    void mLanczos(std::size_t M, int numvec, Base<T>* upperb,
                 Base<T>* ritzv, Base<T>* Tau, Base<T>* ritzV) override
    {}

    void B2C(T* B, std::size_t off1, T* C, std::size_t off2,
             std::size_t block) override
    {
        std::memcpy(C + off2 * N_, B + off1 * N_, block * N_ * sizeof(T));
    }

    void lacpy(char uplo, std::size_t m, std::size_t n, T* a, std::size_t lda,
               T* b, std::size_t ldb) override
    {
    }

    void shiftMatrixForQR(T* A, std::size_t n, T shift) override 
    {
        for (auto i = 0; i < n; i++)
        {
            A[i * n + i] += (T)shift;
        }
    }

    void computeDiagonalAbsSum(T *A, Base<T> *sum, std::size_t n, std::size_t ld)
    {
        *sum = 0.0;
        
        for(auto i = 0; i < n; i++)
        {
            *sum += std::abs(A[i * ld + i]);
        }        
    }

    ChaseMpiMatrices<T>* getChaseMatrices() override { return &matrices_; }

    void nrm2_batch(std::size_t n, Matrix<T>* x, std::size_t incx, int count, Base<T> *nrms) override
    {
        for(auto i = 0; i < count; i++ )
        {
            nrms[i] = t_nrm2(n, x->ptr() + i *  n, incx);
        }
    }
    
    void scal_batch(std::size_t N, T* a, Matrix<T>* x, std::size_t incx, int count) override
    {
        //t_scal(N, a, x, incx);
        for(auto i = 0; i < count; i++)
        {
            t_scal(N, &a[i], x->ptr() + i * x->ld(), incx);
        }
    }

    void applyVec(Matrix<T>* v, Matrix<T>* w, std::size_t n) override
    {
        T alpha = T(1.0);
        T beta = T(0.0);
        std::size_t k = n;
        t_gemm<T>(CblasColMajor, CblasConjTrans, CblasNoTrans, N_,
                  k, N_, &alpha, H_, ldh_,
                  v->ptr(), v->ld(), &beta, w->ptr(), w->ld());

    }

    void dot_batch(std::size_t n, Matrix<T>* x, std::size_t incx, Matrix<T>* y,
          std::size_t incy, T *products, int count) override
    {
        //return t_dot(n, x, incx, y, incy);
        for(auto i = 0; i < count; i++)
        {
            products[i] = t_dot(n, x->ptr() + i * x->ld(), incx, y->ptr() + i * y->ld(), incy);
        }
    }

    void axpy_batch(std::size_t N, T* alpha, Matrix<T>* x, std::size_t incx, Matrix<T>* y,
              std::size_t incy, int count) override
    {
        //t_axpy(N, alpha, x, incx, y, incy);
        for(auto i = 0; i < count; i++)
        {
            t_axpy(N, &alpha[i], x->ptr() + i * x->ld(), incx, y->ptr() + i * y->ld(), incy);
        }
    }    


private:
    enum NextOp
    {
        cAb,
        bAc
    };
    NextOp next_; //!< it is to manage the switch of operation from `V2=H*V1` to
                  //!< `V1=H'*V2` in filter

    std::size_t N_;      //!< global dimension of the symmetric/Hermtian matrix
    std::size_t locked_; //!< number of converged eigenpairs
    std::size_t maxBlock_;  //!< `maxBlock_=nev_ + nex_`
    std::size_t nex_;       //!< number of extral searching space
    std::size_t nev_;       //!< number of required eigenpairs
    T* H_;                  //!< a pointer to the Symmetric/Hermtian matrix
    std::size_t ldh_;       //!< leading dimension of Hermitian matrix
    std::unique_ptr<T> V1_; //!< a matrix of size `N_*(nev_+nex_)`
    std::unique_ptr<T> V2_; //!< a matrix of size `N_*(nev_+nex_)`
    std::unique_ptr<Matrix<T>> C2vec;
    std::unique_ptr<Matrix<T>> B2vec; 
    T* C_; //!< a pointer to a matrix of size `N_*(nev_+nex_)`
    T* B_; //!< a pointer to a matrix of size `N_*(nev_+nex_)`
    T* C2_;
    T* B2_;
    T* A_;
    Matrix<T> *v_0, *v_1, *v_2;
    ChaseMpiMatrices<T> matrices_;

};

template <typename T>
struct is_skewed_matrixfree<ChaseMpiDLABlaslapackSeq<T>>
{
    static const bool value = false;
};

} // namespace mpi
} // namespace chase
