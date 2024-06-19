/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2023, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <cstring>
#include <memory>

#include "ChASE-MPI/chase_mpidla_interface.hpp"

namespace chase
{
namespace mpi
{

//! @brief A derived class of ChaseMpiDLAInterface which implements ChASE
//! targeting shared-memory architectures with only CPUs available.
/*! It implements in a inplace mode, in which the buffer of `V1` and `V2` are
 * swapped and reused, which reduces the required memory to be allocted.
 */
template <class T>
class ChaseMpiDLABlaslapackSeqInplace : public ChaseMpiDLAInterface<T>
{
public:
    //! A constructor of ChaseMpiDLABlaslapackSeqInplace.
    /*! @param matrices: it is an object of ChaseMpiMatrices, which allocates
       the required buffer.
        @param n: size of matrix defining the eigenproblem.
        @param maxBlock: maximum column number of matrix `V`, which equals to
       `nev+nex`.
    */
    ChaseMpiDLABlaslapackSeqInplace(T* H, std::size_t ldh, T* V1,
                                    Base<T>* ritzv, std::size_t n,
                                    std::size_t nev, std::size_t nex)
        : N_(n), nex_(nex), nev_(nev), maxblock_(nev_ + nex_),
          matrices_(0, N_, nev_ + nex_, H, ldh, V1, ritzv)
    {

        V1_ = matrices_.C().ptr();
        V2_ = matrices_.B().ptr();
        H_ = matrices_.H().ptr();
        A_ = matrices_.A().ptr();

        ldh_ = matrices_.get_ldh();
    }

    ~ChaseMpiDLABlaslapackSeqInplace()
    {
    }
    void initVecs() override
    {
        t_lacpy('A', N_, nev_ + nex_, V1_, N_, V2_, N_);
    }
    void initRndVecs() override
    {
        std::mt19937 gen(1337.0);
        std::normal_distribution<> d;
        for (auto j = 0; j < (nev_ + nex_); j++)
        {
            for (auto i = 0; i < N_; i++)
            {
                V1_[i + j * N_] = getRandomT<T>([&]() { return d(gen); });
            }
        }
    }

    void preApplication(T* V, std::size_t locked, std::size_t block) override
    {
        locked_ = locked;
        std::memcpy(V1_, V + locked * N_, N_ * block * sizeof(T));
    }

    void apply(T alpha, T beta, std::size_t offset, std::size_t block,
               std::size_t locked) override
    {
        t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N_,
               static_cast<std::size_t>(block), N_, &alpha, H_, ldh_,
               V1_ + offset * N_ + locked * N_, N_, &beta,
               V2_ + locked * N_ + offset * N_, N_);

        std::swap(V1_, V2_);
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
    }

    void applyVec(T* B, T* C, std::size_t n) override
    {
        T One = T(1.0);
        T Zero = T(0.0);

        t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, //
               N_, n, N_,                                 //
               &One,                                      //
               H_, ldh_,                                  //
               B, N_,                                     //
               &Zero,                                     //
               C, N_);
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

        t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, N_, block, N_, &One,
               H_, ldh_, V1_ + locked * N_, N_, &Zero, V2_ + locked * N_, N_);

        // A <- W' * V
        t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, block, block, N_,
               &One, V2_ + locked * N_, N_, V1_ + locked * N_, N_, &Zero, A_,
               nev_ + nex_);

        t_heevd(LAPACK_COL_MAJOR, 'V', 'L', block, A_, nev_ + nex_, ritzv);

        t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N_, block, block,
               &One, V1_ + locked * N_, N_, A_, nev_ + nex_, &Zero,
               V2_ + locked * N_, N_);

        std::swap(V1_, V2_);
    }

    void syherk(char uplo, char trans, std::size_t n, std::size_t k, T* alpha,
                T* a, std::size_t lda, T* beta, T* c, std::size_t ldc,
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

        t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, N_, unconverged, N_,
               &alpha, H_, ldh_, V1_ + locked * N_, N_, &beta,
               V2_ + locked * N_, N_);

        for (std::size_t i = 0; i < unconverged; ++i)
        {
            beta = T(-ritzv[i]);
            t_axpy(N_, &beta, (V1_ + locked * N_) + N_ * i, 1,
                   (V2_ + locked * N_) + N_ * i, 1);

            resid[i] = nrm2(N_, (V2_ + locked * N_) + N_ * i, 1);
        }
    }

    void hhQR(std::size_t locked) override
    {
        auto nevex = nev_ + nex_;

        std::unique_ptr<T[]> tau(new T[nevex]);

        std::memcpy(V2_, V1_, locked * N_ * sizeof(T));

        t_geqrf(LAPACK_COL_MAJOR, N_, nevex, V1_, N_, tau.get());
        t_gqr(LAPACK_COL_MAJOR, N_, nevex, nevex, V1_, N_, tau.get());
    }

    int cholQR1(std::size_t locked) override
    {
        auto nevex = nev_ + nex_;
        T one = T(1.0);
        T zero = T(0.0);
        int info = 1;

        std::memcpy(V2_, V1_, locked * N_ * sizeof(T));

        t_syherk('U', 'C', nevex, N_, &one, V1_, N_, &zero, A_, nevex);
        info = t_potrf('U', nevex, A_, nevex); 

        if(info != 0)
        {
            return info;
        }else
        {
            t_trsm('R', 'U', 'N', 'N', N_, nevex, &one, A_, nevex, V1_, N_);
#ifdef CHASE_OUTPUT
            std::cout << std::setprecision(2) << "choldegree: 1" << std::endl;
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

        std::memcpy(V2_, V1_, locked * N_ * sizeof(T));

        t_syherk('U', 'C', nevex, N_, &one, V1_, N_, &zero, A_, nevex);
        info = t_potrf('U', nevex, A_, nevex); 

        if(info != 0)
        {
            return info;
        }else
        {
            t_trsm('R', 'U', 'N', 'N', N_, nevex, &one, A_, nevex, V1_, N_);
            t_syherk('U', 'C', nevex, N_, &one, V1_, N_, &zero, A_, nevex);
            info = t_potrf('U', nevex, A_, nevex);
            t_trsm('R', 'U', 'N', 'N', N_, nevex, &one, A_, nevex, V1_, N_); 

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

        std::memcpy(V2_, V1_, locked * N_ * sizeof(T));

        t_syherk('U', 'C', nevex, N_, &one, V1_, N_, &zero, A_, nevex);
        Base<T> nrmf = 0.0;
    
        this->computeDiagonalAbsSum(A_, &nrmf, nevex, nevex);
                
        shift = std::sqrt(N_) * nrmf * std::numeric_limits<Base<T>>::epsilon();
        
        this->shiftMatrixForQR(A_, nevex, (T)shift);
        info = t_potrf('U', nevex, A_, nevex); 

        if(info != 0)
	    {
	        return info;
	    }

        t_trsm('R', 'U', 'N', 'N', N_, nevex, &one, A_, nevex, V1_, N_);
        t_syherk('U', 'C', nevex, N_, &one, V1_, N_, &zero, A_, nevex);
        info = t_potrf('U', nevex, A_, nevex);
        if(info != 0)
	{
	    return info;
	}
        t_trsm('R', 'U', 'N', 'N', N_, nevex, &one, A_, nevex, V1_, N_); 
        t_syherk('U', 'C', nevex, N_, &one, V1_, N_, &zero, A_, nevex);
        info = t_potrf('U', nevex, A_, nevex);
        t_trsm('R', 'U', 'N', 'N', N_, nevex, &one, A_, nevex, V1_, N_); 

#ifdef CHASE_OUTPUT
        std::cout << std::setprecision(2) << "choldegree: 2, shift = " << shift << std::endl;
#endif 
        return info;
    }

    void estimated_cond_evaluator(std::size_t locked, Base<T> cond)
    {
        auto nevex = nev_ + nex_;
        std::vector<Base<T>> S(nevex - locked);
        std::vector<Base<T>> norms(nevex - locked);
        std::vector<T> V2(N_ * (nevex));
        V2.assign(V1_, V1_ + N_ * nevex);
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
        std::memcpy(V1_, V2_, locked * N_ * sizeof(T));
    }     

    void Swap(std::size_t i, std::size_t j) override
    {
        T* tmp = new T[N_];

        memcpy(tmp, V1_ + N_ * i, N_ * sizeof(T));
        memcpy(V1_ + N_ * i, V1_ + N_ * j, N_ * sizeof(T));
        memcpy(V1_ + N_ * j, tmp, N_ * sizeof(T));
    }

    void LanczosDos(std::size_t idx, std::size_t m, T* ritzVc) override
    {
        T alpha = T(1.0);
        T beta = T(0.0);

        t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N_, idx, m, &alpha,
               V1_, N_, ritzVc, m, &beta, V2_, N_);
        std::memcpy(V1_, V2_, m * N_ * sizeof(T));
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

        std::memcpy(v_1->ptr(), V1_, N_ * numvec * sizeof(T));
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
                    std::memcpy(V1_ + k * N_, v_1->ptr() + i * N_, N_ * sizeof(T));
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
            std::memcpy(V1_, v_1->ptr(), N_ * numvec * sizeof(T));  
        }

    }

    void B2C(T* B, std::size_t off1, T* C, std::size_t off2,
             std::size_t block) override
    {
    }

    void B2C(Matrix<T>* B, std::size_t off1, Matrix<T>* C, std::size_t off2,
                        std::size_t block) override
    {}

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
    std::size_t N_;      //!< global dimension of the symmetric/Hermtian matrix
    std::size_t locked_; //!< number of converged eigenpairs
    std::size_t nev_;    //!< number of required eigenpairs
    std::size_t nex_;    //!< number of extral searching space
    std::size_t maxblock_; //!< `maxBlock_=nev_ + nex_`
    std::size_t ldh_;      //!< leading dimension of Hermitian matrix
    T* H_;                 //!< a pointer to the Symmetric/Hermtian matrix
    T* V1_;                //!< a matrix of size `N_*(nev_+nex_)`
    T* A_;
    T* V2_; //!< a matrix of size `N_*(nev_+nex_)`
    Matrix<T> *v_0, *v_1, *v_2;
    ChaseMpiMatrices<T> matrices_;
};

template <typename T>
struct is_skewed_matrixfree<ChaseMpiDLABlaslapackSeqInplace<T>>
{
    static const bool value = false;
};

} // namespace mpi
} // namespace chase
