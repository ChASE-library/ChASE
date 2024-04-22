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

        v0_ = (T*)malloc(N_ * sizeof(T));
        v1_ = (T*)malloc(N_ * sizeof(T));
        w_ = (T*)malloc(N_ * sizeof(T));
    }

    ~ChaseMpiDLABlaslapackSeqInplace()
    {
        free(v0_);
        free(v1_);
        free(w_);
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
    void Lanczos(std::size_t M,
                 Base<T>* r_beta) override
    {
#ifdef USE_BLANCZOS  
        std::cout << "calling Block Lanczos..." << std::endl;

        std::size_t block_size = 4;
        std::size_t div = std::floor(static_cast<double>(M) / block_size);

        M = (div * block_size + 1 == M) ? M : block_size * (div + 1) + 1;
        std::vector<Base<T>> ritzv_(M);

        T One = T(1.0);
        T Zero = T(0.0);
        T NegOne = T(-1.0);

        Matrix<T> alpha(0, block_size, block_size);
        Matrix<T> beta(0, block_size, block_size);
        Matrix<T> submatrix(0, M, M);
        Matrix<T> v0(0, N_, block_size);
        Matrix<T> v1(0, N_, block_size);
        Matrix<T> w(0, N_, block_size);
        Matrix<T> A(0, block_size, 2 * block_size);

        std::memcpy(v1.ptr(), V1_, N_ * block_size * sizeof(T));

        //CholQR
        int info = -1;
        t_syherk('U', 'C', block_size, N_, &One, v1.ptr(), N_, &Zero, A.ptr(), block_size);
        info = t_potrf('U', block_size, A.ptr(), block_size);
        if(info == 0){
            t_trsm('R', 'U', 'N', 'N', N_, block_size, &One, A.ptr(), block_size, v1.ptr(), N_);
        }else
        {
            t_geqrf(LAPACK_COL_MAJOR, N_, block_size, v1.ptr(), N_, A.ptr());
            t_gqr(LAPACK_COL_MAJOR, N_, block_size, block_size, v1.ptr(), N_, A.ptr());
        }

        for(auto k = 0; k < M; k = k + block_size)
        {
            auto nb = std::min(block_size, M - k);
            this->applyVec(v1.ptr(), w.ptr(), nb);
            //alpha = v1^T * w
            t_gemm(CblasColMajor, CblasConjTrans,  CblasNoTrans, 
                    nb, nb, N_,
                    &One, v1.ptr(), N_, w.ptr(), N_, 
                    &Zero, alpha.ptr(), block_size);

            //w = - v * alpha + w
            t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                    N_, nb, nb, 
                    &NegOne, v1.ptr(), N_, alpha.ptr(), block_size,
                    &One, w.ptr(), N_);

            //save alpha onto the block diag
            t_lacpy('A', nb, nb, alpha.ptr(), block_size, submatrix.ptr() + k + k * M, M);

            //w = - v0 * beta + w
            if(k > 0){
                t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                        N_, nb, nb, 
                        &NegOne, v0.ptr(), N_, beta.ptr(), block_size,
                        &One, w.ptr(), N_);
            }

            //CholeskyQR2
            // A = V^T * V
            t_syherk('U', 'C', nb, N_, &One, w.ptr(), N_, &Zero, A.ptr(), block_size);
            // A = Chol(A)
            info = t_potrf('U', nb, A.ptr(), block_size);
            if(info == 0){
                // w = W * A^(-1)
                t_trsm('R', 'U', 'N', 'N', N_, nb, &One, A.ptr(), block_size, w.ptr(), N_);
                // A' = V^T * V
                t_syherk('U', 'C', nb, N_, &One, w.ptr(), N_, &Zero, A.ptr() + block_size * block_size, block_size);
                // A' = Chol(A)
                info = t_potrf('U', nb, A.ptr() + block_size * block_size, block_size);
                // w = W * A'^(-1)
                t_trsm('R', 'U', 'N', 'N', N_, nb, &One, A.ptr() + block_size * block_size, block_size, w.ptr(), N_);
                // beta = A'*A
                t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                    nb, nb, nb, 
                    &One, A.ptr() + block_size * block_size, block_size, A.ptr(), block_size,
                    &Zero, beta.ptr(), block_size);
            }else
            {
                t_geqrf(LAPACK_COL_MAJOR, N_, block_size, w.ptr(), N_, A.ptr());
                t_lacpy('U', nb, nb, w.ptr(), N_, beta.ptr(), block_size);
                t_gqr(LAPACK_COL_MAJOR, N_, block_size, block_size, w.ptr(), N_, A.ptr());
            }
            
            if (k == M - 1)
                break;

            //save beta to the off-diagonal
            if(k > 0)
            {
                t_lacpy('U', nb, nb, beta.ptr(), block_size, submatrix.ptr() + k * M + (k - 1), M);
            }

            v1.swap(v0);
            v1.swap(w);                
        }

        t_heevd(LAPACK_COL_MAJOR, 'N', 'U', M, submatrix.ptr(), M, ritzv_.data());
        
        *r_beta = *std::max_element(ritzv_.begin(), ritzv_.end()) + std::abs((beta.ptr())[0]);
#else
        std::cout << "calling Lanczos..." << std::endl;
        std::vector<Base<T>> ritzv_(M);

        Base<T> real_beta;
        std::vector<Base<T>> d(M);
        std::vector<Base<T>> e(M);

        T alpha = T(1.0);
        T beta = T(0.0);

        std::fill(v0_, v0_ + N_, T(0));

#ifdef USE_NSIGHT
        nvtxRangePushA("Lanczos Init vec");
#endif

        std::mt19937 gen(2342.0);
        std::normal_distribution<> normal_distribution;

        for (std::size_t k = 0; k < N_; ++k)
        {
            v1_[k] =
                getRandomT<T>([&]() { return normal_distribution(gen); });
        }
        
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
        // ENSURE that v1 has one norm
#ifdef USE_NSIGHT
        nvtxRangePushA("Lanczos: loop");
#endif
        Base<T> real_alpha = this->nrm2(N_, v1_, 1);
        alpha = T(1 / real_alpha);
        this->scal(N_, &alpha, v1_, 1);
        for (std::size_t k = 0; k < M; k = k + 1)
        {
            this->applyVec(v1_, w_, 1);
            alpha = this->dot(N_, v1_, 1, w_, 1);
            alpha = -alpha;
            this->axpy(N_, &alpha, v1_, 1, w_, 1);
            alpha = -alpha;

            d[k] = std::real(alpha);

            beta = T(-real_beta);
            this->axpy(N_, &beta, v0_, 1, w_, 1);
            beta = -beta;

            real_beta = this->nrm2(N_, w_, 1);

            beta = T(1.0 / real_beta);

            if (k == M - 1)
                break;
                
            this->scal(N_, &beta, w_, 1);

            e[k] = real_beta;

            std::swap(v1_, v0_);
            std::swap(v1_, w_);
        }
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
        int notneeded_m;
        std::size_t vl, vu;
        Base<T> ul, ll;
        int tryrac = 0;
        std::vector<int> isuppz(2 * M);

        t_stemr<Base<T>>(LAPACK_COL_MAJOR, 'N', 'A', M, d.data(), e.data(), ul, ll, vl, vu,
                         &notneeded_m, ritzv_.data(), NULL, M, M, isuppz.data(), &tryrac);

        *r_beta = std::max(std::abs(ritzv_[0]), std::abs(ritzv_[M - 1])) +
                  std::abs(real_beta);
#endif    
    }

    void mLanczos(std::size_t M, int numvec, Base<T>* d, Base<T>* e,
                 Base<T>* r_beta) override
    {
        Base<T>* real_alpha = new Base<T>[numvec]();
        Base<T>* real_beta = new Base<T>[numvec]();
        std::vector<T> alpha(numvec, T(1.0));
        std::vector<T> beta(numvec, T(0.0));

        std::vector<T> v0(N_ * numvec);
        std::vector<T> v1(N_ * numvec);
        std::vector<T> w(N_ * numvec);

        std::fill(v0.begin(), v0.end(), T(0));

        for(auto i = 0; i < numvec; i++)
        {
            std::memcpy(v1.data() + i * N_, V2_ + i * N_, N_ * sizeof(T));
        }

        for(auto i = 0; i < numvec; i++)
        {
            real_alpha[i] = this->nrm2(N_,V2_ + i * N_, 1);
            alpha[i] = T(1 / real_alpha[i]);
        }

        for(auto i = 0; i < numvec; i++)
        {
            this->scal(N_, &alpha[i], v1.data() + i * N_, 1);
        }

        for (std::size_t k = 0; k < M; k = k + 1)
        {
            for(auto i = 0; i < numvec; i++)
            {
                std::memcpy(V1_ + k * N_, v1.data() + i * N_, N_ * sizeof(T));
            }
            
            this->applyVec(v1.data(), w.data(), numvec);

            for(auto i = 0; i < numvec; i++)
            {
                alpha[i] = this->dot(N_, v1.data() + i * N_, 1, w.data() + i * N_, 1);
                alpha[i] = -alpha[i];
            }
            for(auto i = 0; i < numvec; i++)
            {
                this->axpy(N_, &alpha[i], v1.data() + i * N_, 1, w.data() + i * N_, 1);
                alpha[i] = -alpha[i];
            }
            
            for(auto i = 0; i < numvec; i++)
            {
                d[k + M * i] = std::real(alpha[i]);
            }            

            if (k == M - 1)
                break;

            for(auto i = 0; i < numvec; i++)
            {
                beta[i] = T(-real_beta[i]);
            }

            for(auto i = 0; i < numvec; i++)
            {
                this->axpy(N_, &beta[i], v0.data() + i * N_, 1, w.data() + i * N_, 1);
                beta[i] = -beta[i];
            }

            for(auto i = 0; i < numvec; i++)
            {
                real_beta[i] = this->nrm2(N_, w.data() + i * N_, 1);
                beta[i] = T(1.0 / real_beta[i]);
            }

            for(auto i = 0; i < numvec; i++)
            {
                this->scal(N_, &beta[i], w.data() + i * N_, 1);
            }

            for(auto i = 0; i < numvec; i++)
            {
                e[k + M * i] = real_beta[i];
            }

            v1.swap(v0);
            v1.swap(w);
        }    

        for(auto i = 0; i < numvec; i++)
        {
            r_beta[i] = real_beta[i];
        }

        delete[] real_beta;
        delete[] real_alpha;

    }

    void B2C(T* B, std::size_t off1, T* C, std::size_t off2,
             std::size_t block) override
    {
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
    T* v0_; //!< a vector of size `N_`, which is allocated in this
            //!< class for Lanczos
    T* v1_; //!< a vector of size `N_`, which is allocated in this
            //!< class for Lanczos
    T* w_;
    ChaseMpiMatrices<T> matrices_;
};

template <typename T>
struct is_skewed_matrixfree<ChaseMpiDLABlaslapackSeqInplace<T>>
{
    static const bool value = false;
};

} // namespace mpi
} // namespace chase
