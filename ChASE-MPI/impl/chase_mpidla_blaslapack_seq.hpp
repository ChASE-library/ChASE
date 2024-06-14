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
        v0_ = (T*)malloc(N_ * sizeof(T));
        v2_ = (T*)malloc(N_ * sizeof(T));
    }

    ChaseMpiDLABlaslapackSeq() = delete;
    //! Destructor
    ChaseMpiDLABlaslapackSeq(ChaseMpiDLABlaslapackSeq const& rhs) = delete;

    ~ChaseMpiDLABlaslapackSeq()
    {
        free(v0_);
        free(v2_);
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

        std::memcpy(v1.ptr(), C_, N_ * block_size * sizeof(T));

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
            this->applyVec(v1.ptr(), B2_, nb);
            this->B2C(B2_, 0, w.ptr(), 0, nb);

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
            //std::swap(v1_, v0_);
            //std::swap(v1_, w_);                
        }

        t_heevd(LAPACK_COL_MAJOR, 'N', 'U', M, submatrix.ptr(), M, ritzv_.data());
        
        *r_beta = *std::max_element(ritzv_.begin(), ritzv_.end()) + std::abs((beta.ptr())[0]);
#else
        std::cout << "calling Lanczos..." << std::endl;     
        std::vector<Base<T>> ritzv_(M);
        Base<T> real_beta;
        Base<T>* d = new Base<T>[M]();
        Base<T>* e = new Base<T>[M]();

        T alpha = T(1.0);
        T beta = T(0.0);

        std::fill(v0_, v0_ + N_, T(0));

#ifdef USE_NSIGHT
        nvtxRangePushA("Lanczos Init vec");
#endif

        std::mt19937 gen(2342.0);
        std::normal_distribution<> normal_distribution;
        v1_ = C2_;
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
            this->applyVec(v1_, B2_, 1);
            this->B2C(B2_, 0, v2_, 0, 1);
            // B2C
            alpha = this->dot(N_, v1_, 1, v2_, 1);
            alpha = -alpha;
            this->axpy(N_, &alpha, v1_, 1, v2_, 1);
            alpha = -alpha;

            d[k] = std::real(alpha);

            beta = T(-real_beta);
            this->axpy(N_, &beta, v0_, 1, v2_, 1);
            beta = -beta;

            real_beta = this->nrm2(N_, v2_, 1);

            beta = T(1.0 / real_beta);

            if (k == M - 1)
                break;

            this->scal(N_, &beta, v2_, 1);

            e[k] = real_beta;

            std::swap(v1_, v0_);
            std::swap(v1_, v2_);
        }
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
        int notneeded_m;
        std::size_t vl, vu;
        Base<T> ul, ll;
        int tryrac = 0;
        std::vector<int> isuppz(2 * M);

        t_stemr<Base<T>>(LAPACK_COL_MAJOR, 'N', 'A', M, d, e, ul, ll, vl, vu,
                         &notneeded_m, ritzv_.data(), NULL, M, M, isuppz.data(), &tryrac);

        *r_beta = std::max(std::abs(ritzv_[0]), std::abs(ritzv_[M - 1])) +
                  std::abs(real_beta);

        delete[] d;
        delete[] e;
#endif        
    }

    void mLanczos(std::size_t M, int numvec, Base<T>* d, Base<T>* e,
                 Base<T>* r_beta) override
    {
        Base<T>* real_alpha = new Base<T>[numvec]();
        Base<T>* real_beta = new Base<T>[numvec]();
        std::vector<T> alpha(numvec, T(1.0));
        std::vector<T> beta(numvec, T(0.0));

        T *v1 = C2_;
        T *v0 = (T*)malloc(N_ * numvec * sizeof(T)) ;
        T *v2 = (T*)malloc(N_ * numvec * sizeof(T)) ;
        
        std::fill(v0, v0 + N_ * numvec, T(0));

        for(auto i = 0; i < numvec; i++)
        {
            real_alpha[i] = this->nrm2(N_,v1 + i * N_, 1);
            alpha[i] = T(1 / real_alpha[i]);
        }

        for(auto i = 0; i < numvec; i++)
        {
            this->scal(N_, &alpha[i], v1 + i * N_, 1);
        }

        for (std::size_t k = 0; k < M; k = k + 1)
        {
            for(auto i = 0; i < numvec; i++)
            {
                std::memcpy(C_ + k * N_, v1 + i * N_, N_ * sizeof(T));
            }
            
            this->applyVec(v1, B2_, numvec);
            
            this->B2C(B2_, 0, v2, 0, numvec);

            for(auto i = 0; i < numvec; i++)
            {
                alpha[i] = this->dot(N_, v1 + i * N_, 1, v2 + i * N_, 1);
                alpha[i] = -alpha[i];
            }
            for(auto i = 0; i < numvec; i++)
            {
                this->axpy(N_, &alpha[i], v1 + i * N_, 1, v2 + i * N_, 1);
                alpha[i] = -alpha[i];
            }

            for(auto i = 0; i < numvec; i++)
            {
                d[k + M * i] = std::real(alpha[i]);
            }

            for(auto i = 0; i < numvec; i++)
            {
                beta[i] = T(-real_beta[i]);
            }

            for(auto i = 0; i < numvec; i++)
            {
                this->axpy(N_, &beta[i], v0 + i * N_, 1, v2 + i * N_, 1);
                beta[i] = -beta[i];
            }

            for(auto i = 0; i < numvec; i++)
            {
                real_beta[i] = this->nrm2(N_, v2 + i * N_, 1);
                beta[i] = T(1.0 / real_beta[i]);
            }

            if (k == M - 1)
                break;

            for(auto i = 0; i < numvec; i++)
            {
                this->scal(N_, &beta[i], v2 + i * N_, 1);
            }

            for(auto i = 0; i < numvec; i++)
            {
                e[k + M * i] = real_beta[i];
            }

            std::swap(v1, v0);
            std::swap(v1, v2);
        }
                
        for(auto i = 0; i < numvec; i++)
        {
            r_beta[i] = real_beta[i];
        }
        delete[] real_beta;
        delete[] real_alpha;    
        delete[] v0;
        delete[] v2;
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
    {}  
    void scal_batch(std::size_t N, T* a, Matrix<T>* x, std::size_t incx, int count) override
    {}   
    void applyVec(Matrix<T>* v, Matrix<T>* w, std::size_t n) override
    {} 
    void dot_batch(std::size_t n, Matrix<T>* x, std::size_t incx, Matrix<T>* y,
          std::size_t incy, T *products, int count) override
    {} 
    void axpy_batch(std::size_t N, T* alpha, Matrix<T>* x, std::size_t incx, Matrix<T>* y,
              std::size_t incy, int count) override
    {}  
    void gemmStrideBatch(char transa, char transb, 
              std::size_t m, std::size_t n, std::size_t k,
              T* alpha, Matrix<T>* A, std::size_t strideA, 
              Matrix<T>* B, std::size_t strideB,
              T* beta, Matrix<T>* C, std::size_t strideC, int batchCount) override
    {} 
    void syherkStrideBatch(char uplo, char trans, std::size_t n, std::size_t k, T* alpha,
                Matrix<T>* a, std::size_t strideA, T* beta, Matrix<T>* c, std::size_t strideC, int batchCount,
                bool first = true) override
    {}
    int *potrfStrideBatch(char uplo, std::size_t n, Matrix<T>* a, std::size_t strideA, int batchCount, bool isinfo = true) override {int *info; return info;}   


    void trsmStrideBatch(char side, char uplo, char trans, char diag, std::size_t m,
              std::size_t n, T* alpha, Matrix<T>* a, std::size_t strideA, Matrix<T>* b,
              std::size_t strideB, int batchCount, bool first = false) override {}

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
    T* v0_; //!< a vector of size `N_`, which is allocated in this
            //!< class for Lanczos
    T* v1_;
    T* v2_;
    std::unique_ptr<Matrix<T>> C2vec;
    std::unique_ptr<Matrix<T>> B2vec; 
    T* C_; //!< a pointer to a matrix of size `N_*(nev_+nex_)`
    T* B_; //!< a pointer to a matrix of size `N_*(nev_+nex_)`
    T* C2_;
    T* B2_;
    T* A_;
    ChaseMpiMatrices<T> matrices_;

};

template <typename T>
struct is_skewed_matrixfree<ChaseMpiDLABlaslapackSeq<T>>
{
    static const bool value = false;
};

} // namespace mpi
} // namespace chase
