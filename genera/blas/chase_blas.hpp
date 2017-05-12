/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
#pragma once

#include <cstring> //memcpy
#include <iostream>
#include <random>

#include "ChASE.hpp"
#include "template_wrapper.hpp"

// TODO:
// -- random vectors for lanczos?

template <class T>
class ChASE_Blas : public ChASE<T> {
public:
    ChASE_Blas(ChASE_Config _config, T* H_, T* V_, Base<T>* ritzv_)
        : N(_config.getN())
        , nev(_config.getNev())
        , nex(_config.getNex())
        , locked(0)
        , config(_config)
        , dealloc(false)
        , H(H_)
        , V(V_)
        , ritzv(ritzv_)
    {
        W = new T[N * (nev + nex)]();
        approxV = V;
        workspace = W;
    };

    ChASE_Blas(ChASE_Config _config)
        : N(_config.getN())
        , nev(_config.getNev())
        , nex(_config.getNex())
        , locked(0)
        , config(_config)
        , dealloc(true)
    {
        H = new T[N * N]();
        V = new T[N * (nev + nex)]();
        W = new T[N * (nev + nex)]();
        ritzv = new Base<T>[ (nev + nex) ];
        approxV = V;
        workspace = W;
    };

    ChASE_Blas(const ChASE_Blas&) = delete;

    ~ChASE_Blas()
    {
        if (dealloc) {
            delete[] H;
            delete[] V;
            delete[] ritzv;
        }
        delete[] W;
    };

    ChASE_PerfData getPerfData() { return perf; }

    ChASE_Config getConfig() { return config; }

    std::size_t getN() { return N; }

    CHASE_INT getNev() { return nev; }

    CHASE_INT getNex() { return nex; }

    Base<T>* getRitzv() { return ritzv; }

    // TODO: everything should be owned by chASE_Blas, deg should be part of
    // ChasE_Config
    void solve()
    {
        perf = ChASE_Algorithm<T>::solve(this, N, ritzv, nev, nex);
    }

    T* getMatrixPtr() { return H; }

    T* getVectorsPtr() { return approxV; }

    T* getWorkspacePtr() { return workspace; }

    void shift(T c, bool isunshift = false)
    {
        for (CHASE_INT i = 0; i < N; ++i)
            H[i * N + i] += c;
    };

    // todo this is wrong we want the END of V
    void cpy(CHASE_INT new_converged)
    {
        //    memcpy( workspace+locked*N, approxV+locked*N,
        //    N*(new_converged)*sizeof(T) );
        memcpy(approxV + locked * N, workspace + locked * N,
            N * (new_converged) * sizeof(T));
    };

    void threeTerms(CHASE_INT nev, T alpha, T beta, CHASE_INT offset)
    {
        t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, nev, N, &alpha, H, N,
            approxV + (locked + offset) * N, N, &beta,
            workspace + (locked + offset) * N, N);
        std::swap(approxV, workspace);
    };

    void Hv(T alpha);

    void QR(CHASE_INT fixednev)
    {
        CHASE_INT nevex = nev + nex;
        T* tau = workspace + fixednev * N;

        memcpy(workspace, approxV, N * fixednev * sizeof(T));
        t_geqrf(LAPACK_COL_MAJOR, N, nevex, approxV, N, tau);
        t_gqr(LAPACK_COL_MAJOR, N, nevex, nevex, approxV, N, tau);

        memcpy(approxV, workspace, N * fixednev * sizeof(T));
    };

    void RR(Base<T>* ritzv, CHASE_INT block)
    {
        // CHASE_INT block = nev+nex - fixednev;

        T* A = new T[block * block]; // For LAPACK.

        T One = T(1.0, 0.0);
        T Zero = T(0.0, 0.0);

        // V <- H*V
        t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, block, N, &One, H, N,
            approxV + locked * N, N, &Zero, workspace + locked * N, N);

        // A <- W * V
        t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, block, block, N, &One,
            approxV + locked * N, N, workspace + locked * N, N, &Zero, A, block);

        t_heevd(LAPACK_COL_MAJOR, 'V', 'L', block, A, block, ritzv);

        t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, block, block, &One,
            approxV + locked * N, N, A, block, &Zero, workspace + locked * N, N);

        std::swap(approxV, workspace);
        // we can swap, since the locked part were copied over as part of the QR

        delete[] A;
    };

    void resd(Base<T>* ritzv, Base<T>* resid, CHASE_INT fixednev)
    {
        T alpha = T(1.0, 0.0);
        T beta = T(0.0, 0.0);
        CHASE_INT unconverged = (nev + nex) - fixednev;

        Base<T> norm = this->getNorm();

        t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, unconverged, N, &alpha,
            H, N, approxV + locked * N, N, &beta, workspace + locked * N, N);

        Base<T> norm1, norm2;
        for (std::size_t i = 0; i < unconverged; ++i) {
            beta = T(-ritzv[i], 0.0);
            t_axpy(N, &beta, (approxV + locked * N) + N * i, 1,
                (workspace + locked * N) + N * i, 1);

            norm1 = t_nrm2(N, (workspace + locked * N) + N * i, 1);
            resid[i] = norm1 / norm;
        }
    };

    void swap(CHASE_INT i, CHASE_INT j)
    {
        T* ztmp = new T[N];
        memcpy(ztmp, approxV + N * i, N * sizeof(T));
        memcpy(approxV + N * i, approxV + N * j, N * sizeof(T));
        memcpy(approxV + N * j, ztmp, N * sizeof(T));
        memcpy(ztmp, workspace + N * i, N * sizeof(T));
        memcpy(workspace + N * i, workspace + N * j, N * sizeof(T));
        memcpy(workspace + N * j, ztmp, N * sizeof(T));
        delete[] ztmp;
    };

    Base<T> getNorm() { return norm; };

    void setNorm(Base<T> norm_) { norm = norm_; };

    void lanczos(CHASE_INT m, Base<T>* upperb)
    {
        // todo
        CHASE_INT n = N;

        T* v1 = workspace;
        for (std::size_t k = 0; k < N; ++k)
            v1[k] = V[k];

        // assert( m >= 1 );
        Base<T>* d = new Base<T>[ m ]();
        Base<T>* e = new Base<T>[ m ]();

        // SO C++03 5.3.4[expr.new]/15
        T* v0_ = new T[n]();
        T* w_ = new T[n]();

        T* v0 = v0_;
        T* w = w_;

        T alpha = T(1.0, 0.0);
        T beta = T(0.0, 0.0);
        T One = T(1.0, 0.0);
        T Zero = T(0.0, 0.0);

        //  T *v1 = V;
        // ENSURE that v1 has one norm
        Base<T> real_alpha = t_nrm2(n, v1, 1);
        alpha = T(1 / real_alpha, 0.0);
        t_scal(n, &alpha, v1, 1);
        Base<T> real_beta = 0;

        real_beta = 0;

        for (std::size_t k = 0; k < m; ++k) {
            t_gemv(CblasColMajor, CblasNoTrans, n, n, &One, H, n, v1, 1, &Zero, w, 1);

            t_dot(n, v1, 1, w, 1, &alpha);

            alpha = -alpha;
            t_axpy(n, &alpha, v1, 1, w, 1);
            alpha = -alpha;

            d[k] = alpha.real();
            if (k == m - 1)
                break;

            beta = T(-real_beta, 0);
            t_axpy(n, &beta, v0, 1, w, 1);
            beta = -beta;

            real_beta = t_nrm2(n, w, 1);
            beta = T(1.0 / real_beta, 0.0);

            t_scal(n, &beta, w, 1);

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
        Base<T>* ritzv = new Base<T>[ m ];

        t_stemr<Base<T> >(LAPACK_COL_MAJOR, 'N', 'A', m, d, e, ul, ll, vl, vu,
            &notneeded_m, ritzv, NULL, m, m, isuppz, &tryrac);

        *upperb = std::max(std::abs(ritzv[0]), std::abs(ritzv[m - 1])) + std::abs(real_beta); // TODO

        delete[] ritzv;
        delete[] isuppz;
        delete[] d;
        delete[] e;
    };

    // we need to be careful how we deal with memory here
    // we will operate within Workspace
    void lanczos(CHASE_INT M, CHASE_INT idx, Base<T>* upperb, Base<T>* ritzv,
        Base<T>* Tau, Base<T>* ritzV)
    {
        // todo
        CHASE_INT m = M;
        CHASE_INT n = N;

        // assert( m >= 1 );

        // The first m*N part is reserved for the lanczos vectors
        Base<T>* d = new Base<T>[ m ]();
        Base<T>* e = new Base<T>[ m ]();

        // SO C++03 5.3.4[expr.new]/15
        T* v0_ = new T[n]();
        T* w_ = new T[n]();

        T* v0 = v0_;
        T* w = w_;

        T alpha = T(1.0, 0.0);
        T beta = T(0.0, 0.0);
        T One = T(1.0, 0.0);
        T Zero = T(0.0, 0.0);

        // V is filled with randomness
        T* v1 = workspace;
        for (std::size_t k = 0; k < N; ++k)
            v1[k] = V[k + idx * N];

        // ENSURE that v1 has one norm
        Base<T> real_alpha = t_nrm2(n, v1, 1);
        alpha = T(1 / real_alpha, 0.0);
        t_scal(n, &alpha, v1, 1);
        Base<T> real_beta = 0;

        real_beta = static_cast<Base<T> >(0);
        for (std::size_t k = 0; k < m; ++k) {
            if (workspace + k * n != v1)
                memcpy(workspace + k * n, v1, n * sizeof(T));

            t_gemv(CblasColMajor, CblasNoTrans, n, n, &One, H, n, v1, 1, &Zero, w, 1);

            // std::cout << "lanczos Av\n";
            // for (std::size_t ll = 0; ll < 2; ++ll)
            //   std::cout << w[ll] << "\n";

            t_dot(n, v1, 1, w, 1, &alpha);

            alpha = -alpha;
            t_axpy(n, &alpha, v1, 1, w, 1);
            alpha = -alpha;

            d[k] = alpha.real();
            if (k == m - 1)
                break;

            beta = T(-real_beta, 0);
            t_axpy(n, &beta, v0, 1, w, 1);
            beta = -beta;

            real_beta = t_nrm2(n, w, 1);
            beta = T(1.0 / real_beta, 0.0);

            t_scal(n, &beta, w, 1);

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

        *upperb = std::max(std::abs(ritzv[0]), std::abs(ritzv[m - 1])) + std::abs(real_beta);

        std::cout << "upperb: " << *upperb << "\n";

        for (std::size_t k = 1; k < m; ++k) {
            Tau[k] = std::abs(ritzV[k * m]) * std::abs(ritzV[k * m]);
            // std::cout << Tau[k] << "\n";
        }

        delete[] isuppz;
        delete[] d;
        delete[] e;
    };

    void lock(CHASE_INT new_converged)
    {
        memcpy(workspace + locked * N, approxV + locked * N,
            N * (new_converged) * sizeof(T));
        locked += new_converged;
    };

    double compare(T* V_)
    {
        double norm = 0;
        for (CHASE_INT i = 0; i < (nev + nex) * N; ++i)
            norm += std::abs(V_[i] - approxV[i]) * std::abs(V_[i] - approxV[i]);
        std::cout << "norm: " << norm << "\n";

        norm = 0;
        for (CHASE_INT i = 0; i < (locked)*N; ++i)
            norm += std::abs(V_[i] - approxV[i]) * std::abs(V_[i] - approxV[i]);
        std::cout << "norm: " << norm << "\n";
    }

    void lanczosDoS(CHASE_INT idx, CHASE_INT m, T* ritzVc)
    {
        T alpha = T(1, 0);
        T beta = T(0, 0);
        t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, idx, m, &alpha,
            workspace, N, ritzVc, m, &beta, approxV, N);

        /*
        // TODO this may not be necessary, check memory footprint of
        //      lanczos on approxV
        {
            std::mt19937 gen(2342.0);
            std::normal_distribution<> d;
            for (std::size_t k = 0; k < N * (nev + nex - idx); ++k) {
                approxV[N * idx + k] = T(d(gen), d(gen));
            }
        }
        */
    }

    Base<T> residual()
    {
        for (CHASE_INT j = 0; j < N * (nev + nex); ++j) {
            W[j] = V[j];
        }

        //    memcpy(W, V, sizeof(MKL_Complex16)*N*nev);
        T one(1.0);
        T zero(0.0);
        T eigval;
        int iOne = 1;
        for (int ttz = 0; ttz < nev; ttz++) {
            eigval = -1.0 * ritzv[ttz];
            t_scal(N, &eigval, W + ttz * N, 1);
        }
        t_hemm(CblasColMajor, CblasLeft, CblasLower, N, nev, &one, H, N, V, N, &one,
            W, N);
        Base<T> norm = t_lange('M', N, nev, W, N);
        // TR.registerValue( i, "resd", norm);
        return norm;
    }

    Base<T> orthogonality()
    {
        T one(1.0);
        T zero(0.0);
        // Check eigenvector orthogonality
        T* unity = new T[nev * nev];
        T neg_one(-1.0);
        for (int ttz = 0; ttz < nev; ttz++) {
            for (int tty = 0; tty < nev; tty++) {
                if (ttz == tty)
                    unity[nev * ttz + tty] = 1.0;
                else
                    unity[nev * ttz + tty] = 0.0;
            }
        }

        t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nev, nev, N, &one, V, N,
            V, N, &neg_one, unity, nev);
        Base<T> norm = t_lange('M', nev, nev, unity, nev);
        delete[] unity;
        return norm;
    }

    void output(std::string str)
    {
        std::cout << str;
    }

private:
    std::size_t N, nev, nex, locked;
    T *H, *V, *W;
    T *approxV, *workspace;
    Base<T> norm;
    Base<T>* ritzv;

    ChASE_Config config;
    ChASE_PerfData perf;

    const bool dealloc;
};

// TODO
/*
void check_params(std::size_t N, std::size_t nev, std::size_t nex,
                  const double tol, std::size_t deg )
{
  bool abort_flag = false;
  if(tol < 1e-14)
    std::clog << "WARNING: Tolerance too small, may take a while." << std::endl;
  if(deg < 8 || deg > ChASE_Config::chase_max_deg)
    std::clog << "WARNING: Degree should be between 8 and " <<
ChASE_Config::chase_max_deg << "."
              << " (current: " << deg << ")" << std::endl;
  if((double)nex/nev < 0.15 || (double)nex/nev > 0.75)
    {
      std::clog << "WARNING: NEX should be between 0.15*NEV and 0.75*NEV."
                << " (current: " << (double)nex/nev << ")" << std::endl;
      //abort_flag = true;
    }
  if(nev+nex > N)
    {
      std::cerr << "ERROR: NEV+NEX has to be smaller than N." << std::endl;
      abort_flag = true;
    }

  if(abort_flag)
    {
      std::cerr << "Stopping execution." << std::endl;
      exit(-1);
    }
 }
*/
