/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set ts=8 sts=2 et sw=2 tw=80: */
/* TODO License */
#pragma once

#include "ChASE.hpp"

#include "BMMMPI.hpp"
#include "template_wrapper.hpp"
#include <cstring> //mempcpy
#include <iostream>
#include <mpi.h>
#include <random>

// TODO:
// -- random vectors for lanczos?

template <class T>
class ChASE_MPI : public ChASE<T> {
public:
    ChASE_MPI(ChASE_Config _config, MPI_Comm aComm, T* H_, T* V_, Base<T>* ritzv_)
        : N(_config.getN())
        , nev(_config.getNev())
        , nex(_config.getNex())
        , locked(0)
        , config(_config)
        , dealloc(false)
        , H(H_)
        , V(V_)
        , ritzv(ritzv_)
        , mComm(aComm)
    {
        W = new T[N * (nev + nex)]();
        approxV = V;
        workspace = W;

        mpi_handle = new MPI_Handler<T>;
        MPI_handler_init(mpi_handle, aComm, N, nev + nex);
        MPI_distribute_H(mpi_handle, H_);
    };

    ChASE_MPI(ChASE_Config _config, MPI_Comm aComm, T* V_, Base<T>* ritzv_)
        : N(_config.getN())
        , nev(_config.getNev())
        , nex(_config.getNex())
        , locked(0)
        , config(_config)
        , dealloc(false)
        , V(V_)
        , ritzv(ritzv_)
        , mComm(aComm)
    {
        W = new T[N * (nev + nex)]();
        approxV = V;
        workspace = W;

        mpi_handle = new MPI_Handler<T>;
        MPI_handler_init(mpi_handle, aComm, N, nev + nex);

        CHASE_INT xoff;
        CHASE_INT yoff;
        CHASE_INT xlen;
        CHASE_INT ylen;
        get_off(&xoff, &yoff, &xlen, &ylen);

        // H = new T[static_cast<std::size_t>(xlen)
        //     * static_cast<std::size_t>(ylen)]();
        H = mpi_handle->A;
    };

    ChASE_MPI(ChASE_Config _config, MPI_Comm aComm)
        : N(_config.getN())
        , nev(_config.getNev())
        , nex(_config.getNex())
        , locked(0)
        , config(_config)
        , dealloc(true)
        , mComm(aComm)
    {
        //H = new T[N * N]();
        V = new T[N * (nev + nex)]();
        W = new T[N * (nev + nex)]();
        ritzv = new Base<T>[ (nev + nex) ];
        approxV = V;
        workspace = W;

        mpi_handle = new MPI_Handler<T>;
        MPI_handler_init(mpi_handle, aComm, N, nev + nex);
    };

    ChASE_MPI(const ChASE_MPI&) = delete;

    ~ChASE_MPI()
    {
        if (dealloc) {
            //delete[] H;
            delete[] V;
            delete[] ritzv;
        }
        delete[] W;
        MPI_destroy(mpi_handle);
        delete mpi_handle;
    };

    ChASE_PerfData getPerfData() { return perf; }

    void get_off(CHASE_INT* xoff, CHASE_INT* yoff,
        CHASE_INT* xlen, CHASE_INT* ylen)
    {
        MPI_get_off(mpi_handle, xoff, yoff, xlen, ylen);
    }

    ChASE_Config getConfig() { return config; }

    std::size_t getN() { return N; }

    CHASE_INT getNev() { return nev; }

    CHASE_INT getNex() { return nex; }

    Base<T>* getRitzv() { return ritzv; }

    void distribute_H(T* aH)
    {
        MPI_distribute_H(mpi_handle, aH);
    }

    // TODO: everything should be owned by chASE_MPI, deg should be part of
    // ChasE_Config
    void solve()
    {
        MPI_Barrier(mComm);
        perf = ChASE_Algorithm<T>::solve(this, N, ritzv, nev, nex);
    }

    T* getMatrixPtr() { return H; }

    T* getVectorsPtr() { return approxV; }

    T* getWorkspacePtr() { return workspace; }

    void shift(T c, bool isunshift = false)
    {
        shiftA(mpi_handle, c);
        // for (CHASE_INT i = 0; i < N; ++i)
        //     H[i * N + i] += c;

        if (isunshift) {

            // we now do this in QR

            // TODO: assemble full vector approxV from distributed MPI
            /*
            std::vector<T> vec(mpi_handle->global_n * (nev + nex - locked));
            {
                assemble_C(mpi_handle, nev + nex - locked, vec.data());
                Base<T> norm = 0;
                int idx = (nev + nex - locked);
                for (auto kk = 0; kk < N * idx; ++kk)
                    norm += std::abs(vec.data()[kk] - approxV[locked * N + kk])
                        * std::abs(vec.data()[kk] - approxV[locked * N + kk]);
                std::cout << "norm mpi filter: " << norm << "\n";
            }
            *
            /*
            {
                mpi_handle->next = 'b';
                assemble_C(mpi_handle, nev + nex - locked, vec.data());

                Base<T> norm = 0;
                for (auto kk = 0; kk < N * (nev + nex - locked); ++kk)
                    norm += std::abs(vec.data()[kk] - approxV[locked * N + kk])
                        * std::abs(vec.data()[kk] - approxV[locked * N + kk]);
                std::cout << "norm mpi filter: " << norm << "\n";
            }
            {
                mpi_handle->next = 'c';
                assemble_C(mpi_handle, nev + nex - locked, vec.data());

                Base<T> norm = 0;
                for (auto kk = 0; kk < N * (nev + nex - locked); ++kk)
                    norm += std::abs(vec.data()[kk] - approxV[locked * N + kk])
                        * std::abs(vec.data()[kk] - approxV[locked * N + kk]);
                std::cout << "norm mpi filter: " << norm << "\n";
            }
            */
        } else {
            MPI_distribute_V(mpi_handle, approxV + locked * N, nev + nex - locked);
        }
    };

    void test()
    {
        std::vector<T> vec(N * (nev + nex));
        int nevex = 1;
        MPI_distribute_V(mpi_handle, approxV + locked * N, nevex);
        threeTerms(nevex, T(1, 0), T(0, 0), 0);
        //threeTerms( nevex, T(1,0), T(0,0), 0 );
        assemble_C(mpi_handle, nevex, vec.data());
        Base<T> norm = 0;
        for (auto kk = 0; kk < N * (nevex); ++kk)
            norm += std::abs(vec.data()[kk] - approxV[locked * N + kk])
                * std::abs(vec.data()[kk] - approxV[locked * N + kk]);
    }

    // todo this is wrong we want the END of V
    void cpy(CHASE_INT new_converged){
        // memcpy(approxV + locked * N, workspace + locked * N,
        //     N * (new_converged) * sizeof(T));
        // cpy_vectors(mpi_handle, new_converged, locked);
    };

    void threeTerms(CHASE_INT nev, T alpha, T beta, CHASE_INT offset)
    {
        //std::cout << "full local gemm disabled\n";
        // t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, nev, N, &alpha, H, N,
        //     approxV + (locked + offset) * N, N, &beta,
        //     workspace + (locked + offset) * N, N);
        // std::swap(approxV, workspace);

        MPI_doGemm(mpi_handle, alpha, beta, offset, nev);
    };

    void Hv(T alpha);

    void QR(CHASE_INT fixednev)
    {

        assemble_C(mpi_handle, nev + nex - locked, approxV + locked * N);

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
        // t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, block, N, &One, H, N,
        //     approxV + locked * N, N, &Zero, workspace + locked * N, N);

        MPI_distribute_V(mpi_handle, approxV + locked * N, block);
        MPI_doGemm(mpi_handle, T(1.0), T(0.0), 0, block);
        assemble_C(mpi_handle, block, workspace + locked * N);

        // A <- W * V
        t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, block, block, N, &One,
            approxV + locked * N, N, workspace + locked * N, N, &Zero, A, block);

        t_heevd(LAPACK_COL_MAJOR, 'V', 'L', block, A, block, ritzv);

        t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, block, block, &One,
            approxV + locked * N, N, A, block, &Zero, workspace + locked * N, N);

        std::swap(approxV, workspace);

        delete[] A;
    };

    void resd(Base<T>* ritzv, Base<T>* resid, CHASE_INT fixednev)
    {
        T alpha = T(1.0, 0.0);
        T beta = T(0.0, 0.0);
        CHASE_INT unconverged = (nev + nex) - fixednev;

        Base<T> norm = this->getNorm();

        // t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, unconverged, N, &alpha,
        //     H, N, approxV + locked * N, N, &beta, workspace + locked * N, N);

        MPI_distribute_V(mpi_handle, approxV + locked * N, unconverged);
        MPI_doGemm(mpi_handle, T(1.0), T(0.0), 0, unconverged);
        assemble_C(mpi_handle, unconverged, workspace + locked * N);

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
        std::size_t n = N;

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
            //t_gemv(CblasColMajor, CblasNoTrans, n, n, &One, H, n, v1, 1, &Zero, w, 1);
            MPI_distribute_V(mpi_handle, v1, 1);
            MPI_doGemm(mpi_handle, T(1.0), T(0.0), 0, 1);
            assemble_C(mpi_handle, 1, w);

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

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        //std::cout << "[" << rank << "] upperb: " << *upperb << "\n";

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

            //t_gemv(CblasColMajor, CblasNoTrans, n, n, &One, H, n, v1, 1, &Zero, w, 1);
            MPI_distribute_V(mpi_handle, v1, 1);
            MPI_doGemm(mpi_handle, T(1.0), T(0.0), 0, 1);
            assemble_C(mpi_handle, 1, w);

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

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        //std::cout << "[" << rank << "] upperb: " << *upperb << "\n";

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

    void lanczosDoS(CHASE_INT idx, CHASE_INT m, T* ritzVc)
    {
        T alpha = T(1, 0);
        T beta = T(0, 0);
        t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, idx, m, &alpha,
            workspace, N, ritzVc, m, &beta, approxV, N);

        // TODO this may not be necessary, check memory footprint of
        //      lanczos on approxV
        /*
        {
            std::mt19937 gen(2342.0);
            std::normal_distribution<> d;
            for (std::size_t k = 0; k < N * (m - idx); ++k) {
                approxV[k + idx * N] = T(d(gen), d(gen));
            }
        }
        */
    }

    Base<T> residual()
    {
        for (CHASE_INT j = 0; j < N * (nev + nex); ++j) {
            workspace[j] = approxV[j];
        }

        T one(1.0);
        T zero(0.0);
        T eigval;
        int iOne = 1;
        for (int ttz = 0; ttz < nev; ttz++) {
            eigval = -1.0 * ritzv[ttz];
            t_scal(N, &eigval, workspace + ttz * N, 1);
        }

        // t_hemm(CblasColMajor, CblasLeft, CblasLower, N, nev, &one, H, N,
        //     approxV, N, &one, workspace, N);
        MPI_distribute_V(mpi_handle, approxV, nev);
        MPI_distribute_W(mpi_handle, workspace, nev);
        MPI_doGemm(mpi_handle, T(1.0), T(1.0), 0, nev);
        assemble_C(mpi_handle, nev, workspace);

        Base<T> norm = t_lange('M', N, nev, workspace, N);
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

        t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nev, nev, N, &one,
            approxV, N, approxV, N, &neg_one, unity, nev);

        Base<T> norm = t_lange('M', nev, nev, unity, nev);
        delete[] unity;
        return norm;
    }

    void output(std::string str)
    {
        int rank;
        MPI_Comm_rank(mComm, &rank);
        if (rank == 0)
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

    MPI_Comm mComm;
    MPI_Handler<T>* mpi_handle;

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
