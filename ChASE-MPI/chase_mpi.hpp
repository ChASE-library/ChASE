/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2023, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <cstring> //memcpy
#include <iostream>
#include <memory>

#include "mpi.h"

#include <random>

#include "algorithm/chase.hpp"

#include "blas_templates.hpp"
#include "chase_mpi_matrices.hpp"

#include "./impl/chase_mpidla.hpp"

namespace chase
{
namespace mpi
{

//! @brief A derived class of Chase to implement ChASE based on MPI and Dense
//! Linear Algebra (**DLA**) routines.
/*!
  This is a calls derived from the Chase class which plays the
    role of interface for the kernels used by the library.
    - All members of the Chase class are virtual functions. These
    functions are re-implemented in the ChaseMpi
    class.
    - All the members functions of ChaseMpi, which are the implementation of the
  virtual functions in class Chase, are implemented using the *DLA* routines
  provided by the class ChaseMpiDLAInterface.
    -  The DLA functions in ChaseMpiDLAInterface are also vritual functions,
  which are differently implemented targeting different computing architectures
  (sequential/parallel, CPU/GPU, shared-memory/distributed-memory, etc). In the
  class ChaseMpi, the calling of DLA functions are indeed calling their
  implementations from different derived classes. Thus this ChaseMpi class is
  able to have customized implementation for various architectures.
    - For the implementation of the class ChaseMpi targeting distributed-memory
  platforms based on MPI, the setup of MPI environment and communication scheme,
    and the distribution of data (matrix, vectors) across MPI nodes are
  following the ChaseMpiProperties class, the distribution of matrix can be
  either **Block** or **Block-Cyclic** scheme.
    @tparam MF: A class derived from ChaseMpiDLAInterface, which indicates the
  selected implementation of DLA that to be used by ChaseMpi Object.
    @tparam T: the scalar type used for the application. ChASE is templated
    for real and complex numbers with both Single Precision and Double
  Precision, thus `T` can be one of `float`, `double`, `std::complex<float>` and
    `std::complex<double>`.

*/
template <template <typename> class MF, class T>
class ChaseMpi : public chase::Chase<T>
{
public:
    //! A constructor of the ChaseMpi class which gives an implenentation of
    //! ChASE for shared-memory architecture, without MPI.
    /*!
       The private members of this classes are initialized by the parameters of
       this constructor.
       - For the
       variable `N_`, it is initialized by the first parameter of this
       constructor `N`.
       - The variables `nev_` and
       and `nex_` are initialized by the parameters of this constructor `nev`
       and `nex`, respectively.
       - The variable `rank_` is set to be 0 since non MPI is supported. The
       variable `locked_` is initially set to be 0.
       - The variable `config_` is setup by the constructor of ChaseConfig which
       takes the parameters `N`, `nev` and `nex`.
       - The variable `matrices_` is setup directly by the constructor of
       ChaseMpiMatrices which takes the paramter `H`, `ldh`, `N`, `nev`, `nex`,
       `V1`, `ritzv`, `V2` and `resid`.
       - The variable `dla_` is initialized by `MF` which is a derived class
       of ChaseMpiDLAInterface. In ChASE, the candidates of `MF` for non-MPI
       case are the classes `ChaseMpiDLABlaslapackSeq`,
       `ChaseMpiDLABlaslapackSeqInplace` and `ChaseMpiDLACudaSeq`.

       @param N: size of the square matrix defining the eigenproblem.
       @param nev: Number of desired extremal eigenvalues.
       @param nex: Number of eigenvalues augmenting the search space. Usually a
       relatively small fraction of `nev`.
       @param H: a pointer to the memory which stores local part of matrix on
       each MPI rank. If `H` is not provided by the user, it will be internally
       allocated in ChaseMpiMatrices class.
       @param ldh: leading dimension of `H`
       @param V1: a pointer to a rectangular matrix of size `N * (nev+nex)`.
       If `V1` is not provided by the user, it will be internally allocated in
       ChaseMpiMatrices class.
       After the solving step, the first `nev` columns of `V1` are overwritten
       by the desired Ritz vectors.
       @param ritzv: a pointer to an array to store the computed Ritz values.
       If `ritzv` is not provided by the user, it will be internally allocated
       in ChaseMpiMatrices class. Its minimal size should be `nev+nex`.
       @param V2: a pointer to anther rectangular matrix of size `N *
       (nev+nex)`. f `V2` is not provided by the user, it will be internally
       allocated in ChaseMpiMatrices class.
       @param resid: a pointer to an array to store the residual of computed
       eigenpairs. If `resid` is not provided by the user, it will be internally
       allocated in ChaseMpiMatrices class. Its minimal size should be
       `nev+nex`.
    */
    ChaseMpi(std::size_t N, std::size_t nev, std::size_t nex, T* H,
             std::size_t ldh, T* V1, Base<T>* ritzv, T* V2 = nullptr,
             Base<T>* resid = nullptr)
        : N_(N), nev_(nev), nex_(nex), rank_(0), locked_(0),
          config_(N, nev, nex),
          dla_(new MF<T>(H, ldh, V1, ritzv, N_, nev_, nex_))
    {

        ritzv_ = dla_->get_Ritzv();
        resid_ = dla_->get_Resids();
    }

    // case 2: MPI
    //! A constructor of the ChaseMpi class which gives an implenentation of
    //! ChASE for distributed-memory architecture, with the support of MPI.
    //! In this case, the buffer for matrix `H` should be allocated externally
    //! and provided by users.
    /*!
       The private members of this classes are initialized by the parameters of
       this constructor.
       - For the
       variable `N_`, `nev_` and `nex_` are initialized by the first parameter
       of this constructor `properties_`.
       - The variable `rank_` is initialized by `MPI_Comm_rank`. The variable
       `locked_` is initially set to be 0.
       - The variable `config_` is setup by the constructor of ChaseConfig which
       takes the parameters `N`, `nev` and `nex`.
       - The variable `matrices_` is constructed by the `create_matrices`
       function defined in ChaseMpiProperties.
       - The variable `dla_` for the distributed-memory is initialized by the
       constructor of ChaseMpiDLA which takes both `properties_` and `MF`. In
       MPI case, the implementation of are split into two classses:
          - the class ChaseMpiDLA implements mainly the MPI collective
       communication part of ChASE with MPI support.
          - The local computation tasks within each MPI is implemented by
       another two classes derived also from the class ChaseMpiDLAInterface:
       ChaseMpiDLABlaslapack for pure-CPUs version and ChaseMpiDLAMultiGPU
       for multi-GPUs version.
          - Thus, for this constructor, a combination of ChaseMpiDLA and one of
       ChaseMpiDLABlaslapack and ChaseMpiDLAMultiGPU is required.

       @param properties: an object of ChaseMpiProperties which setups the MPI
       environment and data distribution scheme for ChaseMpi targeting
       distributed-memory systems.
       @param H: a pointer to a local matrix of the distributed
       Symmetric/Hermtitian matrix to be diagonalised. It should be provided and
       allocated by users. The minimal dimension is `mxn`, in which `m` and `n`
       can be obtained through ChaseMpiProperties::get_m() and
       ChaseMpiProperties::get_n().
       @param ldh: leading dimension of local matrix `H`, it should be `>=`
       ChaseMpiProperties::get_m()
       @param V1: a pointer to a rectangular matrix of size `m * (nev+nex)`.
       `m` can be obtained through ChaseMpiProperties::get_m().
       `V1` is partially
       distributed within each `column communicator` and is redundant among
       different `column communicator`.
       If `V1` is not provided by the user, it will be internally allocated in
       ChaseMpiMatrices class.
       After the solving step, the first `nev` columns of `V1` are overwritten
       by the desired Ritz vectors.
       @param ritzv: a pointer to an array to store the computed Ritz values.
       If `ritzv` is not provided by the user, it will be internally allocated
       in ChaseMpiMatrices class. Its minimal size should be `nev+nex`.
       @param V2: a pointer to anther rectangular matrix of size `n *
       (nev+nex)`. `V2` is partially
       distributed within each `row communicator` and is redundant among
       different `row communicator`.
       If `V2` is not provided by the user, it will be internally
       allocated in ChaseMpiMatrices class. `m` can be obtained through
       ChaseMpiProperties::get_n().
       @param resid: a pointer to an array to store the residual of computed
       eigenpairs. If `resid` is not provided by the user, it will be internally
       allocated in ChaseMpiMatrices class. Its minimal size should be
       `nev+nex`.
    */

    ChaseMpi(ChaseMpiProperties<T>* properties, T* H, std::size_t ldh, T* V1,
             Base<T>* ritzv)
        : N_(properties->get_N()), nev_(properties->GetNev()),
          nex_(properties->GetNex()), locked_(0), config_(N_, nev_, nex_),
          properties_(properties),
          dla_(new ChaseMpiDLA<T>(
              properties_.get(),
              new MF<T>(properties_.get(), H, ldh, V1, ritzv)))
    {
        int init;
        MPI_Initialized(&init);
        if (init)
            MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

        ritzv_ = dla_->get_Ritzv();
        resid_ = dla_->get_Resids();

        static_assert(is_skewed_matrixfree<MF<T>>::value,
                      "MatrixFreeChASE Must be skewed");
    }

    ChaseMpi(ChaseMpiProperties<T>* properties, T* H, std::size_t ldh, T* V1,
             Base<T>* ritzv, ChaseMpiDLAInterface<T>* dla_input)
        : N_(properties->get_N()), nev_(properties->GetNev()),
          nex_(properties->GetNex()), locked_(0), config_(N_, nev_, nex_),
          properties_(properties),
          dla_(std::unique_ptr<ChaseMpiDLAInterface<T>>(dla_input))
    {
        int init;
        MPI_Initialized(&init);
        if (init)
            MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

        ritzv_ = dla_->get_Ritzv();
        resid_ = dla_->get_Resids();
    }

    //! It prevents the copy operation of the constructor of ChaseMpi.
    ChaseMpi(const ChaseMpi&) = delete;

    ~ChaseMpi() {}

    //! This member function implements the virtual one declared in Chase class.
    //! \return `config_`: an object of ChaseConfig class which defines the
    //! eigensolver.
    ChaseConfig<T>& GetConfig() override { return config_; }

    //! This member function implements the virtual one declared in Chase class.
    //! Returns the rank of matrix A which is distributed within 2D MPI grid.
    //! \return `N_`: the rank of matrix `A`.
    std::size_t GetN() const override { return N_; }

    //! This member function implements the virtual one declared in Chase class.
    //! Returns Number of desired extremal eigenpairs, which was set by users.
    //! \return `nev_`: Number of desired extremal eigenpairs.
    std::size_t GetNev() override { return nev_; }

    //! This member function implements the virtual one declared in Chase class.
    //! Returns the Increment of the search subspace so that its total size,
    //! which was set by users. \return `nex_`: Increment of the search
    //! subspace.
    std::size_t GetNex() override { return nex_; }

    //! This member function implements the virtual one declared in Chase class.
    //! Returns the array which stores the computed Ritz values.
    //! \return `ritzv_`: an array stores the computed Ritz values.
    Base<T>* GetRitzv() override { return ritzv_; }

    //! This member function implements the virtual one declared in Chase class.
    //! It resets `locked_` to `0` and start to solve a (new) eigenproblem.
    void Start() override
    {
        locked_ = 0;
        dla_->Start();
    }
    //! This member function implements the virtual one declared in Chase class.
    //! It indicates the finalisation of solving a single eigenproblem.
    void End() override { dla_->End(); }

    bool checkSymmetryEasy() override { is_sym_ = dla_->checkSymmetryEasy(); return is_sym_; }

    bool isSym() {return is_sym_;}

    void symOrHermMatrix(char uplo) override { dla_->symOrHermMatrix(uplo); }

    //! This member function implements the virtual one declared in Chase class.
    //! This member function shifts the diagonal of matrix `A` with a shift
    //! value `c`. It is implemented by calling the different implementation in
    //! the derived class of `ChaseMpiDLAInterface`. For the construction of
    //! ChaseMpi with MPI, this function is naturally in parallel across the MPI
    //! nodes, since the shift operation only takes places on selected local
    //! matrices which stores in a distributed manner.
    //! @param c: the shift value on the diagonal of matrix `A`.
    void Shift(T c, bool isunshift = false) override
    {
        dla_->shiftMatrix(c, isunshift);
    };

    //! This member function implements the virtual one declared in Chase class.
    //! This member function initializes randomly the vectors when necessary.
    /*!
        @param random: a boolean variable indicates if randomness of initial
       vectors is required.
        - For solving a sequence of eigenvalue problems, this variable is
        always `True` when solving the first problem.
        - It could be false when
        the ritzv vectors from previous problem are recycled to speed up the
        convergence.
    */
    void initVecs(bool random) override
    {
        if (random)
        {
#ifdef USE_NSIGHT
            nvtxRangePushA("InitRndVecs");
#endif
            dla_->initRndVecs();
#ifdef USE_NSIGHT
            nvtxRangePop();
#endif
        }
        dla_->initVecs();
    }

    //! This member function implements the virtual one declared in Chase class.
    //! This member function computes \f$V1 = alpha * H*V2 + beta *
    //! V1\f$ or \f$V2 = alpha * H'*V1 + beta *
    //! V2\f$. This operation starts with an offset `offset` of column,
    /*!
        @param block: the number of vectors in `V1` and `V2` for this `HEMM`
        operation.
        @param alpha: a scalar of type `T` which scales `H*V1` or `H'*V2`.

        @param beta: a scalar of type `T` which scales `V1` or `V2`,
       respectively.

        @param offset: the offset of column which the `HEMM` starts from.
    */
    void HEMM(std::size_t block, T alpha, T beta, std::size_t offset) override
    {
#ifdef USE_NSIGHT
        nvtxRangePushA("HEMM");
#endif
        dla_->apply(alpha, beta, offset, block, locked_);
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
    };

    //! This member function performs a QR factorization with an explicit
    //! construction of the unitary matrix `Q`. After the explicit construction
    //! of Q, its first `locked_` number of vectors are overwritten by the
    //! converged eigenvectors stored previously.
    //!
    //! CholQR is used in default, and it can be disabled by adding a
    //! environment variable `CHASE_DISABLE_CHOLQR=1`.
    //! When CholQR is disabled, ScaLAPACK Householder QR is used whenever
    //! possible,
    //! @param fixednev: total number of converged eigenpairs before this time
    //! QR factorization.
    void QR(std::size_t fixednev, Base<T> cond) override
    {
        int grank = 0;
#ifdef USE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &grank);
#endif
        bool isHHqr = false;
        int disable = config_.DoCholQR() ? 0 : 1;
        char* cholddisable = getenv("CHASE_DISABLE_CHOLQR");
        if (cholddisable) {
            disable = std::atoi(cholddisable);
        }

        Base<T> cond_threshold_upper = (sizeof(Base<T>) == 8) ? 1e8 : 1e4;
        Base<T> cond_threshold_lower = (sizeof(Base<T>) == 8) ? 2e1 : 1e1;

        char* chol_threshold = getenv("CHASE_CHOLQR1_THLD");
        if (chol_threshold)
        {
            cond_threshold_lower = std::atof(chol_threshold);
        }

        int display_bounds = 0;
        char* display_bounds_env = getenv("CHASE_DISPLAY_BOUNDS");
        if (display_bounds_env)
        {
            display_bounds = std::atoi(display_bounds_env);
        }
        
        if (disable == 1)
        {
            dla_->hhQR(locked_);
            isHHqr = true;
        }
        else
        {
#ifdef CHASE_OUTPUT
            if (grank == 0)
            {
                std::cout << std::setprecision(2) << "cond(V): " << cond << std::endl;
            }
#endif
            if (display_bounds != 0)
            {
              dla_->estimated_cond_evaluator(locked_, cond);
            }

            int info = 1;

            if (cond > cond_threshold_upper)
            {
                info = dla_->shiftedcholQR2(locked_);
            }
            else if(cond < cond_threshold_lower)
            {
                info = dla_->cholQR1(locked_);
            }
            else
            {
                info = dla_->cholQR2(locked_);                         
            }

            if (info != 0)
            {
#ifdef CHASE_OUTPUT
                if(grank == 0)
                {
                    std::cout << "CholeskyQR doesn't work, Househoulder QR will be used." << std::endl;
                }
#endif
                dla_->hhQR(locked_);
                isHHqr = true;
            }
        }

        dla_->lockVectorCopyAndOrthoConcatswap(locked_, isHHqr);
    }

    //! This member function implements the virtual one declared in Chase class.
    /*! This function performs the Rayleigh-Ritz projection, which projects the
       eigenproblem to be a small one, then solves the small problem and
       reconstructs the eigenvectors.
        @param ritzv: a pointer to the array which stores the computed
       eigenvalues.
        @param block: the number of non-converged eigenpairs, which determines
       the size of small eigenproblem.
    */
    void RR(Base<T>* ritzv, std::size_t block) override
    {
        dla_->RR(block, locked_, ritzv);
    };

    //! This member function implements the virtual one declared in Chase class.
    //! This member function computes the residuals of unconverged eigenpairs.
    /*!
         @param ritzv: an array stores the eigenvalues.
         @param resid: a pointer to an array which stores the residual for each
       eigenpairs.
         @param fixednev: number of converged eigenpairs. Thus the number of
       non-converged one which perform this operation of computing residual is
       `nev_ + nex_ + fixednev_`.
    */
    void Resd(Base<T>* ritzv, Base<T>* resid, std::size_t fixednev) override
    {
        std::size_t unconverged = (nev_ + nex_) - fixednev;
        dla_->Resd(ritzv, resid, locked_, unconverged);
    };

    //! This member function implements the virtual one declared in Chase class.
    //! It swaps the two matrices of vectors used in the Chebyschev filter
    //! @param i: one of the column index to be swapped
    //! @param j: another of the column index to be swapped
    //! rectangular matrices `approxV_` and `workspace_`.
    void Swap(std::size_t i, std::size_t j) override
    {
#ifdef USE_NSIGHT
        nvtxRangePushA("Swap");
#endif
        dla_->Swap(i, j);
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
    };

    //! This member function implements the virtual one declared in Chase class.
    //! It estimates the upper bound of user-interested spectrum by Lanczos
    //! eigensolver
    //! @param m: the iterative steps for Lanczos eigensolver.
    //! @param upperb: a pointer to the upper bound estimated by Lanczos
    //! eigensolver.
    void Lanczos(std::size_t m, Base<T>* upperb) override
    {
        // todo
        Base<T>* d = new Base<T>[m]();
        Base<T>* e = new Base<T>[m]();

        int idx_ = -1;
        Base<T> real_beta;
        dla_->Lanczos(m, idx_, d, e, &real_beta);

#ifdef USE_NSIGHT
        nvtxRangePushA("Stemr");
#endif
        int notneeded_m;
        std::size_t vl, vu;
        Base<T> ul, ll;
        int tryrac = 0;
        int* isuppz = new int[2 * m];
        Base<T>* ritzv = new Base<T>[m];

        t_stemr<Base<T>>(LAPACK_COL_MAJOR, 'N', 'A', m, d, e, ul, ll, vl, vu,
                         &notneeded_m, ritzv, NULL, m, m, isuppz, &tryrac);

        *upperb = std::max(std::abs(ritzv[0]), std::abs(ritzv[m - 1])) +
                  std::abs(real_beta);
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif

        delete[] ritzv;
        delete[] isuppz;
        delete[] d;
        delete[] e;
    };

    // we need to be careful how we deal with memory here
    // we will operate within Workspace
    //! This member function implements the virtual one declared in Chase class.
    //! It estimates the upper bound of user-interested spectrum by Lanczos
    //! eigensolver
    void Lanczos(std::size_t M, std::size_t idx, Base<T>* upperb,
                 Base<T>* ritzv, Base<T>* Tau, Base<T>* ritzV) override
    {
        // todo
        std::size_t m = M;
        Base<T>* d = new Base<T>[m]();
        Base<T>* e = new Base<T>[m]();

        int idx_ = static_cast<int>(idx);
        Base<T> real_beta;

        dla_->Lanczos(m, idx_, d, e, &real_beta);

#ifdef USE_NSIGHT
        nvtxRangePushA("Stemr");
#endif
        int notneeded_m;
        std::size_t vl, vu;
        Base<T> ul, ll;
        int tryrac = 0;
        int* isuppz = new int[2 * m];
        t_stemr(LAPACK_COL_MAJOR, 'V', 'A', m, d, e, ul, ll, vl, vu,
                &notneeded_m, ritzv, ritzV, m, m, isuppz, &tryrac);
	*upperb = std::max(std::abs(ritzv[0]), std::abs(ritzv[m - 1])) +
                  std::abs(real_beta);
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
        for (std::size_t k = 1; k < m; ++k)
        {
            Tau[k] = std::abs(ritzV[k * m]) * std::abs(ritzV[k * m]);
        }

        delete[] isuppz;
        delete[] d;
        delete[] e;
    };

    //! This member function implements the virtual one declared in Chase class.
    //! It locks the `new_converged` eigenvectors, which makes `locked_ +=
    //! new_converged`.
    //! @param new_converged: number of newly converged eigenpairs in the
    //! present iterative step.
    void Lock(std::size_t new_converged) override { locked_ += new_converged; };

    //! This member function implements the virtual one declared in Chase class.
    //! It estimates the spectral distribution of eigenvalues.
    void LanczosDos(std::size_t idx, std::size_t m, T* ritzVc) override
    {
        dla_->LanczosDos(idx, m, ritzVc);
    }

#ifdef CHASE_OUTPUT
    void Output(std::string str) override
    {
        if (rank_ == 0)
            std::cout << str;
    }
#endif

    //! \return `H_`: A pointer to the memory allocated to store (local part if
    //! applicable) of matrix `A`.
    // T* GetMatrixPtr() { return matrices_.get_H(); }

    //! This member function implements the virtual one declared in Chase class.
    //! \return `resid_`: a  pointer to the memory allocated to store the
    //! residual of each computed eigenpair.
    Base<T>* GetResid() override { return resid_; }

    //! This member function return the number of MPI processes used by ChASE
    //! \return the number of MPI ranks in the communicator used by ChASE
    int get_nprocs() override { return properties_.get()->get_nprocs(); }

private:
    //! Global size of the matrix A defining the eigenproblem.
    /*!
      - For the constructor of class ChaseMpi without MPI,
      this variable is initialized by the constructor using the value of the
      first of its input parameters `N`.
      - For the constructor of class ChaseMpi with MPI, this variable is
      initialized by `properties_`, a pointer to an object of
      ChaseMpiProperties. This variable is private, it can be access by the
      member function GetN().
    */
    const std::size_t N_;

    //! Number of desired extremal eigenpairs.
    /*!
      - For the constructor of class ChaseMpi without MPI,
      this variable is initialized by the constructor using the value of the
      second of its input parameters `nev`.
      - For the constructor of class ChaseMpi with MPI, this variable is
      initialized by `properties`, a pointer to an object of ChaseMpiProperties.
      This variable is private, it can be access by the member function
      GetNev().
    */
    const std::size_t nev_;

    //! Increment of the search subspace so that its total size is `nev+nex`.
    /*!
      - For the constructor of class ChaseMpi without MPI,
      this variable is initialized by the constructor using the value of the
      third of its input parameters `nex`.
      - For the constructor of class ChaseMpi with MPI, this variable is
      initialized by `properties`, a pointer to an object of ChaseMpiProperties.
      This variable is private, it can be access by the member function
      GetNex().
    */
    const std::size_t nex_;

    //! The rank of each MPI node with the working MPI communicator
    /*!
      - For the constructor of class ChaseMpi without MPI, this variable is `0`.
      - For the constructor of class ChaseMpi with MPI, this variable is gotten
      through MPI function `MPI_Comm_rank`.
    */
    int rank_;

    //! The number of eigenvectors to be locked, which indicates the number of
    //! eigenpairs converged into acceptable tolerance.
    /*!
      - For the constructor of class ChaseMpi without MPI,
      this variable is initialized `0`.
      - For the constructor of class ChaseMpi with MPI, this variable is
      initialized also `0`.
      - During the process of ChASE solving eigenproblem, the value of this
      variable will increase with more and more eigenpairs computed.
    */
    std::size_t locked_;

    //! A pointer to the memory allocated to store the computed eigenvalues. The
    //! values inside are always real since the matrix of eigenproblem is
    //! Hermitian (symmetric).
    /*!
      - For the constructor of class ChaseMpi without MPI,
      the memory is allocated directly by ChaseMpiMatrices of size `nex_ +
      nex_`.
      - For the constructor of class ChaseMpi with MPI, this variable is a
      pointer to a vector of size `nex_ + nex_`, which are identical on each MPI
      node. It is initalized within the construction of ChaseMpiMatrices.
      - The final converged eigenvalues will be stored in it.
      - This variable is private, and it can be assessed through the member
      function GetRitzv().
    */
    Base<T>* ritzv_;

    //! A pointer to the memory allocated to store the residual of each computed
    //! eigenpair. The values inside are always real.
    /*!
      - For the constructor of class ChaseMpi without MPI,
      the memory is allocated directly by ChaseMpiMatrices of size `nex_ +
      nex_`.
      - For the constructor of class ChaseMpi with MPI, this variable is a
      pointer to a vector of size `nex_ + nex_`, which are identical on each MPI
      node. It is initalized within the construction of ChaseMpiMatrices.
      - The residuals of final converged eigenvalues will be stored in it.
      - This variable is private, and it can be assessed through the member
      function GetResid().
    */
    Base<T>* resid_;

    //! A smart pointer to an object of ChaseMpiProperties class which is used
    //! by ChaseMpi class. This variable only matters for the constructor of
    //! class ChaseMpi with MPI.
    std::unique_ptr<ChaseMpiProperties<T>> properties_;

    //! An object of ChaseMpiMatrices to setup the matrices and vectors used by
    //! ChaseMpi class.
    // ChaseMpiMatrices<T> matrices_;

    //! A smart pointer to an object of ChaseMpiDLAInterface class which is used
    //! by ChaseMpi class.
    /*!
      - For the constructor of class ChaseMpi without MPI,
     this variable is initialized directly by the **template parameter** `MF`.
      - For the constructor of class ChaseMpi with MPI, this variable is
     implemented by considering `properties_`  and `MF`. It is initalized within
     the construction of ChaseMpiMatrices.
    */
    std::unique_ptr<ChaseMpiDLAInterface<T>> dla_;

    //! An object of ChaseConfig class which setup all the parameters of ChASE,
    //! these parameters are either provided by users, or using the default
    //! values. This variable is initialized by the constructor of ChaseConfig
    //! class by using `N`, `nev` and `nex`, which are the parameters of the
    //! constructors of ChaseMpi.
    ChaseConfig<T> config_;

    bool is_sym_;
};

} // namespace mpi
} // namespace chase
