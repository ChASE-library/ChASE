/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#ifndef CHASE_ALGORITHM_INTERFACE_HPP
#define CHASE_ALGORITHM_INTERFACE_HPP

#include "configuration.hpp"
#include "types.hpp"

namespace chase
{

template <class T>
class Chase
{
public:
    //! This member function shifts the diagonal of matrix `A` with a shift
    //! value `c`. 
    //! @param c: the shift value on the diagonal of matrix `A`.
    virtual void Shift(T c, bool isunshift = false) = 0;
    //! This member function computes \f$V1 = alpha * H*V2 + beta *
    //! V1\f$ or \f$V2 = alpha * H'*V1 + beta *
    //! V2\f$. This operation starts with an offset `offset` of column,
    /*!
        @param block: the number of vectors in `V1` and `V2` for this `HEMM`
        operation.
        @param alpha: a scalar of type `T` which scales `H*V1` or `H'*V2`.

        @param beta: a scalar of type `T` which scales `V1` or `V2`, respectively.
        
        @param offset: the offset of column which the `HEMM` starts from.
    */
    virtual void HEMM(std::size_t nev, T alpha, T beta, std::size_t offset) = 0;
    //! This member function performs a QR factorization with an explicit
    //! construction of the unitary matrix `Q`. 
    //!
    //! CholQR is used in default. Optionally, user can decide to use Househoulder QR.
    //! @param fixednev: total number of converged eigenpairs before this time
    //! QR factorization.    
    virtual void QR(std::size_t fixednev, Base<T> cond) = 0;
    /*! This function performs the Rayleigh-Ritz projection, which projects the
       eigenproblem to be a small one, then solves the small problem and
       reconstructs the eigenvectors.
        @param ritzv: a pointer to the array which stores the computed
       eigenvalues.
        @param block: the number of non-converged eigenpairs, which determines
       the size of small eigenproblem.
    */    
    virtual void RR(Base<T>* ritzv, std::size_t block) = 0;
    //! This member function computes the residuals of unconverged eigenpairs.
    /*!
         @param ritzv: an array stores the eigenvalues.
         @param resid: a pointer to an array which stores the residual for each
       eigenpairs.
         @param fixednev: number of converged eigenpairs. Thus the number of
       non-converged one which perform this operation of computing residual is
       `nev_ + nex_ + fixednev_`.
    */    
    virtual void Resd(Base<T>* ritzv, Base<T>* resd, std::size_t fixednev) = 0;
    //! It estimates the upper bound of user-interested spectrum by Lanczos
    //! eigensolver
    //! @param m: the iterative steps for Lanczos eigensolver.
    //! @param upperb: a pointer to the upper bound estimated by Lanczos
    //! eigensolver.
    virtual void Lanczos(std::size_t m, Base<T>* upperb) = 0;
    //! This member function implements the virtual one declared in Chase class.
    //! It estimates the upper bound of user-interested spectrum by Lanczos
    //! eigensolver    
    virtual void Lanczos(std::size_t M, std::size_t idx, Base<T>* upperb,
                         Base<T>* ritzv, Base<T>* Tau, Base<T>* ritzV) = 0;
    virtual void LanczosDos(std::size_t idx, std::size_t m, T* ritzVc) = 0;
    //! This function swaps the two columns in a matrix used in the Chebyschev filter
    //! @param i: one of the column index to be swapped
    //! @param j: another of the column index to be swapped
    virtual void Swap(std::size_t i, std::size_t j) = 0;
    //! It locks the `new_converged` eigenvectors, which makes `locked_ +=
    //! new_converged`.
    //! @param new_converged: number of newly converged eigenpairs in the
    //! present iterative step.    
    virtual void Lock(std::size_t new_converged) = 0;
    //! It indicates the starting point to solve a (new) eigenproblem.    
    virtual void Start() = 0;
    //! It indicates the ending point of solving an eigenproblem
    virtual void End() = 0;
    //! This member function initializes randomly the vectors when necessary.
    /*!
        @param random: a boolean variable indicates if randomness of initial vectors
        is required. 
        - For solving a sequence of eigenvalue problems, this variable is
        always `True` when solving the first problem. 
        - It could be false when
        the ritzv vectors from previous problem are recycled to speed up the 
        convergence.
    */
    virtual void initVecs(bool random) = 0;
    //! Return size of matrix
    virtual std::size_t GetN() const = 0;
    //! Return the number of eigenpairs to be computed 
    virtual std::size_t GetNev() = 0;
    //! Return the external searching space size
    virtual std::size_t GetNex() = 0;
    //! Return the computed ritz values
    virtual Base<T>* GetRitzv() = 0;
    //! Return the residuals of computed ritz pairs
    virtual Base<T>* GetResid() = 0;
    //! Return a class which contains the configuration parameters
    virtual ChaseConfig<T>& GetConfig() = 0;
    //! Return the number of MPI procs used, it is `1` when sequential ChASE is used
    virtual int get_nprocs() = 0;
#ifdef CHASE_OUTPUT
    //! Print some intermediate infos during the solving procedure 
    virtual void Output(std::string str) = 0;
#endif
};
} // namespace chase

#endif
