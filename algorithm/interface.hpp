// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#ifndef CHASE_ALGORITHM_INTERFACE_HPP
#define CHASE_ALGORITHM_INTERFACE_HPP

#include "configuration.hpp"
#include "types.hpp"

namespace chase
{
/**
 * @page chase_algorithm_interface Algorithm Interface Overview
 *
 * This page describes the `ChaseBase` class, which is an abstract base class
 * that defines the interface for various numerical methods used in ChASE. These
 * methods include eigenvalue solvers, matrix factorizations, and other linear
 * algebra operations commonly used in quantum materials simulations.
 *
 * The methods in this interface are intended to be implemented by different
 * algorithmic classes that provide specific functionality for the ChASE
 * library.
 *
 * @defgroup algorithm_interface Algorithm Interface
 * This group includes the core algorithmic interface for ChASE, which defines
 * operations for solving eigenproblems, matrix factorizations, and other linear
 * algebra routines.
 *
 * @{
 */

/**
 * @brief Base class for all ChASE algorithmic operations.
 *
 * This abstract class defines the common interface for all algorithmic
 * operations in ChASE. Derived classes are expected to implement these methods,
 * which include matrix factorization, eigenvalue problem solving, and other
 * numerical operations.
 *
 * @tparam T The type of data used in matrix and vector operations (e.g.,
 * `float`, `double`).
 */
template <class T>
class ChaseBase
{
public:
    /**
     * @brief Shifts the diagonal of matrix `A` with a specified value `c`.
     *
     * This method applies a shift to the diagonal of the matrix `A` by the
     * specified amount `c`. Optionally, the shift can be undone if the
     * `isunshift` flag is set to true.
     *
     * @param c The shift value applied to the diagonal of matrix `A`.
     * @param isunshift If true, undoes the shift applied to the diagonal.
     */
    virtual void Shift(T c, bool isunshift = false) = 0;
    /**
     * @brief Performs a matrix operation of the form \(V1 = \alpha \cdot H
     * \cdot V2 + \beta \cdot V1\).
     *
     * This method computes the matrix operation for the input vectors \(V1\)
     * and \(V2\) using scalars
     * \(\alpha\) and \(\beta\), where \(H\) is a matrix and \(V1\) and \(V2\)
     * are input vectors.
     *
     * @param nev Number of eigenpairs involved in the operation.
     * @param alpha Scalar that scales the result of \(H \cdot V2\).
     * @param beta Scalar that scales the result of \(V1\).
     * @param offset Column offset for the operation.
     */
    virtual void HEMM(std::size_t nev, T alpha, T beta, std::size_t offset) = 0;
    /**
     * @brief Performs QR factorization with optional Householder
     * transformation.
     *
     * This method performs a QR factorization on the matrix, optionally using
     * Householder transformations for the computation.
     *
     * @param fixednev Total number of converged eigenpairs before the QR
     * factorization.
     * @param cond Condition number for controlling the factorization process.
     */
    virtual void QR(std::size_t fixednev, Base<T> cond) = 0;
    /**
     * @brief Performs Rayleigh-Ritz projection for eigenproblem reduction.
     *
     * This method projects the eigenproblem into a smaller subspace, solves the
     * smaller problem, and reconstructs the eigenvectors.
     *
     * @param ritzv Array to store the computed eigenvalues.
     * @param block The number of non-converged eigenpairs used for the small
     * eigenproblem.
     */
    virtual void RR(Base<T>* ritzv, std::size_t block) = 0;
    /**
     * @brief Sort the ritz values, the residuals, and the eigenvectors in
     * ascending order.
     *
     * This method sorts the ritz values, the residuals, and the ritz vectors in
     * ascending order of the ritz values.
     *
     * @param ritzv Array to store the computed eigenvalues.
     * @param residLast Array to store the residuals of the previous iteration.
     * @param resid Array to store the residuals of the current iteration.
     */
    virtual void Sort(Base<T>* ritzv, Base<T>* residLast, Base<T>* resid) = 0;
    /**
     * @brief Computes residuals for unconverged eigenpairs.
     *
     * This method computes the residuals of the eigenpairs that have not yet
     * converged.
     *
     * @param ritzv Array containing the eigenvalues.
     * @param resd Array to store the computed residuals for each eigenpair.
     * @param fixednev Number of converged eigenpairs, which helps determine the
     * number of non-converged eigenpairs.
     */
    virtual void Resd(Base<T>* ritzv, Base<T>* resd, std::size_t fixednev) = 0;
    /**
     * @brief Estimates the upper bound of the user-interested spectrum using
     * Lanczos eigensolver.
     *
     * This method estimates the upper bound of the spectrum of interest using
     * the Lanczos eigensolver.
     *
     * @param m Number of iterative steps for the Lanczos method.
     * @param upperb Pointer to store the estimated upper bound of the spectrum.
     */
    virtual void Lanczos(std::size_t m, Base<T>* upperb) = 0;
    /**
     * @brief Performs Lanczos eigensolver with additional parameters for
     * eigenvector and Tau storage.
     *
     * This method performs the Lanczos eigensolver with more advanced
     * parameters, storing eigenvectors and Tau values for further computation.
     *
     * @param M Number of iterations for the Lanczos algorithm.
     * @param numvec Number of vectors to be used in the Lanczos procedure.
     * @param upperb Pointer to store the upper bound of the spectrum.
     * @param ritzv Pointer to store the Ritz values.
     * @param Tau Pointer to store the Tau values.
     * @param ritzV Pointer to store the Ritz vectors.
     */
    virtual void Lanczos(std::size_t M, std::size_t numvec, Base<T>* upperb,
                         Base<T>* ritzv, Base<T>* Tau, Base<T>* ritzV) = 0;

    /**
     * @brief Lanczos method for Density of States (DOS) computation.
     *
     * This method performs a Lanczos procedure for computing the density of
     * states.
     *
     * @param idx Index for the specific Lanczos vector.
     * @param m Number of iterations for the Lanczos method.
     * @param ritzVc Pointer to store the Ritz vectors.
     */
    virtual void LanczosDos(std::size_t idx, std::size_t m, T* ritzVc) = 0;
    /**
     * @brief Swaps two columns in a matrix for Chebyshev filtering.
     *
     * This method swaps the specified columns of a matrix to facilitate
     * Chebyshev filtering.
     *
     * @param i One of the column indices to be swapped.
     * @param j Another column index to be swapped.
     */
    virtual void Swap(std::size_t i, std::size_t j) = 0;
    /**
     * @brief Locks the newly converged eigenvectors.
     *
     * This method locks the newly converged eigenvectors and increments the
     * locked count.
     *
     * @param new_converged The number of newly converged eigenvectors.
     */
    virtual void Lock(std::size_t new_converged) = 0;
    /**
     * @brief Checks for easy symmetry in the matrix.
     *
     * This method performs a simple check for matrix symmetry.
     *
     * @return `true` if the matrix is symmetric, otherwise `false`.
     */
    virtual bool checkSymmetryEasy() = 0;
    /**
     * @brief Returns if the matrix is Hermitian or not
     *
     * This method simply returns if the matrix is Hermitian or not
     *
     * @return `true` if the matrix is Hermitian, otherwise `false`.
     */
    virtual bool isSym() = 0;
    /**
     * @brief Checks for easy pseudo-hermicity in the matrix.
     *
     * This method performs a simple check for matrix pseudo-hermicity
     *
     * @return `true` if the matrix is pseudo-hermicity, otherwise `false`.
     */
    virtual bool checkPseudoHermicityEasy() = 0;
    /**
     * @brief Returns if the matrix is Pseudo-Hermitian or not
     *
     * This method simply returns if the matrix is Pseudo-Hermitian or not
     *
     * @return `true` if the matrix is Pseudo-Hermitian, otherwise `false`.
     */
    virtual bool isPseudoHerm() = 0;
    /**
     * @brief Sets matrix type for symmetric or Hermitian matrix.
     *
     * This method defines whether the matrix is symmetric or Hermitian based on
     * the specified input.
     *
     * @param uplo Specifies whether the upper or lower triangular part of the
     * matrix is stored.
     */
    virtual void symOrHermMatrix(char uplo) = 0;
    /**
     * @brief Indicates the start of an eigenproblem solution.
     *
     * This method marks the beginning of the eigenproblem-solving process.
     */
    virtual void Start() = 0;
    /**
     * @brief Indicates the end of an eigenproblem solution.
     *
     * This method marks the conclusion of the eigenproblem-solving process.
     */
    virtual void End() = 0;
    /**
     * @brief Initializes vectors randomly, if necessary.
     *
     * This method initializes the vectors either randomly or from previously
     * computed Ritz values, depending on the input flag.
     *
     * @param random Flag to determine whether to initialize vectors randomly.
     */
    virtual void initVecs(bool random) = 0;
    /**
     * @brief Returns the size of the matrix.
     *
     * This method returns the size of the matrix (number of rows or columns).
     *
     * @return The size of the matrix.
     */
    virtual std::size_t GetN() const = 0;
    /**
     * @brief Returns the number of eigenpairs to be computed.
     *
     * This method returns the number of eigenpairs that need to be computed.
     *
     * @return The number of eigenpairs.
     */
    virtual std::size_t GetNev() = 0;
    /**
     * @brief Returns the value of `nex`, which is used for specifying the
     * number of eigenpairs.
     *
     * This method returns the size or number of eigenpairs required by the
     * algorithm for the computation.
     *
     * @return The number of eigenpairs as a `std::size_t` value.
     */
    virtual std::size_t GetNex() = 0;
    /**
     * @brief Returns the number of Lanczos Iterations.
     *
     * @return The number of Lanczos Iterations as a `std::size_t` value.
     */
    virtual std::size_t GetLanczosIter() = 0;
    /**
     * @brief Returns the number of runs of Lanczos.
     *
     * @return The number of runs of Lanczos as a `std::size_t` value.
     */
    virtual std::size_t GetNumLanczos() = 0;
    /**
     * @brief Returns the computed Ritz values.
     *
     * This method provides access to the Ritz values computed during the
     * algorithm's execution. These values are typically used for eigenvalue
     * approximations and further analysis.
     *
     * @return A pointer to an array of Ritz values of type `Base<T>`.
     */
    virtual Base<T>* GetRitzv() = 0;
    /**
     * @brief Returns the residuals of computed Ritz pairs.
     *
     * This method returns the residuals associated with the Ritz values.
     * Residuals are often used to measure the accuracy or convergence of the
     * eigenvalue problem solution.
     *
     * @return A pointer to an array of residuals corresponding to the Ritz
     * values of type `Base<T>`.
     */
    virtual Base<T>* GetResid() = 0;
    /**
     * @brief Returns the configuration parameters used in the algorithm.
     *
     * This method provides access to the configuration object that contains
     * various algorithm settings, parameters, and options.
     *
     * @return A reference to a `ChaseConfig<T>` class containing the
     * configuration parameters.
     */
    virtual ChaseConfig<T>& GetConfig() = 0;
    /**
     * @brief Returns the number of MPI processes used.
     *
     * This method returns the number of MPI processes involved in the
     * computation. If sequential ChASE is used, it returns `1`, indicating a
     * single process is running.
     *
     * @return The number of MPI processes used, represented as an integer.
     */
    virtual int get_nprocs() = 0;
    /**
     * @brief Returns the MPI rank.
     *
     * This method returns the MPI process. If sequential ChASE is used,
     * it returns `0`, indicating a single process is running.
     *
     * @return The number of MPI processes used, represented as an integer.
     */
    virtual int get_rank() = 0;
    /**
     * @brief Save the residuals of early locked eigenpairs
     *
     * This method enables saving statistical data on the early locked
     * eigenpairs to further extract statistical information.
     */
    virtual void
    set_early_locked_residuals(std::vector<Base<T>> early_locked_residuals) {};

#ifdef CHASE_OUTPUT
    /**
     * @brief Prints intermediate information during the solving procedure.
     *
     * This method is used to output intermediate information or debug data
     * during the execution of the algorithm. It is typically enabled when
     * debugging or logging is required.
     *
     * @param str A string containing the information to be printed.
     */
    virtual void Output(std::string str) = 0;
#endif
};
/** @} */
} // namespace chase

#endif
