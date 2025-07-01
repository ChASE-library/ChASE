// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#ifndef CHASE_ALGORITHM_ALGORITHM_HPP
#define CHASE_ALGORITHM_ALGORITHM_HPP

#include <algorithm>
#include <assert.h>
#include <iomanip>
#include <random>

#include "interface.hpp"
#include "Impl/chase_gpu/nvtx.hpp"

namespace chase
{
/**
 * @page algorithm Implementation of ChASE Solver
 * 
 * This page describes `chase::Algorithm`, the implementation of the ChASE solver using the abstract interfaces 
 * of numerical components defined in the `chase::chaseBase` class. The solver includes several 
 * key numerical methods and optimization procedures, such as:
 * - Chebyshev filter implementation
 * - Degree optimization of the filter
 * - Locking converged Ritz pairs
 * - Lanczos method to estimate the required bounds of the spectrum
 * - Solving procedure using the Chebyshev filter
 */

 /**
 * @defgroup algorithm_algorithm ChASE Solver Algorithm
 * 
 * This group includes the core methods that implement the ChASE solver, which are:
 * - Eigenproblem solving procedure
 * - Filter implementation
 * - Degree optimization and locking
 * - Lanczos method for spectrum bounds estimation
 */

/**
 * @brief Implementation of ChASE solver using the abstract of interfaces of numerical components defined in class chase::ChASE.
 * 
 * This class contains the essential methods to solve eigenproblems using the ChASE algorithm, 
 * including Lanczos for spectrum bounds estimation, Chebyshev filters, and degree optimization for the filter.
 * 
 * @tparam T The data type used in the solver (e.g., `float`, `double`).
 * 
 * @ingroup algorithm_algorithm
 */
template <class T>
class Algorithm
{
public:
    /**
     * @brief Solves eigenproblems using the ChASE solver.
     * 
     * This static method is used to solve eigenvalue problems using the ChASE algorithm, 
     * utilizing the provided kernel (which is a `ChaseBase<T>`).
     * 
     * @param single A pointer to an object that implements the `ChaseBase<T>` interface. 
     * The object contains the actual computation logic for the solver.
     */
    static void solve(ChaseBase<T>* single);
    /**
     * @brief Optimizes the degree of the Chebyshev polynomial based on the convergence of Ritz pairs.
     * 
     * This method optimizes the degree of the Chebyshev polynomial filter, adjusting it dynamically 
     * based on the convergence behavior of the Ritz pairs.
     * 
     * @param kernel A pointer to the `ChaseBase<T>` instance, which provides the necessary numerical operations.
     * @param N The total size of the problem.
     * @param unconverged The number of unconverged eigenpairs.
     * @param nex The number of external searching space.
     * @param upperb The upper bound for the spectrum.
     * @param lowerb The lower bound for the spectrum.
     * @param tol The tolerance for the degree calculation.
     * @param ritzv The array storing the Ritz values.
     * @param resid The array storing the residuals of the Ritz values.
     * @param residLast The array storing the residuals from the previous iteration.
     * @param degrees The array storing the degrees for each Ritz pair.
     * @param locked The number of locked eigenpairs.
     * 
     * @return The optimized polynomial degree.
     */
    static std::size_t calc_degrees(ChaseBase<T>* kernel, std::size_t N,
                                    std::size_t unconverged, std::size_t nex,
                                    Base<T> upperb, Base<T> lowerb, Base<T> tol,
                                    Base<T>* ritzv, Base<T>* resid,
                                    Base<T>* residLast, std::size_t* degrees,
                                    std::size_t locked);
    
    /**
     * @brief Specialized degree optimization for quasi-hermitian matrices with cluster-aware features.
     * 
     * This method provides enhanced degree optimization specifically designed for quasi-hermitian 
     * eigenvalue problems, including cluster detection and adaptive degree spacing.
     * 
     * @param kernel A pointer to the `ChaseBase<T>` instance, which provides the necessary numerical operations.
     * @param N The total size of the problem.
     * @param unconverged The number of unconverged eigenpairs.
     * @param nex The number of external searching space.
     * @param upperb The upper bound for the spectrum.
     * @param lowerb The lower bound for the spectrum.
     * @param tol The tolerance for the degree calculation.
     * @param ritzv The array storing the Ritz values.
     * @param resid The array storing the residuals of the Ritz values.
     * @param residLast The array storing the residuals from the previous iteration.
     * @param degrees The array storing the degrees for each Ritz pair.
     * @param locked The number of locked eigenpairs.
     * 
     * @return The optimized polynomial degree.
     */
    static std::size_t calc_degrees_quasi(ChaseBase<T>* kernel, std::size_t N,
                                         std::size_t unconverged, std::size_t nex,
                                         Base<T> upperb, Base<T> lowerb, Base<T> tol,
                                         Base<T>* ritzv, Base<T>* resid,
                                         Base<T>* residLast, std::size_t* degrees,
                                         std::size_t locked);
    
    /**
     * @brief Detects eigenvalue clusters and computes spacing factors for degree optimization.
     * 
     * This helper method analyzes eigenvalue clustering and computes spacing factors 
     * to improve degree optimization for clustered eigenvalues.
     * 
     * @param ritzv The array storing the Ritz values.
     * @param resid The array storing the residuals of the Ritz values.
     * @param tol The convergence tolerance.
     * @param unconverged The number of unconverged eigenpairs.
     * @param nex The number of external searching space.
     * @param upperb The upper bound for the spectrum.
     * @param lowerb The lower bound for the spectrum.
     * @param cluster_factors Output array for computed spacing factors.
     */
    static void detect_eigenvalue_clusters(Base<T>* ritzv, Base<T>* resid, 
                                          Base<T> tol, std::size_t unconverged, 
                                          std::size_t nex, Base<T> upperb, 
                                          Base<T> lowerb, 
                                          std::vector<Base<T>>& cluster_factors);
    /**
     * @brief Sorts the Ritz values based on residuals and locks the converged ones.
     * 
     * This method sorts the Ritz values according to their residuals and locks the converged eigenpairs 
     * to prevent further processing.
     * 
     * @param kernel A pointer to the `ChaseBase<T>` instance.
     * @param N The total size of the problem.
     * @param unconverged The number of unconverged eigenpairs.
     * @param tol The convergence tolerance for residuals.
     * @param ritzv The array storing the Ritz values.
     * @param resid The array storing the residuals of the Ritz values.
     * @param residLast The array storing the residuals from the previous iteration.
     * @param degrees The array storing the degrees of the Ritz pairs.
     * @param locked The number of already locked eigenpairs.
     * 
     * @return The number of locked eigenpairs.
     */
    static std::size_t locking(ChaseBase<T>* kernel, std::size_t N,
                               std::size_t unconverged, Base<T> tol,
                               Base<T>* ritzv, Base<T>* resid,
                               Base<T>* residLast, std::size_t* degrees,
                               std::size_t locked);
    
    static std::size_t locking_quasi(
        ChaseBase<T>* single, std::size_t N,
        std::size_t unconverged, std::size_t nex, Base<T> tol, std::size_t* index,
        Base<T>* Lritzv, Base<T>* resid, Base<T>* residLast,
        std::size_t* degrees, std::size_t locked, std::size_t iteration);
            
    /**
     * @brief Implements the Chebyshev filter.
     * 
     * This method applies the Chebyshev filter for eigenvalue problems, typically used in conjunction 
     * with the other methods for solving large eigenvalue problems efficiently.
     * 
     * @param kernel A pointer to the `ChaseBase<T>` instance.
     * @param n The number of vectors.
     * @param unprocessed The number of unprocessed vectors.
     * @param deg The degree of the Chebyshev filter.
     * @param degrees An array containing the degrees of each Ritz pair.
     * @param lambda_1 The lower bound for the Chebyshev filter.
     * @param lower The lower bound of the spectrum.
     * @param upper The upper bound of the spectrum.
     * 
     * @return The number of processed vectors after applying the filter.
     */
    static std::size_t filter(ChaseBase<T>* kernel, std::size_t n,
                              std::size_t unprocessed, std::size_t deg,
                              std::size_t* degrees, Base<T> lambda_1,
                              Base<T> lower, Base<T> upper);
    /**
     * @brief Implements the Lanczos method to estimate bounds in ChASE.
     * 
     * This method uses the Lanczos algorithm to estimate the required spectrum bounds for ChASE.
     * 
     * @param kernel A pointer to the `ChaseBase<T>` instance.
     * @param N The total size of the problem.
     * @param numvec The number of vectors.
     * @param m The number of Lanczos iterations.
     * @param nevex The number of eigenpairs to be computed.
     * @param upperb The array storing the upper bounds of the spectrum.
     * @param mode A boolean indicating whether to run Lanczos in a specific mode.
     * @param ritzv_ The array storing the Ritz values.
     * 
     * @return The number of bounds estimated by Lanczos.
     */
    static std::size_t lanczos(ChaseBase<T>* kernel, int N, int numvec, int m,
                               int nevex, Base<T>* upperb, bool mode,
                               Base<T>* ritzv_);
};

/**
 * @brief Solves eigenproblems using the ChASE solver.
 * 
 * This is a wrapper function around `Algorithm<T>::solve()` to facilitate solving eigenvalue problems.
 * 
 * @param single A pointer to an object that implements the `ChaseBase<T>` interface, 
 * which contains the actual solver logic.
 */
template <typename T>
void Solve(ChaseBase<T>* single)
{
    Algorithm<T>::solve(single);
}

} // namespace chase

#include "algorithm.inc"

#endif
