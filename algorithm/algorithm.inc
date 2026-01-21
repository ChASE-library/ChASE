// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

namespace chase
{

template <class T>
void swap_kj(std::size_t k, std::size_t j, T* array)
{
    T tmp = array[k];
    array[k] = array[j];
    array[j] = tmp;
}

template <class T>
void Algorithm<T>::detect_eigenvalue_clusters(
    Base<T>* ritzv, Base<T>* resid, Base<T> tol, std::size_t unconverged,
    std::size_t nex, Base<T> upperb, Base<T> lowerb,
    std::vector<Base<T>>& cluster_factors)
{
    // Initialize all factors to 1.0 (no adjustment)
    cluster_factors.assign(unconverged - nex, 1.0);

    // === ENHANCED RESIDUAL-WEIGHTED CLUSTER DETECTION ===
    const Base<T> cluster_threshold =
        std::abs(upperb - lowerb) * 1e-6; // Spatial threshold
    const Base<T> min_spacing_factor =
        0.5; // Minimum spacing to prevent over-clustering
    const Base<T> max_spacing_factor =
        3.0; // Maximum spacing to prevent excessive degrees

    // Step 1: Compute residual-based difficulty weights
    std::vector<Base<T>> residual_weights(unconverged - nex);
    Base<T> mean_residual = 0.0;

    for (std::size_t i = 0; i < unconverged - nex; ++i)
    {
        mean_residual += resid[i];
    }
    mean_residual /= (unconverged - nex);

    for (std::size_t i = 0; i < unconverged - nex; ++i)
    {
        // Weight based on how much worse this residual is compared to average
        Base<T> relative_difficulty = resid[i] / (mean_residual + 1e-14);

        // Apply logarithmic scaling to prevent extreme weights
        residual_weights[i] = 1.0 + std::log(1.0 + relative_difficulty);

        // Cap the residual weight to prevent excessive amplification
        residual_weights[i] = std::min(residual_weights[i], Base<T>(2.5));
    }

    // Step 2: Analyze spatial clustering with residual weighting
    for (std::size_t i = 0; i < unconverged - nex; ++i)
    {
        Base<T> spatial_clustering_factor = 1.0;
        Base<T> convergence_difficulty_factor = residual_weights[i];

        // === SPATIAL CLUSTERING ANALYSIS ===
        Base<T> local_density = 0.0;
        std::size_t neighbors = 0;

        // Analyze local eigenvalue density around current eigenvalue
        for (std::size_t j = 0; j < unconverged - nex; ++j)
        {
            if (i != j)
            {
                Base<T> distance = std::abs(ritzv[i] - ritzv[j]);
                if (distance < cluster_threshold)
                {
                    // Weight neighboring influence by their convergence
                    // difficulty
                    Base<T> neighbor_weight = residual_weights[j];
                    local_density += neighbor_weight / (distance + 1e-14);
                    neighbors++;
                }
            }
        }

        // Calculate spatial clustering factor
        if (neighbors > 0)
        {
            spatial_clustering_factor =
                1.0 + std::log(1.0 + local_density * 0.1);
        }

        // === RESIDUAL-WEIGHTED COMBINATION ===
        // Combine spatial clustering with convergence difficulty
        Base<T> combined_factor =
            spatial_clustering_factor * convergence_difficulty_factor;

        // === ADDITIONAL HEURISTICS ===
        // Extra boost for eigenvalues that are both clustered AND struggling
        if (neighbors > 2 && resid[i] > 2.0 * mean_residual)
        {
            combined_factor *=
                1.2; // 20% additional boost for problematic clusters
        }

        // Boost for eigenvalues with stagnant convergence (high residual
        // relative to tolerance)
        if (resid[i] > 10.0 * tol)
        {
            combined_factor *=
                1.15; // 15% boost for eigenvalues far from convergence
        }

        // Apply bounds to prevent extreme values
        cluster_factors[i] = std::min(
            max_spacing_factor, std::max(min_spacing_factor, combined_factor));
    }

    // Step 3: Apply smoothing to avoid abrupt degree changes
    std::vector<Base<T>> temp_factors = cluster_factors;
    for (std::size_t i = 1; i < unconverged - nex - 1; ++i)
    {
        cluster_factors[i] = 0.25 * temp_factors[i - 1] +
                             0.5 * temp_factors[i] + 0.25 * temp_factors[i + 1];
    }

    // Step 4: Final validation and adjustment
    for (std::size_t i = 0; i < unconverged - nex; ++i)
    {
        // Ensure factors are within valid bounds after smoothing
        cluster_factors[i] =
            std::min(max_spacing_factor,
                     std::max(min_spacing_factor, cluster_factors[i]));
    }
}

template <class T>
std::size_t Algorithm<T>::calc_degrees(ChaseBase<T>* single, std::size_t N,
                                       std::size_t unconverged, std::size_t nex,
                                       Base<T> upperb, Base<T> lowerb,
                                       Base<T> tol, Base<T>* ritzv,
                                       Base<T>* resid, Base<T>* residLast,
                                       std::size_t* degrees, std::size_t locked)
{
    SCOPED_NVTX_RANGE();

    ChaseConfig<T> conf = single->GetConfig();

    Base<T> c = (upperb + lowerb) / 2; // Center of the interval.
    Base<T> e = (upperb - lowerb) / 2; // Half-length of the interval.
    Base<T> rho;

    for (std::size_t i = 0; i < unconverged - nex; ++i)
    {
        Base<T> t = (ritzv[i] - c) / e;
        rho = std::max(std::abs(t - std::sqrt(std::abs(t * t - 1))),
                       std::abs(t + std::sqrt(std::abs(t * t - 1))));

        degrees[i] =
            std::ceil(std::abs(std::log(resid[i] / tol) / std::log(rho)));

        // Ensure minimum degree of 8 for single precision
        if constexpr (std::is_same<Base<T>, float>::value)
        {
            degrees[i] = std::max(degrees[i], std::size_t(8));
        }

        degrees[i] =
            std::min(degrees[i] + conf.GetDegExtra(), conf.GetMaxDeg());
    }

    for (std::size_t i = unconverged - nex; i < unconverged; ++i)
    {
        degrees[i] = degrees[unconverged - 1 - nex];
    }

    for (std::size_t i = 0; i < unconverged; ++i)
    {
        degrees[i] += degrees[i] % 2;
    }

    // we sort according to degrees
    for (std::size_t j = 0; j < unconverged - 1; ++j)
        for (std::size_t k = j; k < unconverged; ++k)
            if (degrees[k] < degrees[j])
            {
                swap_kj(k, j, degrees); // for filter
                swap_kj(k, j, ritzv);
                swap_kj(k, j, resid);
                swap_kj(k, j, residLast);
                single->Swap(k + locked, j + locked);
            }

    return degrees[unconverged - 1];
}

template <class T>
std::size_t Algorithm<T>::calc_degrees_pseudo(
    ChaseBase<T>* single, std::size_t N, std::size_t unconverged,
    std::size_t nex, Base<T> upperb, Base<T> lowerb, Base<T> tol,
    Base<T>* ritzv, Base<T>* resid, Base<T>* residLast, std::size_t* degrees,
    std::size_t locked)
{
    SCOPED_NVTX_RANGE();

    ChaseConfig<T> conf = single->GetConfig();

    Base<T> c = (upperb + lowerb) / 2; // Center of the interval.
    Base<T> e = (upperb - lowerb) / 2; // Half-length of the interval.
    Base<T> rho;

    // === CLUSTER DETECTION AND SPACING FACTORS (PSEUDO-HERMITIAN SPECIFIC) ===
    // When cluster-aware degrees are disabled, fall back to original algorithm
    // exactly
    if (!conf.UseClusterAwareDegrees())
    {
        // Use original calc_degrees implementation exactly
        for (std::size_t i = 0; i < unconverged - nex; ++i)
        {
            Base<T> t = (ritzv[i] - c) / e;
            rho = std::max(std::abs(t - std::sqrt(std::abs(t * t - 1))),
                           std::abs(t + std::sqrt(std::abs(t * t - 1))));

            degrees[i] =
                std::ceil(std::abs(std::log(resid[i] / tol) / std::log(rho)));

            // Ensure minimum degree of 8 for single precision
            if constexpr (std::is_same<Base<T>, float>::value)
            {
                degrees[i] = std::max(degrees[i], std::size_t(8));
            }

            degrees[i] =
                std::min(degrees[i] + conf.GetDegExtra(), conf.GetMaxDeg());
        }

        // Original nex region handling
        for (std::size_t i = unconverged - nex; i < unconverged; ++i)
        {
            degrees[i] = degrees[unconverged - 1 - nex];
        }

        // Original even degree enforcement
        for (std::size_t i = 0; i < unconverged; ++i)
        {
            degrees[i] += degrees[i] % 2;
        }

        // Original sorting
        for (std::size_t j = 0; j < unconverged - 1; ++j)
            for (std::size_t k = j; k < unconverged; ++k)
                if (degrees[k] < degrees[j])
                {
                    swap_kj(k, j, degrees);
                    swap_kj(k, j, ritzv);
                    swap_kj(k, j, resid);
                    // swap_kj(k, j, residLast);
                    single->Swap(k + locked, j + locked);
                }

        return degrees[unconverged - 1];
    }

    // If we reach here, cluster-aware degrees are enabled
    std::vector<Base<T>> cluster_factors;
    detect_eigenvalue_clusters(ritzv, resid, tol, unconverged, nex, upperb,
                               lowerb, cluster_factors);

    // === PSEUDO-HERMITIAN DEGREE CALCULATION WITH CLUSTER AWARENESS ===
    for (std::size_t i = 0; i < unconverged - nex; ++i)
    {
        Base<T> t = (ritzv[i] - c) / e;
        rho = std::max(std::abs(t - std::sqrt(std::abs(t * t - 1))),
                       std::abs(t + std::sqrt(std::abs(t * t - 1))));

        // Original degree calculation
        Base<T> base_degree =
            std::ceil(std::abs(std::log(resid[i] / tol) / std::log(rho)));

        // Apply cluster-aware spacing factor
        degrees[i] = std::size_t(base_degree * cluster_factors[i]);

        // Ensure minimum degree of 8 for single precision
        if constexpr (std::is_same<Base<T>, float>::value)
        {
            degrees[i] = std::max(degrees[i], std::size_t(8));
        }

        // === STAGNATION DETECTION ===
        std::size_t stagnation_bonus = 0.0;
        Base<T> stagnation_residual_threshold = tol * 10.0; // 10x tolerance

        if (resid[i] <= stagnation_residual_threshold)
        {
            Base<T> residual_change = std::abs(resid[i] - residLast[i]);
            Base<T> relative_change = residual_change / (resid[i] + 1e-14);
            Base<T> relative_stagnation = 0.1; // 10% relative change
            if (relative_change < relative_stagnation)
            {
                // Residual is stagnating - increase degrees
                stagnation_bonus = 6;
            }
            degrees[i] = std::size_t(degrees[i] + stagnation_bonus);
        }

        // Apply additional pseudo-hermitian specific adjustments
        // For pseudo-hermitian matrices, we might need more aggressive degrees
        std::size_t pseudo_bonus = 0;
        if (std::abs(ritzv[i]) < std::abs(upperb - lowerb) * 0.1)
        {
            // Near-zero eigenvalues in pseudo-hermitian problems can be
            // challenging
            pseudo_bonus = 2;
        }

        degrees[i] = std::min(degrees[i] + conf.GetDegExtra() + pseudo_bonus,
                              conf.GetMaxDeg());

#ifdef CHASE_OUTPUT

#ifndef NDEBUG
        // Debug output for residual-weighted cluster analysis

        if (cluster_factors[i] > 1.5 || cluster_factors[i] < 0.8 ||
            pseudo_bonus > 0)
        {
            std::ostringstream oss;
            oss << "Residual-weighted cluster adjustment: eigenvalue[" << i
                << "] = " << ritzv[i] << ", residual = " << resid[i]
                << ", cluster_factor = " << cluster_factors[i]
                << ", stagnation_bonus = " << stagnation_bonus
                << ", pseudo_bonus = " << pseudo_bonus
                << ", base_degree = " << base_degree
                << ", final_degree = " << degrees[i] << "\n";
            single->Output(oss.str());
        }
#endif

#endif
    }

    // === ENHANCED NEX REGION HANDLING FOR PSEUDO-HERMITIAN ===
    for (std::size_t i = unconverged - nex; i < unconverged; ++i)
    {
        // For pseudo-hermitian, use weighted average to avoid ghost point
        // contamination
        if (unconverged - nex > 0)
        {
            std::size_t ref_idx =
                std::min(unconverged - nex - 1,
                         std::max(std::size_t(0), unconverged - nex - 3));
            Base<T> avg_degree = 0.0;
            std::size_t count = 0;

            for (std::size_t j = ref_idx; j < unconverged - nex; ++j)
            {
                avg_degree += degrees[j];
                count++;
            }

            if (count > 0)
            {
                degrees[i] = std::size_t(avg_degree / count);
                // Add extra degree for nex region in pseudo-hermitian case
                degrees[i] = std::min(degrees[i] + 2, conf.GetMaxDeg());
            }
            else
            {
                degrees[i] = degrees[unconverged - 1 - nex];
            }
        }
        else
        {
            degrees[i] = conf.GetDeg();
        }
    }

    // === ENSURE ALL DEGREES ARE EVEN (CRITICAL FOR CHEBYSHEV STABILITY) ===
    for (std::size_t i = 0; i < unconverged; ++i)
    {
        degrees[i] += degrees[i] % 2; // Make even: add 1 if odd, add 0 if even
    }

    // === PSEUDO-HERMITIAN SPECIFIC SORTING ===
    // Sort according to degrees for optimal filtering order
    for (std::size_t j = 0; j < unconverged - 1; ++j)
        for (std::size_t k = j; k < unconverged; ++k)
            if (degrees[k] < degrees[j])
            {
                swap_kj(k, j, degrees); // for filter
                swap_kj(k, j, ritzv);
                swap_kj(k, j, resid);
                if (single->isSym())
                {
                    swap_kj(k, j, residLast);
                }
                single->Swap(k + locked, j + locked);
            }

    return degrees[unconverged - 1];
}

template <class T>
std::size_t Algorithm<T>::locking(ChaseBase<T>* single, std::size_t N,
                                  std::size_t unconverged, Base<T> tol,
                                  Base<T>* Lritzv, Base<T>* resid,
                                  Base<T>* residLast,
                                  std::vector<Base<T>>* early_locked_residuals,
                                  std::size_t* degrees, std::size_t locked)
{
    SCOPED_NVTX_RANGE();

    // we build the permutation
    std::vector<int> index(unconverged, 0);
    for (int i = 0; i != index.size(); i++)
    {
        index[i] = i;
    }
    sort(index.begin(), index.end(),
         [&](const int& a, const int& b) { return (Lritzv[a] < Lritzv[b]); });
    std::size_t converged = 0;

    for (auto k = 0; k < unconverged; k++)
    {
        auto j = index[k]; // walk through

#ifndef CHASE_SAVE_RESIDUALS
        if (resid[j] <= tol || (single->isSym() && resid[j] >= residLast[j] &&
                                resid[j] < 100.0 * tol))
#else
        if (resid[j] <= tol)
#endif
        {

#ifndef CHASE_SAVE_RESIDUALS
            if (single->isSym() && resid[j] > tol && resid[j] >= residLast[j] &&
                resid[j] < 100.0 * tol)
            {

#ifdef CHASE_OUTPUT
                std::ostringstream oss;
                oss << "locking unconverged pair is:" << resid[j]
                    << " was:" << residLast[j] << " tolerance is: " << tol
                    << " val: " << Lritzv[j] << "\n";
                single->Output(oss.str());
#endif
                early_locked_residuals->push_back(resid[j]);
            }
#endif
            if (j != converged)
            {
                swap_kj(j, converged, resid);     // if we filter again
                swap_kj(j, converged, residLast); // if we filter again
                swap_kj(j, converged, Lritzv);
                single->Swap(j + locked, converged + locked);
            }

            converged++;
        }
    }

    return converged;
}

template <class T>
std::size_t Algorithm<T>::locking_pseudo(
    ChaseBase<T>* single, std::size_t N, std::size_t unconverged,
    std::size_t nex, Base<T> tol, std::size_t* index, Base<T>* Lritzv,
    Base<T>* resid, Base<T>* residLast,
    std::vector<Base<T>>* early_locked_residuals, std::size_t* degrees,
    std::size_t locked, std::size_t iteration)
{
    SCOPED_NVTX_RANGE();

    std::vector<std::size_t> index_unconverged;

    std::vector<Base<T>> residLast_unconverged(unconverged);
    std::copy_n(resid, unconverged, residLast_unconverged.begin());

    // 3. Locking step
    std::size_t converged = 0;
    for (std::size_t k = 0; k < unconverged - nex; ++k)
    {
        int j = index[k];

#ifndef CHASE_SAVE_RESIDUALS
        if (resid[j] <= tol || (resid[j] > tol && resid[j] >= residLast[k] &&
                                resid[j] <= 1000.0 * tol && iteration >= 4))
#else
        if (resid[j] <= tol)
#endif
        {

#ifndef CHASE_SAVE_RESIDUALS
            if (resid[j] > tol && resid[j] <= 1000.0 * tol &&
                resid[j] >= residLast[k] && iteration >= 4)
            {
#ifdef CHASE_OUTPUT
                std::ostringstream oss;
                oss << "locking unconverged pair is:" << resid[j]
                    << " was:" << residLast[k] << " tolerance is: " << tol
                    << " val: " << Lritzv[j] << "\n";
                single->Output(oss.str());
#endif
                early_locked_residuals->push_back(resid[j]);
            }
#endif
            if (j != converged)
            {
                swap_kj(j, converged, resid);
                swap_kj(j, converged, Lritzv);
                swap_kj(j, converged, degrees);
                single->Swap(j + locked, converged + locked);
            }

            ++converged;
        }
        else
        {
            index_unconverged.push_back(index[k]);
        }
    }

    for (std::size_t k = unconverged - nex; k < unconverged; k++)
    {
        index_unconverged.push_back(index[k]);
    }
    // We need to update the index set for the eventual calculation of the new
    // lowerb, because the all the sets and vectors will be shifted by
    // converged.
    std::copy_n(index_unconverged.begin(), unconverged - converged, index);

    for (std::size_t k = converged; k < unconverged; k++)
    {
        residLast[k] = residLast_unconverged[index_unconverged[k - converged]];
        // std::cout << "k = " << k << " residLast = " << residLast[k] <<
        // std::endl;
    }

    return converged;
}

template <class T>
std::size_t Algorithm<T>::filter(ChaseBase<T>* single, std::size_t n,
                                 std::size_t unprocessed, std::size_t deg,
                                 std::size_t* degrees, Base<T> lambda_1,
                                 Base<T> lower, Base<T> upper)
{
    SCOPED_NVTX_RANGE();

    Base<T> c = (upper + lower) / 2;
    Base<T> e = (upper - lower) / 2;
    Base<T> sigma_1 = e / (lambda_1 - c);
    Base<T> sigma = sigma_1;
    Base<T> sigma_new;

    std::size_t offset = 0;
    std::size_t num_mult = 0;
    std::size_t Av = 0;

    //----------------------------------- A = A-cI
    //-------------------------------
    single->Shift(-c);
    //------------------------------- Y = alpha*(A-cI)*V
    //-------------------------
    T alpha = T(sigma_1 / e);
    T beta = T(0.0);

    single->HEMM(unprocessed, alpha, beta, offset / n);

    Av += unprocessed;
    num_mult++;

    // this is really not possible, since the minimum degree is 3
    while (unprocessed >= 0 && *degrees <= num_mult)
    {
        degrees++; // V+=n; W+=n;
        unprocessed--;
        offset += n;
    };

    for (std::size_t i = 2; i <= deg; ++i)
    {
        sigma_new = 1.0 / (2.0 / sigma_1 - sigma);

        //----------------------- V = alpha(A-cI)W + beta*V
        //----------------------
        alpha = T(2.0 * sigma_new / e);
        beta = T(-sigma * sigma_new);

        single->HEMM(unprocessed, alpha, beta, offset / n);

        sigma = sigma_new;
        Av += unprocessed;
        num_mult++;
        while (unprocessed != 0 && *degrees <= num_mult)
        {
            degrees++; // V+=n; W+=n;
            unprocessed--;
            offset += n;
        }

    } // for(i = 2; i <= deg; ++i)

    //----------------------------------RESTORE-A---------------------------------
    single->Shift(+c, true);

    return Av;
}

template <class T>
std::size_t Algorithm<T>::lanczos(ChaseBase<T>* single, int N, int numvec,
                                  int m, int nevex, Base<T>* upperb, bool mode,
                                  Base<T>* ritzv_)
{
    SCOPED_NVTX_RANGE();

    assert(m >= 1);

    if (!mode)
    {
        // all we need is the upper bound
        /* for( auto i=0; i < N; ++i) */
        /*   V_[i] = T( d(gen), d(gen) ); */
        single->Lanczos(m, upperb);
        return 0;
    }

    // we need a bound for lambda1.

    // We will do numvec many Lanczos procedures and save all the eigenvalues,
    // and the first entrieXs of the eigenvectors
    Base<T>* Theta = new Base<T>[numvec * m]();
    Base<T>* Tau = new Base<T>[numvec * m]();

    Base<T>* ritzV = new Base<T>[m * m]();
    Base<T> lowerb, lambda;

    single->Lanczos(m, numvec, upperb, Theta, Tau, ritzV);

    double* ThetaSorted = new double[numvec * m];
    for (auto k = 0; k < numvec * m; ++k)
        ThetaSorted[k] = Theta[k];
    std::sort(ThetaSorted, ThetaSorted + numvec * m, std::less<double>());
    lambda = ThetaSorted[0];

    double curr, prev = 0;
    const double sigma = 0.25;
    const double threshold = 2 * sigma * sigma / 10;
    const double search = static_cast<double>(nevex) / static_cast<double>(N);
    // CDF of a Gaussian, erf is a c++11 function
    const auto G = [&](double x) -> double
    { return 0.5 * (1 + std::erf(x / sqrt(2 * sigma * sigma))); };

    std::size_t bound = m;
    if (single->isPseudoHerm())
    {
        // The spectrum of a Pseudo-Hermitian matrix is symmetric over the
        // y-axis.
        bound /= 2;
    }

    for (auto i = 0; i < numvec * bound - 1; ++i)
    {
        curr = 0;
        for (int j = 0; j < numvec * bound; ++j)
        {
            if (ThetaSorted[i] < (Theta[j] - threshold))
                curr += 0;
            else if (ThetaSorted[i] > (Theta[j] + threshold))
                curr += Tau[j] * 1;
            else
                curr += Tau[j] * G(ThetaSorted[i] - Theta[j]);
        }
        curr = curr / numvec;

        if (curr > search)
        {
            // CHANGED: safely select next Theta or fallback
            if (std::abs(curr - search) < std::abs(prev - search))
            {
                if (i + 1 < numvec * bound)
                    lowerb = ThetaSorted[i + 1]; // CHANGED
                else
                    lowerb = ThetaSorted[i]; // CHANGED fallback
            }
            else
            {
                lowerb = ThetaSorted[i]; // CHANGED
            }
            break;
        }
        prev = curr;
    }

#ifdef CHASE_OUTPUT
    {
#ifndef NDEBUG
        std::ostringstream oss;

        oss << "m = " << m << std::endl;
        oss << "numvec = " << numvec << std::endl;
        oss << "bound  = " << bound << std::endl;
        oss << "lowerb = " << lowerb << std::endl;
        oss << "search = " << search << std::endl;

        single->Output(oss.str());
#endif
    }
#endif

    // Now we extract the Eigenvectors that correspond to eigenvalues < lowerb
    int idx = 0;
    for (int i = 0; i < m; ++i)
    {
        if (Theta[(numvec - 1) * m + i] > lowerb)
        {
            idx = i - 1;
            break;
        }
    }

#ifdef CHASE_OUTPUT
    {
        std::ostringstream oss;
        oss << "extracted " << idx << " vectors from DoS\n";
        single->Output(oss.str());
    }
#endif

    if (idx > 0)
    {
        // cast to (generally complex T)
        T* ritzVc = new T[m * m]();
        for (auto i = 0; i < m * m; ++i)
            ritzVc[i] = T(ritzV[i]);
        single->LanczosDos(idx, m, ritzVc);

        delete[] ritzVc;
    }

    // lowerb = lowerb + std::abs(lowerb)*0.25;
    for (auto i = 0; i < idx; ++i)
    {
        ritzv_[i] = Theta[(numvec - 1) * m + i];
    }
    for (auto i = idx; i < nevex - 1; ++i)
    {
        ritzv_[i] = lambda;
    }
    ritzv_[nevex - 1] = lowerb;

    // intersperse lanczos vectors
    for (auto i = 1; i < idx; ++i)
    {
        auto j = i * (nevex / idx);
        single->Swap(i, j);
        std::swap(ritzv_[i], ritzv_[j]);
    }

    // Cleanup
    delete[] ThetaSorted;
    delete[] Theta;
    delete[] Tau;
    delete[] ritzV;
    //    delete[] V;
    return idx;
}

template <class T>
void Algorithm<T>::solve(ChaseBase<T>* single)
{
    ChaseConfig<T>& config = single->GetConfig();
    /*
        if(config.DoSymCheck())
        {
            if(single->checkSymmetryEasy())
            {
    #ifdef CHASE_OUTPUT
                {
                    std::ostringstream oss;

                    oss << "The input matrix is explicitly symmetric/Hermtian,
    ChASE can continue to proceed\n";

                    single->Output(oss.str());
                }
    #endif
            }else
            {
    #ifdef CHASE_OUTPUT
                {
                    std::ostringstream oss;

                    oss << "!!! The input matrix is NOT explicitly
    symmetric/Hermtian, ChASE cannot continue to proceed\n"; oss << "Plase
    consider to use function symmetrizeOrHermitianizeMatrix() to
    symmetrize/Hermitianize at first\n";

                    single->Output(oss.str());
                }
    #endif
                return;
            }
        }
    */
    single->Start();

    std::size_t N = config.GetN();
    std::size_t nev = config.GetNev();
    const std::size_t nex = config.GetNex();
    const std::size_t num_lanczos = config.GetNumLanczos();

    Base<T>* resid_ = single->GetResid();
    Base<T>* ritzv_ = single->GetRitzv();

    const double tol = config.GetTol();

    const std::size_t nevex = nev + nex;
    std::size_t unconverged = nev + nex;

    // To store the approximations obtained from lanczos().
    Base<T> lowerb, upperb, lambda;

    Base<T> cond;

    std::vector<std::size_t> degrees_(nev + nex);
    std::vector<Base<T>> residLast_(nevex);
    std::vector<Base<T>> early_locked_residuals;
    // this will be copied into residLast
    for (auto i = 0; i < nevex; ++i)
    {
        residLast_[i] = std::numeric_limits<Base<T>>::max();
        resid_[i] = std::numeric_limits<Base<T>>::max();
    }

    // store input values
    std::size_t deg = config.GetDeg();
    deg += deg % 2;
    std::size_t* degrees = degrees_.data();
    Base<T>* ritzv = ritzv_;
    Base<T>* resid = resid_;
    Base<T>* residLast = residLast_.data();

    //-------------------------------- VALIDATION
    //--------------------------------
    assert(degrees != NULL);
    deg = std::min(deg, config.GetMaxDeg());
    for (std::size_t i = 0; i < nevex; ++i)
        degrees[i] = deg;
    bool random = !config.UseApprox();
    single->initVecs(random);
    if (random)
    {
        single->QR(0, 1.0);
    }
    // --------------------------------- LANCZOS
    // ---------------------------------

    std::size_t lanczos_iter =
        std::min(nevex, std::min(N / 2, config.GetLanczosIter()));
    if (single->isPseudoHerm() && 2.0 * (lanczos_iter / 2) < lanczos_iter)
    {
        config.SetLanczosIter(lanczos_iter - 1);
        lanczos_iter = config.GetLanczosIter();
        /* We enforce the number of ritz values to be even since the spectrum is
         * symmetric in the Pseudo-Hermitian case */
    }
    lanczos(single, N, num_lanczos, lanczos_iter, nevex, &upperb, random,
            random ? ritzv : NULL);

    std::size_t locked = 0; // Number of converged eigenpairs.
    std::size_t locked_prev =
        0; // Number of converged eigenpairs of the previous iteration
    std::size_t iteration = 0; // Current iteration.
    lowerb = *std::max_element(ritzv, ritzv + unconverged);
    lambda = *std::min_element(ritzv_, ritzv_ + nevex);
    Base<T> lambda_temp, lowerb_temp;
    lowerb = lowerb * config.GetDecayingRate();
    if (single->isPseudoHerm())
    {
        if (upperb > 0)
        {
            upperb = upperb * config.GetUpperbScaleRate();
        }
        else
        {
            upperb = upperb / config.GetUpperbScaleRate();
        }
    }
    // upperb = -lambda;
    std::size_t new_converged;

    std::vector<std::size_t> index(unconverged); // unconverged = nev+nex-locked
    std::vector<std::size_t> index_res(
        unconverged); // unconverged = nev+nex-locked
    std::vector<std::size_t> index_temp(unconverged);

    std::iota(index.begin(), index.end(), 0);

#ifdef CHASE_SAVE_RESIDUALS

    std::ofstream residual_file;
    std::string residual_file_name;
    std::vector<std::size_t> index_write(unconverged);

    if (!single->get_rank())
    {
        residual_file_name = "residual_file_" + std::to_string(N);

        if (single->isPseudoHerm())
        {
            residual_file_name = residual_file_name + "_pseudo";
        }

#ifdef XGEEV_EXISTS
        residual_file_name = residual_file_name + "_geev";
#endif
        residual_file_name = residual_file_name + ".csv";
        residual_file.open(residual_file_name);
        residual_file << "iteration" << "," << "residual" << "\n";
    }
#endif

    while (unconverged > nex && iteration < config.GetMaxIter())
    {
        int cnt;
        for (cnt = 0; cnt < unconverged; cnt++)
        {
            if (resid[cnt] > 5e-1)
            {
                break;
            }
        }
        if (single->isSym() && cnt == unconverged)
        {
            lowerb = ritzv[unconverged - 1];
        }

        /*If no locking we assume that the unconverged ritz provides a descent
          lowerb. We keep it until convergence once we start locking, except if
          one iteration does not provide convergence for at least one
          eigenvector.*/

        if (single->isPseudoHerm() && iteration > 0)
        {

            lowerb_temp =
                ritzv[index.data()[std::size_t(unconverged * 0.95) - 1]] *
                config.GetDecayingRate();
            if (lowerb_temp > lambda)
            {
                lowerb = std::min(lowerb, lowerb_temp);
            } /*
             if(iteration > 3)
                 {
                         lambda = std::max(lambda,lambda_temp);
                 }	*/
        }

#ifdef CHASE_OUTPUT
        {
            std::ostringstream oss;

            oss << std::scientific << "iteration: " << iteration << "\t"
                << std::setprecision(6) << lambda << "\t"
                << std::setprecision(6) << lowerb << "\t"
                << std::setprecision(6) << upperb << "\t" << unconverged
                << std::endl;

            single->Output(oss.str());
        }
#endif
        //    assert( lowerb < upperb );
        if (lowerb > upperb)
        {
            std::cout << "ASSERTION FAILURE lowerb > upperb\n";
            lowerb = upperb;
        }

        if (single->isSym())
        {
            for (auto i = 0; i < unconverged; ++i)
                residLast[i] = std::min(residLast[i], resid[i]);
        }

        //-------------------------------- DEGREES
        //--------------------------------
        if (config.DoOptimization() && iteration != 0)
        {
            if (single->isPseudoHerm())
            {
                // Use specialized pseudo-hermitian degree calculation with
                // cluster-aware features
                deg = calc_degrees_pseudo(single, N, unconverged, nex, upperb,
                                          lowerb, tol, ritzv, resid, residLast,
                                          degrees, locked);
            }
            else
            {
                deg =
                    calc_degrees(single, N, unconverged, nex, upperb, lowerb,
                                 tol, ritzv, resid, residLast, degrees, locked);
            }
        }

        //------------------------------- FILTER -------------------------------
        filter(single, N, unconverged, deg, degrees, lambda, lowerb, upperb);

        //----------------------------------- QR
        //-----------------------------------
        Base<T> cc = (upperb + lowerb) / 2; // Center of the interval.
        Base<T> ee = (upperb - lowerb) / 2; // Half-length of the interval.
        Base<T> rho_1, rho_k;
        Base<T> t_1, t_k;
        t_1 = (single->GetRitzv()[0] - cc) / ee;
        t_k = (ritzv[0] - cc) / ee;
        rho_1 = std::max(std::abs(t_1 - std::sqrt(t_1 * t_1 - 1)),
                         std::abs(t_1 + std::sqrt(t_1 * t_1 - 1)));
        rho_k = std::max(std::abs(t_k - std::sqrt(t_k * t_k - 1)),
                         std::abs(t_k + std::sqrt(t_k * t_k - 1)));

        cond =
            std::pow(rho_k, degrees[0]) *
            std::pow(rho_1,
                     (*std::max_element(degrees, degrees + nev + nex - locked) -
                      degrees[0]));

        single->QR(locked, cond);

        // ----------------------------- RAYLEIGH  RITZ
        // ----------------------------
        single->RR(ritzv, unconverged);

        // --------------------------- RESIDUAL & LOCKING
        // --------------------------
        single->Resd(ritzv, resid, locked);

        if (!single->isSym())
        {
            // Build index vector for all unconverged eigenpairs
            std::iota(index_res.begin(), index_res.begin() + unconverged, 0);

            // Build index vector for all unconverged eigenpairs
            std::iota(index.begin(), index.begin() + unconverged, 0);

            // 2. Sort first nev by |Lritzv| (ascending)
            std::sort(index_res.begin(),
                      index_res.begin() +
                          unconverged, // sort ritzv values only 1st nev-locked
                      [&](int a, int b) { return (resid[a] < resid[b]); });

            if (locked == 0)
            {
                std::sort(
                    index.begin(),
                    index.begin() +
                        std::size_t(0.95 * unconverged), // sort ritzv values
                                                         // only 1st nev-locked
                    [&](int a, int b)
                    {
                        return (ritzv[index_res.data()[a]] <
                                ritzv[index_res.data()[b]]);
                    });
            }
            else
            {
                std::sort(
                    index.begin(),
                    index.begin() +
                        std::size_t(0.95 * unconverged), // sort ritzv values
                                                         // only 1st nev-locked
                    [&](int a, int b)
                    {
                        return (ritzv[index_res.data()[a]] <
                                ritzv[index_res.data()[b]]);
                    }); // MIGHT BE WRONG HERE ACTUALLY. KEEP 0.95?
            }

            for (std::size_t k = 0; k < unconverged; k++)
            {
                index_temp.data()[k] = index_res.data()[index.data()[k]];
            }

            std::swap(index_temp, index);
        }

#ifdef CHASE_OUTPUT
        {
            std::ostringstream oss;
            oss << "unconverged = " << "\t" << unconverged << "\t"
                << "new_converged" << "\t" << new_converged << "\n";
            oss << "index\tdegrees\tresid\tresidLast\tritzv\n";

#ifndef NDEBUG
            for (std::size_t k = 0; k < unconverged; ++k)
#else
            for (std::size_t k = 0; k < std::min(std::size_t(20), unconverged);
                 ++k)
#endif
                oss << index.data()[k] << "\t" << (degrees)[index.data()[k]]
                    << "\t" << (resid)[index.data()[k]] << "\t"
                    << (residLast)[k] << "\t" << (ritzv)[index.data()[k]]
                    << "\n";

            single->Output(oss.str());
        }
#endif

#ifdef CHASE_SAVE_RESIDUALS

#ifdef CHASE_OUTPUT
        {
            std::ostringstream oss;

            oss << "Saving the residuals into" << residual_file_name
                << std::endl;
            single->Output(oss.str());
        }
#endif
        if (!single->get_rank())
        {
            for (std::size_t idx = 0; idx < nev + nex - unconverged; idx++)
            {
                residual_file << iteration << "," << -1.0 << "\n";
            }
            // Build index vector for all unconverged eigenpairs
            std::iota(index_write.begin(), index_write.begin() + unconverged,
                      0);

            std::sort(index_write.begin(), index_write.begin() + unconverged,
                      [&](int a, int b) { return (ritzv[a] < ritzv[b]); });

            for (std::size_t idx = 0; idx < unconverged; idx++)
            {
                residual_file << iteration << ","
                              << resid[index_write.data()[idx]] << "\n";
            }
        }

#endif

        if (single->isSym())
        {
            new_converged = locking(
                single, N, unconverged - single->isSym() * nex, tol, ritzv,
                resid, residLast, &early_locked_residuals, degrees, locked);
        }
        else
        {
            new_converged = locking_pseudo(
                single, N, unconverged, nex, tol, index.data(), ritzv, resid,
                residLast, &early_locked_residuals, degrees, locked, iteration);
        }

        // ---------------------------- Update pointers
        // ---------------------------- Since we double buffer we need the
        // entire locked portion in W and V
        single->Lock(new_converged);

        lambda_temp = ritzv[index.data()[0]];

        locked_prev = locked;

        locked += new_converged;
        unconverged -= new_converged;

        resid += new_converged;
        residLast += new_converged;
        ritzv += new_converged;
        degrees += new_converged;

        iteration++;
    } // while ( converged < nev && iteration < omp_maxiter )

    //---------------------SORT-EIGENPAIRS-ACCORDING-TO-EIGENVALUES---------------
    // Create a permutation vector to track the desired final positions of
    // elements
    std::vector<std::size_t> perm(nev);
    // Initialize permutation with identity mapping (0 to n-1)
    std::iota(perm.begin(), perm.end(), 0);
    // Sort the permutation indices based on eigenvalue ordering
    // This creates a mapping from current positions to desired positions
    std::sort(perm.begin(), perm.end(), [&](std::size_t i, std::size_t j)
              { return ritzv_[i] < ritzv_[j]; });

    // Track which elements have been processed to avoid infinite loops
    std::vector<bool> visited(perm.size(), false);

    // Process each element in the permutation
    for (std::size_t i = 0; i < perm.size(); ++i)
    {
        if (visited[i] || perm[i] == i)
            continue;

        // Cycle decomposition start
        std::size_t current = i;
        chase::Base<T> temp_ritz = ritzv_[i];
        chase::Base<T> temp_resid = resid_[i];

        std::vector<std::size_t> cycle_indices;

        // Collect cycle indices
        while (!visited[current])
        {
            visited[current] = true;
            cycle_indices.push_back(current);
            current = perm[current];
        }

        // Rotate ritzv_ and resid_ scalars along cycle
        for (size_t k = 0; k < cycle_indices.size() - 1; ++k)
        {
            ritzv_[cycle_indices[k]] = ritzv_[cycle_indices[k + 1]];
            resid_[cycle_indices[k]] = resid_[cycle_indices[k + 1]];
        }
        ritzv_[cycle_indices.back()] = temp_ritz;
        resid_[cycle_indices.back()] = temp_resid;

        // Rotate eigenvectors along cycle by swapping adjacent pairs
        // For a cycle of indices like [a, b, c, d], do swaps:
        // Swap(a,b), Swap(b,c), Swap(c,d)
        for (size_t k = 0; k < cycle_indices.size() - 1; ++k)
        {
            single->Swap(cycle_indices[k], cycle_indices[k + 1]);
        }
    }

    single->set_early_locked_residuals(early_locked_residuals);

#ifdef CHASE_SAVE_RESIDUALS

    if (!single->get_rank())
    {
        residual_file.close();
    }

#endif

    single->End();
}
} // namespace chase
