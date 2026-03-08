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
                //swap_kj(k, j, residLast);
                single->Swap(k + locked, j + locked);
            }

    return degrees[unconverged - 1];
}

template <class T>
std::size_t Algorithm<T>::calc_degrees_pseudo_H2(
    ChaseBase<T>* single, std::size_t N, std::size_t unconverged,
    std::size_t nex, Base<T> upperb, Base<T> lowerb, Base<T> tol,
    Base<T>* ritzv, Base<T>* resid, Base<T>* residLast, std::size_t* degrees,
    std::size_t locked)
{
    SCOPED_NVTX_RANGE();

    ChaseConfig<T> conf = single->GetConfig();
    const std::size_t deg_extra = conf.GetDegExtra();
    const std::size_t deg_max = conf.GetMaxDeg();

    std::vector<Base<T>> cluster_factors;
    if (conf.UseClusterAwareDegrees())
    {
        detect_eigenvalue_clusters(ritzv, resid, tol, unconverged, nex, upperb, lowerb, cluster_factors);
    }

    // H² spectrum interval: center and half-length (for Chebyshev ellipse)
    Base<T> c_h2 = (upperb + lowerb) / 2;
    Base<T> e_h2 = (upperb - lowerb) / 2;

    if (e_h2 <= Base<T>(0))
    {
        for (std::size_t i = 0; i < unconverged; ++i)
            degrees[i] = deg_max + (deg_max % 2);
        return deg_max + (deg_max % 2);
    }

    // ritzv, resid, residLast, degrees are already shifted (no locked values).
    for (std::size_t i = 0; i < unconverged; ++i)
    {
        Base<T> lambda_ritz = ritzv[i];           // H Ritz value (real)
        Base<T> lambda_sq = lambda_ritz * lambda_ritz; // H² eigenvalue
        Base<T> residual_i = resid[i];           // H residual

        Base<T> t = (lambda_sq - c_h2) / e_h2;
        // rho from Chebyshev ellipse (use complex sqrt when t*t - 1 < 0)
        std::complex<Base<T>> z(t * t - 1, 0);
        Base<T> rho = std::max(
            std::abs(std::complex<Base<T>>(t, 0) - std::sqrt(z)),
            std::abs(std::complex<Base<T>>(t, 0) + std::sqrt(z)));

        std::size_t deg;
        if (!std::isfinite(rho) || rho <= Base<T>(1))
        {
            deg = deg_max;
        }
        else
        {
            Base<T> log_ratio = std::log(residual_i / tol) / std::log(rho);
            if (!std::isfinite(log_ratio))
                deg = deg_max;
            else
            {
                deg = static_cast<std::size_t>(
                    std::ceil(std::abs(static_cast<double>(log_ratio))));

                if (conf.UseClusterAwareDegrees())
                {
                    if (i < cluster_factors.size())
                    {
                        deg = static_cast<std::size_t>(deg * cluster_factors[i]);
                    }
                    else
                    {
                        deg = static_cast<std::size_t>(deg * 1.0);
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
                        deg = std::size_t(deg + stagnation_bonus);
                    }

                    std::size_t pseudo_bonus = 0;
                    if (std::abs(ritzv[i]) < std::abs(upperb - lowerb) * 0.1)
                    {
                        // Near-zero eigenvalues in pseudo-hermitian problems can be
                        // challenging
                        pseudo_bonus = 2;
                    }
                    deg = std::size_t(deg + pseudo_bonus);
                }

                deg = std::min(deg + deg_extra, deg_max);
            }
        }

        if constexpr (std::is_same<Base<T>, float>::value)
            deg = std::max(deg, std::size_t(8));

        degrees[i] = deg + (deg % 2); // force even
    }

    // Assume positive Ritz values in [0, unconverged). Sort these columns by degree (ascending).
        // Original sorting
        for (std::size_t j = 0; j < unconverged - 1; ++j)
            for (std::size_t k = j; k < unconverged; ++k)
                if (degrees[k] < degrees[j])
                {
                    swap_kj(k, j, degrees);
                    swap_kj(k, j, ritzv);
                    swap_kj(k, j, resid);
                    single->Swap(k + locked, j + locked);
                }

    const std::size_t* max_deg_ptr =
        std::max_element(degrees, degrees + unconverged); //What is the max degree for?
    return *max_deg_ptr;
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
        std::size_t stagnation_bonus = 0;
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
                swap_kj(k, j, degrees);
                swap_kj(k, j, ritzv);
                swap_kj(k, j, resid);
                swap_kj(k, j, residLast);
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
std::size_t Algorithm<T>::locking_pseudo_v2(
    ChaseBase<T>* single, std::size_t N, std::size_t unconverged,
    std::size_t nex, Base<T> tol, std::size_t* index, Base<T>* Lritzv,
    Base<T>* resid, Base<T>* residLast,
    std::vector<Base<T>>* early_locked_residuals, std::size_t* degrees,
    std::size_t locked, std::size_t iteration, std::size_t nev,
    std::size_t n_found_neg, std::size_t n_found_pos, std::size_t n_skip_tail)
{
    SCOPED_NVTX_RANGE();

    std::vector<std::size_t> index_unconverged;
    std::vector<Base<T>> residLast_unconverged(unconverged);
    std::copy_n(resid, unconverged, residLast_unconverged.begin());

    std::size_t converged = 0;
    std::size_t locked_neg_this = 0;
    std::vector<std::size_t> index_converged;

    for (std::size_t k = 0; k < (nev-n_found_neg); k++)
    {
        int j = index[k];
        
        bool is_neg = (Lritzv[j] < Base<T>(0));

#ifndef CHASE_SAVE_RESIDUALS
        if ((resid[j] <= tol || (resid[j] > tol && resid[j] >= residLast[k] &&
                                resid[j] <= 1000.0 * tol && iteration >= 4)) && ((is_neg && n_found_neg+locked_neg_this < nev) || !is_neg))
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
                oss << "early locking unconverged pair is:" << resid[j]
                    << " was:" << residLast[k] << " tolerance is: " << tol
                    << " val: " << Lritzv[j] << "\n";
                single->Output(oss.str());
#endif
                early_locked_residuals->push_back(resid[j]);
            }
#ifdef CHASE_OUTPUT
            else{
                std::ostringstream oss;
                oss << "locking converged pair is:" << j << " " << resid[j]
                    << " was:" << residLast[k] << " tolerance is: " << tol
                    << " val: " << Lritzv[j] << "\n";
                single->Output(oss.str());
            }

#endif
#endif
            index_converged.push_back(j);

            ++converged;
            if(is_neg)
            {
                ++locked_neg_this;
            }
        }
        else
        {
            index_unconverged.push_back(j);
        }
    }

    for(std::size_t k = (nev-n_found_neg); k < (unconverged-(nev-n_found_pos)); k++)
    {
        index_unconverged.push_back(index[k]);
    }

    for (std::size_t k = (unconverged-(nev-n_found_pos)); k < unconverged; k++)
    {
        int j = index[k];
        
        bool is_neg = (Lritzv[j] < Base<T>(0));

#ifndef CHASE_SAVE_RESIDUALS
        if ((resid[j] <= tol || (resid[j] > tol && resid[j] >= residLast[k] &&
                                resid[j] <= 1000.0 * tol && iteration >= 4)) && ((is_neg && n_found_neg+locked_neg_this < nev) || !is_neg))
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
                oss << "early locking unconverged pair is:" << resid[j]
                    << " was:" << residLast[k] << " tolerance is: " << tol
                    << " val: " << Lritzv[j] << "\n";
                single->Output(oss.str());
#endif
                early_locked_residuals->push_back(resid[j]);
            }
#ifdef CHASE_OUTPUT
            else{
                std::ostringstream oss;
                oss << "locking converged pair is:" << j << " "<< resid[j]
                    << " was:" << residLast[k] << " tolerance is: " << tol
                    << " val: " << Lritzv[j] << "\n";
                single->Output(oss.str());
            }

#endif
#endif
            index_converged.push_back(j);

            ++converged;
            if(is_neg)
            {
                ++locked_neg_this;
            }
        }
        else
        {
            index_unconverged.push_back(j);
        }
    }

    // Swap converged columns to the beginning (positions 0 .. converged-1).
    for (std::size_t t = 0; t < converged; ++t)
    {
        std::size_t j = index_converged[t];
        if (j != t)
        {
            swap_kj(j, t, resid);
            swap_kj(j, t, degrees);
            swap_kj(j, t, Lritzv);
            single->Swap(j + locked, t + locked); //TODO : We need to ensure that swapping twice do not reswap a locked column into the subspace
        }
    }

    for (std::size_t i = converged; i < unconverged; ++i)
    {
        residLast[i] = residLast_unconverged[index_unconverged[i - converged]];
    }

    return converged;
}

template <class T>
std::size_t Algorithm<T>::locking_pseudo_v3(
    ChaseBase<T>* single, std::size_t N, std::size_t unconverged,
    std::size_t nex, Base<T> tol, std::size_t* index, Base<T>* Lritzv,
    Base<T>* resid, Base<T>* residLast,
    std::vector<Base<T>>* early_locked_residuals, std::size_t* degrees,
    std::size_t locked, std::size_t iteration, std::size_t nev)
{
    SCOPED_NVTX_RANGE();

    std::vector<std::size_t> index_converged;
    std::vector<std::size_t> index_unconverged;
    std::vector<Base<T>> residLast_unconverged(2*unconverged);

    std::copy_n(resid, 2*unconverged, residLast_unconverged.begin());

    std::size_t converged = 0;

    for (std::size_t k = 0; k < unconverged-nex; k++)
    {
        std::size_t j = index[k];

#ifndef CHASE_SAVE_RESIDUALS
        if ((resid[j] <= tol || (resid[j] > tol && resid[j] >= residLast[k] &&
                                resid[j] <= 1000.0 * tol && iteration >= 4)))
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
                oss << "early locking unconverged pair is:" << resid[j]
                    << " was:" << residLast[k] << " tolerance is: " << tol
                    << " val: " << Lritzv[j] << "\n";
                single->Output(oss.str());
#endif
                early_locked_residuals->push_back(resid[j]);
                //early_locked_residuals->push_back(resid[2*unconverged-j-1]);
            }
#ifdef CHASE_OUTPUT
            else{
                std::ostringstream oss;
                oss << "locking converged pair is:" << j << " " << resid[j]
                    << " was:" << residLast[k] << " tolerance is: " << tol
                    << " val: " << Lritzv[j] << "\n";

                oss << "locking converged pair is:" << 2*unconverged-j-1 << " " << resid[2*unconverged-j-1]
                    << " was:" << residLast[k] << " tolerance is: " << tol
                    << " val: " << Lritzv[2*unconverged-j-1] << "\n";
                single->Output(oss.str());
            }
#endif
            if (j != converged)
            {
                swap_kj(j, converged, resid);
                swap_kj(j, converged, Lritzv);
                single->Swap(j + locked, converged + locked);

                /*swap_kj(2*unconverged-j-1, 2*unconverged-converged-1, resid);
                swap_kj(2*unconverged-j-1, 2*unconverged-converged-1, Lritzv);
                single->Swap(2*unconverged-j-1 + locked, 2*unconverged-converged-1 + locked);*/
            }

            index_converged.push_back(j);

            converged++;
#endif
        }
        else
        {
            index_unconverged.push_back(j);
        }
    }

    for (std::size_t k = unconverged-nex; k < unconverged; ++k)
    {
        index_unconverged.push_back(index[k]);
    }

    for (std::size_t i = converged; i < unconverged; ++i)
    {
        residLast[i] = residLast_unconverged[index_unconverged[i - converged]];
        residLast[2*unconverged-converged-1-(i-converged)] = residLast_unconverged[index_unconverged[i - converged]];
    }

    return converged;
}

template <class T>
std::size_t Algorithm<T>::locking_pseudo(
    ChaseBase<T>* single, std::size_t N, std::size_t unconverged,
    std::size_t nex, Base<T> tol, std::size_t* index, Base<T>* Lritzv,
    Base<T>* resid, Base<T>* residLast,
    std::vector<Base<T>>* early_locked_residuals, std::size_t* degrees,
    std::size_t locked, std::size_t iteration, std::size_t nev,
    std::size_t n_found_neg, std::size_t n_found_pos, std::size_t n_skip_tail)
{
    SCOPED_NVTX_RANGE();

    std::vector<std::size_t> index_unconverged;
    std::vector<Base<T>> residLast_unconverged(unconverged);
    std::copy_n(resid, unconverged, residLast_unconverged.begin());

    std::size_t converged = 0;
    std::size_t locked_neg_this = 0;
    std::vector<std::size_t> index_converged;

    std::size_t idx = 0;
    while (Lritzv[index[idx]] < Base<T>(0) && idx < unconverged)
    {
        idx++;
    }
    idx -= (nev - n_found_neg);

    for (std::size_t k = 0; k < idx; k++)
    {
        index_unconverged.push_back(index[k]);
    }

    for (std::size_t k = idx; k < idx+(2*nev-n_found_pos-n_found_neg); ++k)
    {
        int j = index[k];
        
        bool is_neg = (Lritzv[j] < Base<T>(0));

#ifndef CHASE_SAVE_RESIDUALS
        if ((resid[j] <= tol || (resid[j] > tol && resid[j] >= residLast[k] &&
                                resid[j] <= 1000.0 * tol && iteration >= 4)) && ((is_neg && n_found_neg+locked_neg_this < nev) || !is_neg))
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
                oss << "early locking unconverged pair is:" << resid[j]
                    << " was:" << residLast[k] << " tolerance is: " << tol
                    << " val: " << Lritzv[j] << "\n";
                single->Output(oss.str());
#endif
                early_locked_residuals->push_back(resid[j]);
            }
#ifdef CHASE_OUTPUT
            else{
                std::ostringstream oss;
                oss << "locking converged pair is:" << j << " "<< idx << " "<< resid[j]
                    << " was:" << residLast[k] << " tolerance is: " << tol
                    << " val: " << Lritzv[j] << "\n";
                single->Output(oss.str());
            }

#endif
#endif
            /*if (j != static_cast<int>(converged))
            {
                swap_kj(j, converged, resid);
                swap_kj(j, converged, Lritzv);
                swap_kj(j, converged, degrees);
                single->Swap(j + locked, converged + locked);
            }*/

            index_converged.push_back(j);

            ++converged;
            if(is_neg)
            {
                ++locked_neg_this;
            }
        }
        else
        {
            index_unconverged.push_back(j);
        }
    }

    for (std::size_t k = idx + (2 * nev - n_found_pos - n_found_neg); k < unconverged; ++k)
    {
        index_unconverged.push_back(index[k]);
    }

    // Swap converged columns to the beginning (positions 0 .. converged-1).
    for (std::size_t t = 0; t < converged; ++t)
    {
        std::size_t j = index_converged[t];
        if (j != t)
        {
            swap_kj(j, t, resid);
            swap_kj(j, t, degrees);
            swap_kj(j, t, Lritzv);
            single->Swap(j + locked, t + locked); //TODO : We need to ensure that swapping twice do not reswap a locked column into the subspace
        }
    }
    // Index for caller: unconverged positions in [0 .. unconverged-converged-1] after they advance pointers.
    // TODO : updating index is useless. To erase. 
    for (std::size_t i = 0; i < unconverged - converged; ++i)
    {
        index[i] = index_unconverged[i] - converged;
    }

    for (std::size_t i = converged; i < unconverged; ++i)
    {
        residLast[i] = residLast_unconverged[index_unconverged[i - converged]];
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

    single->FilterPhaseStart();
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
    single->FilterPhaseEnd();

    return Av;
}

template <class T>
std::size_t Algorithm<T>::filter_H2(ChaseBase<T>* single, std::size_t n,
                                    std::size_t unconverged, std::size_t* degrees,
                                    Base<T> lambda_1, Base<T> lower,
                                    Base<T> upper)
{
    SCOPED_NVTX_RANGE();
    (void)n;

    if (lower >= upper)
    {
        std::swap(lower, upper);
    }

    Base<T> c = (upper + lower) / 2;
    Base<T> e_half = (upper - lower) / 2;
    Base<T> sigma_1 = e_half / (lambda_1 - c);
    Base<T> sigma = sigma_1;

    std::size_t deg_max = 0;
    for (std::size_t i = 0; i < unconverged; ++i)
        deg_max = std::max(deg_max, degrees[i]);

    std::size_t Av = 0;

    // First step: W = alpha*(H²*V - c*V) on first half only (cols [0, unconverged))
    T alpha_first = T(sigma_1 / e_half);
    T gamma_first = T(-alpha_first * c);
    single->HEMM_H2(unconverged, alpha_first, T(0), gamma_first, 0, 0);
    Av += 2 * unconverged;

    // Recurrence: t = 2 .. deg_max. Active columns [s, unconverged); retire only from the left.
    std::size_t s = 0;
    for (std::size_t t = 2; t <= deg_max; ++t)
    {
        if (s >= unconverged)
            break;
        std::size_t active_count = unconverged - s;

        Base<T> tau = 1.0 / (2.0 / sigma_1 - sigma);
        T alpha = T(2.0 * tau / e_half);
        Base<T> beta_recur = sigma * tau;

        single->HEMM_H2(unconverged, alpha, T(-beta_recur), T(-alpha * c), s, 0);
        Av += 2 * active_count;

        sigma = tau;

        while (s < unconverged && degrees[s] <= t)
            ++s;
    }

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
        single->Lanczos(m, upperb);
        return 0;
    }

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

    delete[] ThetaSorted;
    delete[] Theta;
    delete[] Tau;
    delete[] ritzV;
    return idx;
}

template <class T>
std::size_t Algorithm<T>::lanczos_for_H2(ChaseBase<T>* single, int N,
                                          int numvec, int m, int nevex,
                                          Base<T>* upperb, bool mode,
                                          Base<T>* ritzv_)
{
    SCOPED_NVTX_RANGE();

    assert(m >= 1);

    if (!mode)
    {
        single->Lanczos(m, upperb);
        if (upperb)
            *upperb = (*upperb) * (*upperb);
        return 0;
    }

    // Same workflow as lanczos, but H² bounds: μ_1 = (min |λ|)²,
    // μ_nevnex = (λ at DOS quantile)², b_sup = (max |λ|)²; use full Ritz set
    // and search_hi = (N/2 - nev - nex - 1)/N for pseudo-Hermitian.
    Base<T>* Theta = new Base<T>[numvec * m]();
    Base<T>* Tau = new Base<T>[numvec * m]();
    Base<T>* ritzV = new Base<T>[m * m]();

    single->Lanczos(m, numvec, upperb, Theta, Tau, ritzV);

    double* ThetaSorted = new double[numvec * m];
    for (auto k = 0; k < numvec * m; ++k)
        ThetaSorted[k] = Theta[k];
    std::sort(ThetaSorted, ThetaSorted + numvec * m, std::less<double>());

    const double sigma = 0.25;
    const double thresh = 2 * sigma * sigma / 10;
    const auto G = [&](double x) -> double
    { return 0.5 * (1 + std::erf(x / std::sqrt(2 * sigma * sigma))); };

    // H² bounds: b_sup = (max |λ|)², μ_1 = (min |λ|)²
    Base<T> max_abs = 0;
    Base<T> min_abs = std::abs(Theta[0]);
    int i_minabs = 0;

    for (int i = 0; i < numvec * m; ++i)
    {
        Base<T> a = std::abs(Theta[i]);
        if (a > max_abs)
            max_abs = a;
        if (a < min_abs)
        {
            min_abs = a;
            i_minabs = i;
        }
    }
    Base<T> mu_1 = Theta[i_minabs] * Theta[i_minabs];
    Base<T> b_sup_full = max_abs * max_abs;
    if (upperb)
        *upperb = b_sup_full;

    // DOS quantile: search_hi = (N/2 - nev - nex - 1)/N; λ where CDF(λ) = search_hi
    // (matches Julia _dos_quantile: Gaussian-broadened, return closer of i / i-1, else last)
    ChaseConfig<T>& config = single->GetConfig();
    std::size_t nev = config.GetNev();
    std::size_t nex = config.GetNex();
    double search_hi =
        (static_cast<double>(N) / 2 - static_cast<double>(nev) -
         static_cast<double>(nex) - 1) /
        static_cast<double>(N);
    if (search_hi < 0.0)
        search_hi = 0.0;
    if (search_hi > 1.0)
        search_hi = 1.0;

    Base<T> lambda_nevnex = static_cast<Base<T>>(ThetaSorted[numvec * m - 1]);
    double prev = 0;
    const std::size_t n_dos = numvec * m;
    // TODO : Update lambda_nevnex IF = lambda1. 
    for (std::size_t i = 0; i < n_dos; ++i)
    {
        double curr = 0;
        for (std::size_t j = 0; j < n_dos; ++j)
        {
            if (ThetaSorted[i] < (Theta[j] - thresh))
                curr += 0;
            else if (ThetaSorted[i] > (Theta[j] + thresh))
                curr += Tau[j] * 1;
            else
                curr += Tau[j] * G(ThetaSorted[i] - Theta[j]);
        }
        curr /= numvec;

        if (curr > search_hi)
        {
            if (std::abs(curr - search_hi) < std::abs(prev - search_hi))
                lambda_nevnex = static_cast<Base<T>>(ThetaSorted[i]);
            else
                lambda_nevnex = static_cast<Base<T>>(
                    i > 0 ? ThetaSorted[i - 1] : ThetaSorted[i]);
            break;
        }
        prev = curr;
        lambda_nevnex = static_cast<Base<T>>(ThetaSorted[i]);
        /*if(lambda_nevnex == Theta[i_minabs])
        {
            lambda_nevnex = static_cast<Base<T>>(ThetaSorted[i+1]);
        }*/
    }
    Base<T> mu_nevnex = lambda_nevnex * lambda_nevnex;

    // threshold_V = λ_nevnex (Julia): extract eigenvectors with eigenvalue ≤ λ_nevnex.
    // C++ uses the last Lanczos run only; Julia fills from all nvec runs.
    int idx = 0;
    for (int i = 0; i < m; ++i)
    {
        if (Theta[(numvec - 1) * m + i] > lambda_nevnex)
        {
            idx = i - 1;
            break;
        }
        idx = i + 1;  // all so far ≤ lambda_nevnex
    }
    if (idx < 0)
        idx = 0;

    if (idx > 0)
    {
        T* ritzVc = new T[m * m]();
        for (auto i = 0; i < m * m; ++i)
            ritzVc[i] = T(ritzV[i]);
        single->LanczosDos(idx, m, ritzVc);
        delete[] ritzVc;
    }

    // Fill ritzv_ with H² values: first idx entries = squared Ritz, then μ_1,
    // last = μ_nevnex
    if (ritzv_)
    {
        for (int i = 0; i < idx; ++i)
            ritzv_[i] = Theta[(numvec - 1) * m + i] * Theta[(numvec - 1) * m + i];
        for (int i = idx; i < nevex - 1; ++i)
            ritzv_[i] = mu_1;
        ritzv_[nevex - 1] = mu_nevnex;
    }

    // Intersperse (same as lanczos)
    for (auto i = 1; i < idx; ++i)
    {
        auto j = i * (nevex / idx);
        single->Swap(i, j);
        if (ritzv_)
            std::swap(ritzv_[i], ritzv_[j]);
    }

    delete[] ThetaSorted;
    delete[] Theta;
    delete[] Tau;
    delete[] ritzV;
    return static_cast<std::size_t>(idx);
}

template <class T>
void Algorithm<T>::solve(ChaseBase<T>* single)
{
    ChaseConfig<T>& config = single->GetConfig();
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

    Base<T> lowerb, upperb, lambda;

    Base<T> cond;

    std::vector<std::size_t> degrees_(nev + nex);
    std::vector<Base<T>> residLast_(nevex);
    std::vector<Base<T>> early_locked_residuals;
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
    if (2.0 * (lanczos_iter / 2) < lanczos_iter)
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
            deg =
                calc_degrees(single, N, unconverged, nex, upperb, lowerb,
                                tol, ritzv, resid, residLast, degrees, locked);            
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
                    });
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

template <class T>
std::size_t Algorithm<T>::reinit_phantom_eigenvectors_pseudo(
    ChaseBase<T>* single, std::size_t unconverged, std::size_t locked,
    std::size_t nex, Base<T>* ritzv)
{
    constexpr Base<T> PURGE_RATIO_THRESHOLD = 1.5;
    std::size_t n_neg = 0;
    for (std::size_t i = 0; i < unconverged; ++i)
        if (ritzv[i] < Base<T>(0))
            ++n_neg;
    std::size_t n_pos = unconverged - n_neg;

    std::vector<std::size_t> reinit_indices;
    std::size_t npairs = std::min(nex, std::min(n_neg, n_pos));
    for (std::size_t k = 0; k < npairs; ++k)
    {
        std::size_t i = n_neg - 1 - k;
        std::size_t j = n_neg + k;
        Base<T> la = std::abs(ritzv[i]);
        Base<T> lb = std::abs(ritzv[j]);
        Base<T> r = (la < lb) ? (lb / (la + 1e-30)) : (la / (lb + 1e-30));
        if (r > PURGE_RATIO_THRESHOLD)
        {
            reinit_indices.push_back(i);
            reinit_indices.push_back(j);
        }
    }
    std::size_t n_reinit = reinit_indices.size();
    if (n_reinit > 0)
    {
        single->ReinitColumns(locked, reinit_indices.data(), n_reinit);
#ifdef CHASE_OUTPUT
        std::ostringstream oss;
        oss << "  [purge] Reinitialized " << n_reinit
            << " column(s) (outlier ± pairs, |λ| ratio > "
            << static_cast<double>(PURGE_RATIO_THRESHOLD)
            << "); excluding from bounds and locking this iter.\n";
        single->Output(oss.str());
#endif
    }
    return n_reinit;
}

template <class T>
void Algorithm<T>::solve_pseudo(ChaseBase<T>* single)
{
    // Pseudo-Hermitian (BSE) path: subspace size 2*nevex; filter on H²; bounds
    // from Lanczos on H². Step 1: up to Lanczos and get bounds.
    single->Start();

    ChaseConfig<T>& config = single->GetConfig();
    const std::size_t N = config.GetN();
    const std::size_t nev = config.GetNev();
    const std::size_t nex = config.GetNex();
    const std::size_t nevex = nev + nex;
    std::size_t unconverged = nevex; // subspace size for pseudo

    Base<T>* ritzv_ = single->GetRitzv();
    Base<T>* ritzv = ritzv_;

    //Degree initialization for H2
    std::size_t deg = config.GetDeg();
    deg += deg % 2;
    deg = std::min(deg, config.GetMaxDeg());
    std::vector<std::size_t> degrees_(2*nevex);
    for (std::size_t i = 0; i < unconverged; ++i)
    {
        degrees_[i] = deg;
    }
    std::size_t* degrees = degrees_.data();

    //Residual initialization for H2
    const double tol = config.GetTol();
    Base<T>* resid_ = single->GetResid();
    std::vector<Base<T>> residLast_(2 * nevex);
    for (std::size_t i = 0; i < 2*unconverged; ++i)
    {
        resid_[i] = std::numeric_limits<Base<T>>::max();
        residLast_[i] = std::numeric_limits<Base<T>>::max();
    }
    Base<T>* resid = resid_;
    Base<T>* residLast = residLast_.data();

    // Initialize vectors (2*nevex columns in kernel) and optional QR
    bool random = !config.UseApprox();
    single->initVecs(random);
    if (random)
        single->QR(0, 1.0);

    // Lanczos for H²: even iteration count for symmetric spectrum
    Base<T> upperb = Base<T>(0);
    const std::size_t num_lanczos = config.GetNumLanczos();
    std::size_t lanczos_iter = std::min(nevex, std::min(N / 2, config.GetLanczosIter()));
    if (2u * (lanczos_iter / 2u) < lanczos_iter)
    {
        config.SetLanczosIter(lanczos_iter - 1);
        lanczos_iter = config.GetLanczosIter();
    }

#ifdef CHASE_OUTPUT
    {
        std::ostringstream oss;
        oss << "entering lanczos_for_H2..." << std::endl;
        single->Output(oss.str());
    }
#endif
    
    std::size_t idx = lanczos_for_H2(single, static_cast<int>(N),
                                     static_cast<int>(num_lanczos),
                                     static_cast<int>(lanczos_iter),
                                     static_cast<int>(nevex), &upperb, true,
                                     ritzv_);
    (void)idx; // used when filter/QR/locking are implemented

    #ifdef CHASE_OUTPUT
    {
        std::ostringstream oss;
        oss << "exiting lanczos_for_H2..." << std::endl;
        single->Output(oss.str());
    }
#endif

    // H² bounds: b_sup = upperb, μ_1 = (min λ²), μ_nevnex = ritzv_[nevex-1]
    Base<T> mu_1 = *std::min_element(ritzv_, ritzv_ + nevex - 1); 
    Base<T> mu_nevnex = ritzv_[nevex - 1];
    if (upperb > 0)
    {
        upperb = upperb * config.GetUpperbScaleRate();
    }
    else
    {
        upperb = upperb / config.GetUpperbScaleRate();
    }
    //Chebyshev polynomial parameters for H2: [lambda_1, lower, b_sup]
    Base<T> lambda_1 = mu_1;  
    Base<T> lower = mu_nevnex;
    Base<T> new_mu_nevex = lower;
    Base<T> new_lambda_1 = lambda_1;
    Base<T> b_sup = upperb;    
    lower = lower * config.GetDecayingRate();

#ifdef CHASE_OUTPUT
    {
        std::ostringstream oss;
        oss << "solve_pseudo H² bounds: lambda_1 = " << std::scientific
            << std::setprecision(6) << lambda_1 << "\t lower(mu_nevnex) = "
            << std::setprecision(6) << lower << "\t b_sup = "
            << std::setprecision(6) << b_sup << std::endl;
        single->Output(oss.str());
    }
#endif

    //Initalize index vectors to keep track of the residuals and the indices of the eigenpairs
    std::vector<std::size_t> index(2*unconverged); //initial unconverged = 2*(nev+nex)
    std::iota(index.begin(), index.begin() + 2*unconverged, 0);

    std::vector<std::size_t> index_temp(unconverged);
    std::vector<std::size_t> index_res(unconverged);

    Base<T> cond;
    std::vector<Base<T>> early_locked_residuals;

    std::size_t locked = 0;
    std::size_t new_converged = 0;
    std::size_t prev_new_converged = 0;
    std::size_t iteration = 0;

    while (locked < nev && unconverged > 0 && iteration < config.GetMaxIter())
    {
        // Update lower parameter in [lambda_1, lower, b_sup] for the polynomial of H2
        if (iteration > 0)
        {
            new_mu_nevex = new_mu_nevex * new_mu_nevex;
            new_lambda_1 = new_lambda_1 * new_lambda_1;

            if (new_lambda_1 > lambda_1 && new_lambda_1 < new_mu_nevex)
            {
                //lambda_1 = new_lambda_1;
            }
            
            if (new_mu_nevex < lower && new_mu_nevex > lambda_1)
            {
                lower = new_mu_nevex;
            }
        }

#ifdef CHASE_OUTPUT
        {
            std::ostringstream oss;
            oss << std::scientific << "iteration: " << iteration << "\t"
                << std::setprecision(6) << lambda_1 << "\t"
                << std::setprecision(6) << lower << "\t"
                << std::setprecision(6) << b_sup << "\t" << unconverged
                << std::endl;
            single->Output(oss.str());
        }
#endif

        // ------------------------------ DEGREES (adaptive or uniform) --------

        if (config.DoOptimization() && iteration != 0)
        {
            // compute new degrees and reorder based on new degrees (symmetric locking: locked_left = locked_right = locked)
            deg = calc_degrees_pseudo_H2(
                single, N, unconverged, nex, b_sup, lower, tol, ritzv, resid,
                residLast, degrees, locked);
        }

        // ------------------------------ FILTER (H²) --------------------------
        single->FilterPhaseStart();
        filter_H2(single, N, unconverged, degrees, lambda_1, lower, b_sup);
        single->FilterPhaseEnd();
        
        single->ApplyKconjugate(unconverged);

        Base<T> cc = (b_sup + lower) / 2;
        Base<T> ee = (b_sup - lower) / 2;
        if (ee <= Base<T>(0))
            ee = std::abs(lower - b_sup) / 2;
        Base<T> t_1 = (lambda_1 - cc) / ee;
        Base<T> t_k =
            (iteration > 0)
                ? (ritzv[0] * ritzv[0] - cc) / ee
                : t_1;
        std::complex<Base<T>> z1(t_1 * t_1 - 1, 0);
        std::complex<Base<T>> zk(t_k * t_k - 1, 0);
        Base<T> rho_1 = std::max(
            std::abs(std::complex<Base<T>>(t_1, 0) - std::sqrt(z1)),
            std::abs(std::complex<Base<T>>(t_1, 0) + std::sqrt(z1)));
        Base<T> rho_k = std::max(
            std::abs(std::complex<Base<T>>(t_k, 0) - std::sqrt(zk)),
            std::abs(std::complex<Base<T>>(t_k, 0) + std::sqrt(zk)));
        std::size_t deg_max = *std::max_element(degrees, degrees + unconverged);
        cond = std::pow(rho_k, static_cast<Base<T>>(degrees[0])) *std::pow(rho_1,static_cast<Base<T>>(deg_max - degrees[0]));

        //TODO : I don't think that the condition number should be based on squared values. 

        single->QR(locked, cond);

        // ------------------------- RAYLEIGH-RITZ -----------------------------
        single->RR(ritzv, unconverged);

        // ------------------------- RESIDUALS ---------------------------------
        single->Resd(ritzv, resid, locked);
                
        std::iota(index.begin(), index.begin() + 2*unconverged, 0);

        std::size_t n_reinit = 0;//reinit_phantom_eigenvectors_pseudo(single, unconverged, locked, nex, ritzv);

#ifdef CHASE_OUTPUT
        {
            std::ostringstream oss;
            oss << "unconverged = " << "\t" << unconverged << "\t"
                << "new_converged" << "\t" << prev_new_converged << "\n";
            oss << "index\tdegrees\tresid\tresidLast\tritzv\n";
            
            for (std::size_t i = 0; i < 2*unconverged; ++i)
            {
                oss << index[i] << "\t" << degrees[index[i]] << "\t"
                    << resid[index[i]] << "\t" << residLast[i] << "\t" << ritzv[index[i]]
                    << "\n";
            }

            single->Output(oss.str());
        }
#endif
        std::iota(index_temp.begin(), index_temp.begin() + unconverged, 0);

        if(locked == 0)
	    {
            std::sort(index_temp.begin(), index_temp.begin() + std::size_t(1.0*unconverged), //sort ritzv values only 1st nev-locked
                [&](int a, int b) { return (ritzv[a] < ritzv[b]); });
	    }
	    else
	    {
            std::sort(index_temp.begin(), index_temp.begin() + std::size_t(1.0*unconverged), //sort ritzv values only 1st nev-locked 
                [&](int a, int b) { return (ritzv[a] < ritzv[b]); }); //MIGHT BE WRONG HERE ACTUALLY. KEEP 0.95?

	    }

        std::size_t idx_new_mu = index_temp.data()[std::size_t(unconverged*0.95)-1];
        new_mu_nevex = ritzv[idx_new_mu] * config.GetDecayingRate();
        new_lambda_1 = ritzv[index_temp.data()[0]];

        // --------------- Locking ----------------------------------------------
        new_converged = locking_pseudo_v3(
            single, N, unconverged, nex, tol, index.data(), ritzv, resid,
            residLast, &early_locked_residuals, degrees, locked, iteration, nev);

        if(new_converged > 0)
        {
            single->ApplyKconjugate(new_converged);
        }

        //new_converged is the number of newly locked positive eigenpairs

        single->Lock(new_converged);

        locked += new_converged;
        unconverged -= new_converged;
        resid += new_converged;
        residLast += new_converged;
        ritzv += new_converged;
        degrees += new_converged;

#ifdef CHASE_OUTPUT
        if (new_converged > 0)
        {
            std::ostringstream oss;
            oss << "new_converged = " << new_converged
                << "  locked = " << locked
                << "\n";
            single->Output(oss.str());
        }
#endif
        prev_new_converged = new_converged;

        ++iteration;
    }

    // --------------- Final output: smallest positive first, then the rest ---------
    // Sort so positive Ritz values come first (ascending), then non-positive (ascending).
    std::size_t n_reorder = locked + unconverged;
    if (n_reorder == 0)
        n_reorder = 1;

    std::vector<std::size_t> perm(n_reorder);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(),
              [&](std::size_t i, std::size_t j) {
                  bool i_pos = (ritzv_[i] > Base<T>(0));
                  bool j_pos = (ritzv_[j] > Base<T>(0));
                  if (i_pos && !j_pos) return true;
                  if (!i_pos && j_pos) return false;
                  return ritzv_[i] < ritzv_[j];
              });

    std::vector<bool> visited(n_reorder, false);
    for (std::size_t i = 0; i < n_reorder; ++i)
    {
        if (visited[i] || perm[i] == i)
            continue;

        std::size_t current = i;
        Base<T> temp_ritz = ritzv_[i];
        Base<T> temp_resid = resid_[i];
        std::vector<std::size_t> cycle_indices;

        while (!visited[current])
        {
            visited[current] = true;
            cycle_indices.push_back(current);
            current = perm[current];
        }

        for (std::size_t k = 0; k < cycle_indices.size() - 1; ++k)
        {
            ritzv_[cycle_indices[k]] = ritzv_[cycle_indices[k + 1]];
            resid_[cycle_indices[k]] = resid_[cycle_indices[k + 1]];
            single->Swap(cycle_indices[k], cycle_indices[k + 1]);
        }
        ritzv_[cycle_indices.back()] = temp_ritz;
        resid_[cycle_indices.back()] = temp_resid;
    }

    single->set_early_locked_residuals(early_locked_residuals);
    single->End();
}

} // namespace chase
