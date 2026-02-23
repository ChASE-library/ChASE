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
    std::size_t locked_left, std::size_t locked_right)
{
    SCOPED_NVTX_RANGE();

    ChaseConfig<T> conf = single->GetConfig();
    const std::size_t deg_extra = conf.GetDegExtra();
    const std::size_t deg_max = conf.GetMaxDeg();

    // H² spectrum interval: center and half-length (for Chebyshev ellipse)
    Base<T> c_h2 = (upperb + lowerb) / 2;
    Base<T> e_h2 = (upperb - lowerb) / 2;

    if (e_h2 <= Base<T>(0))
    {
        for (std::size_t i = 0; i < unconverged; ++i)
            degrees[i] = deg_max + (deg_max % 2);
        return deg_max + (deg_max % 2);
    }

    // Main block: discard nex from both sides of unconverged block (total 2*nex).
    // ritzv, resid, residLast, degrees are already shifted (no locked values).
    const std::size_t n_main =
        (unconverged >= 2 * nex) ? (unconverged - 2 * nex) : 0;

    for (std::size_t i = 0; i < n_main; ++i)
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
                deg = std::min(deg + deg_extra, deg_max);
            }
        }

        if constexpr (std::is_same<Base<T>, float>::value)
            deg = std::max(deg, std::size_t(8));

        degrees[i] = deg + (deg % 2); // force even
    }

    std::size_t tail_deg = (n_main > 0) ? degrees[n_main - 1] : (deg_max + (deg_max % 2));
    for (std::size_t i = n_main; i < unconverged; ++i)
        degrees[i] = tail_deg;

    // Pair-wise symmetric degrees: pair i with (unconverged - 1 - i), both get max
    for (std::size_t i = 0; i < unconverged / 2; ++i)
    {
        std::size_t j = unconverged - 1 - i;
        std::size_t m = std::max(degrees[i], degrees[j]);
        degrees[i] = degrees[j] = m;
    }

    // Permutation by degree (negatives ascending, positives descending),
    // as in Julia chebyshev_filter_H2.jl / pseudo_evp!:
    //   idx_neg = findall(real.(Λ_display).<0); idx_pos = findall(real.(Λ_display).>0)
    //   perm_neg = idx_neg[sortperm(ritz_degrees[idx_neg])]
    //   perm_pos = idx_pos[sortperm(ritz_degrees[idx_pos], rev=true)]
    //   display_perm = vcat(perm_neg, perm_pos)
    std::vector<std::size_t> idx_neg, idx_pos;
    for (std::size_t i = 0; i < unconverged; ++i)
    {
        if (ritzv[i] < Base<T>(0))
            idx_neg.push_back(i);
        else
            idx_pos.push_back(i);
    }
    std::sort(idx_neg.begin(), idx_neg.end(),
              [degrees](std::size_t a, std::size_t b) {
                  return degrees[a] < degrees[b];
              });
    std::sort(idx_pos.begin(), idx_pos.end(),
              [degrees](std::size_t a, std::size_t b) {
                  return degrees[a] > degrees[b];
              });
    std::vector<std::size_t> display_perm;
    display_perm.reserve(unconverged);
    for (std::size_t k : idx_neg)
        display_perm.push_back(k);
    for (std::size_t k : idx_pos)
        display_perm.push_back(k);

    // Apply permutation to degrees, ritzv, resid, residLast via temp arrays
    std::vector<std::size_t> tmp_degrees(unconverged);
    std::vector<Base<T>> tmp_ritzv(unconverged);
    std::vector<Base<T>> tmp_resid(unconverged);
    std::vector<Base<T>> tmp_residLast(unconverged);
    for (std::size_t k = 0; k < unconverged; ++k)
    {
        tmp_degrees[k] = degrees[display_perm[k]];
        tmp_ritzv[k] = ritzv[display_perm[k]];
        tmp_resid[k] = resid[display_perm[k]];
        tmp_residLast[k] = residLast[display_perm[k]];
    }
    for (std::size_t k = 0; k < unconverged; ++k)
    {
        degrees[k] = tmp_degrees[k];
        ritzv[k] = tmp_ritzv[k];
        resid[k] = tmp_resid[k];
        //residLast[k] = tmp_residLast[k];
    }

    // Inverse permutation: inv_perm[old] = new
    std::vector<std::size_t> inv_perm(unconverged);
    for (std::size_t k = 0; k < unconverged; ++k)
        inv_perm[display_perm[k]] = k;

    // Apply permutation to vectors via Swap(locked_left + i, locked_left + j)
    // by following cycles of inv_perm
    std::vector<bool> visited(unconverged, false);
    for (std::size_t start = 0; start < unconverged; ++start)
    {
        if (visited[start])
            continue;
        std::size_t cur = start;
        std::vector<std::size_t> cycle;
        do
        {
            visited[cur] = true;
            cycle.push_back(cur);
            cur = inv_perm[cur];
        } while (cur != start);
        // Rotate cycle: content at cycle[L-1] -> cycle[0], cycle[L-2] -> cycle[L-1], ...
        for (std::size_t s = cycle.size() - 1; s >= 1; --s)
            single->Swap(locked_left + cycle[0], locked_left + cycle[s]);
    }

    const std::size_t* max_deg_ptr =
        std::max_element(degrees, degrees + unconverged);
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


    // Index order (Julia): [neg 0..n_neg-1, pos unconverged-1..n_neg]. Count neg/pos for cap.
    std::size_t n_neg = 0;
    for (std::size_t k = 0; k < unconverged; ++k)
        if (Lritzv[index[k]] < Base<T>(0))
            ++n_neg;
    std::size_t n_pos = unconverged - n_neg;
    // Julia: cap at nev negative and nev positive locked. n_skip_tail = reinit (excluded this iter).
    std::size_t locked_neg_this = 0;
    std::size_t locked_pos_this = 0;

    // 3. Locking step
    std::size_t converged = 0;
    
    for(std::size_t k = 0; k < nex; k++)
        index_unconverged.push_back(index[k]);

    for (std::size_t k = nex; k < unconverged-nex; ++k)
    {
        int j = index[k];
        
        bool is_neg = (Lritzv[j] < Base<T>(0));

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
            if (j != static_cast<int>(converged))
            {
                swap_kj(j, converged, resid);
                swap_kj(j, converged, Lritzv);
                swap_kj(j, converged, degrees);
                single->Swap(j + locked, converged + locked);
            }

            ++converged;
            if (nev > 0)
            {
                if (is_neg)
                    ++locked_neg_this;
                else
                    ++locked_pos_this;
            }
        }
        else
        {
            index_unconverged.push_back(index[k]);
        }
    }

    for (std::size_t k = unconverged-nex; k < unconverged; k++)
    {
        index_unconverged.push_back(index[k]);
    }
    // We need to update the index set for the eventual calculation of the new
    // lowerb, because the all the sets and vectors will be shifted by
    // converged.
    std::copy_n(index_unconverged.begin(), unconverged - converged, index);

    for (std::size_t i = converged; i < unconverged; ++i)
        residLast[i] = residLast_unconverged[index_unconverged[i-converged]];

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
                                    std::size_t nevex, std::size_t* degrees,
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
    for (std::size_t i = 0; i < nevex; ++i)
        deg_max = std::max(deg_max, degrees[i]);

    std::size_t Av = 0;

    // First step: W = alpha*(H²*V - c*V), alpha = sigma_1 / e_half
    T alpha_first = T(sigma_1 / e_half);
    T gamma_first = T(-alpha_first * c);
    single->HEMM_H2(nevex, alpha_first, T(0), gamma_first, 0, 0);
    Av += 2 * nevex;

    // Recurrence: t = 2 .. deg_max. Active columns [s, r] with r = nevex - s - 1
    // (0-based). Retire from both ends when degrees[s] <= t.
    std::size_t s = 0;
    for (std::size_t t = 2; t <= deg_max; ++t)
    {
        std::size_t r = nevex - s;
        if (s >= r)
            break;
        std::size_t active_count = r - s;

        Base<T> tau = 1.0 / (2.0 / sigma_1 - sigma);
        T alpha = T(2.0 * tau / e_half);
        Base<T> beta_recur = sigma * tau;

        single->HEMM_H2(nevex, alpha, T(-beta_recur), T(-alpha * c), s, 2 * s);
        Av += 2 * active_count;

        sigma = tau;

        while (s < nevex && degrees[s] <= t)
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
    const std::size_t num_lanczos = config.GetNumLanczos();

    Base<T>* ritzv_ = single->GetRitzv();
    Base<T>* resid_ = single->GetResid();
    Base<T> upperb = Base<T>(0);
    std::size_t unconverged = 2 * nevex; // subspace size for pseudo
    const double tol = config.GetTol();

    // Initialize vectors (2*nevex columns in kernel) and optional QR
    bool random = !config.UseApprox();
    single->initVecs(random);
    if (random)
        single->QR(0, 1.0);

    // Lanczos for H²: even iteration count for symmetric spectrum
    std::size_t lanczos_iter =
        std::min(nevex, std::min(N / 2, config.GetLanczosIter()));
    if (2u * (lanczos_iter / 2u) < lanczos_iter)
    {
        config.SetLanczosIter(lanczos_iter - 1);
        lanczos_iter = config.GetLanczosIter();
    }
    std::size_t idx = lanczos_for_H2(single, static_cast<int>(N),
                                     static_cast<int>(num_lanczos),
                                     static_cast<int>(lanczos_iter),
                                     static_cast<int>(nevex), &upperb, true,
                                     ritzv_);
    (void)idx; // used when filter/QR/locking are implemented

    // H² bounds: b_sup = upperb, μ_1 = (min λ²), μ_nevnex = ritzv_[nevex-1]
    Base<T> mu_1 = 0;//*std::min_element(ritzv_, ritzv_ + nevex - 1);
    Base<T> mu_nevnex = ritzv_[nevex - 1];
    if (upperb > 0)
        upperb = upperb * config.GetUpperbScaleRate();
    else
        upperb = upperb / config.GetUpperbScaleRate();
    Base<T> lambda_1 = mu_1;   // for filter_H2 / ellipse
    Base<T> lower = mu_nevnex; // lower end of H² interval for degrees
    Base<T> b_sup = upperb;    // upper H² bound
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

    // Degrees, residLast, index vectors (size 2*nevex)
    std::vector<std::size_t> degrees_(2 * nevex);
    std::size_t deg = config.GetDeg();
    deg += deg % 2;
    deg = std::min(deg, config.GetMaxDeg());
    for (std::size_t i = 0; i < unconverged; ++i)
        degrees_[i] = deg;

    std::vector<Base<T>> residLast_(2 * nevex);
    std::vector<std::size_t> index(2 * nevex);
    std::vector<std::size_t> index_temp(2 * nevex);
    std::vector<std::size_t> index_res(2 * nevex);

    std::vector<Base<T>> early_locked_residuals;

    for (std::size_t i = 0; i < unconverged; ++i)
    {
        resid_[i] = std::numeric_limits<Base<T>>::max();
        residLast_[i] = std::numeric_limits<Base<T>>::max();
    }

    Base<T>* ritzv = ritzv_;
    Base<T>* resid = resid_;
    Base<T>* residLast = residLast_.data();
    std::size_t* degrees = degrees_.data();
    std::size_t locked = 0;
    std::size_t n_found_pos = 0;  // number of positive eigenvalues locked (Julia: stop when n_found_pos >= nev)
    std::size_t iteration = 0;
    std::size_t prev_new_converged = 0;  // for CHASE_OUTPUT table header (same style as Solve)
    std::iota(index.begin(), index.begin() + unconverged, 0);

    while (n_found_pos < nev && unconverged > 0 && iteration < config.GetMaxIter())
    {

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
            // calc_degrees_pseudo_H2 uses locked_left as offset of unconverged
            // block in kernel (Swap(locked_left+i, ...)); kernel layout is
            // [locked | unconverged], so we pass locked_left = locked.
            std::size_t locked_left = locked;
            std::size_t locked_right = 0;
            deg = calc_degrees_pseudo_H2(
                single, N, unconverged, nex, b_sup, lower, tol, ritzv, resid,
                residLast, degrees, locked_left, locked_right);
        }

        // ------------------------------ FILTER (H²) --------------------------
        single->FilterPhaseStart();
        filter_H2(single, N, unconverged, degrees, lambda_1, lower, b_sup);
        single->FilterPhaseEnd();

        // --------------------------------- QR --------------------------------
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
        std::size_t deg_max =
            *std::max_element(degrees, degrees + unconverged);
        Base<T> cond =
            std::pow(rho_k, static_cast<Base<T>>(degrees[0])) *
            std::pow(rho_1,
                    static_cast<Base<T>>(deg_max - degrees[0]));
        single->QR(locked, cond);

        // ------------------------- RAYLEIGH-RITZ -----------------------------
        single->RR(ritzv, unconverged);

        // ------------------------- RESIDUALS ---------------------------------
        single->Resd(ritzv, resid, locked);

        std::iota(index_res.begin(), index_res.begin() + unconverged, 0);
                
        // Build index vector for all unconverged eigenpairs
        std::iota(index.begin(), index.begin() + unconverged, 0);

        // 2. Sort first nev by |Lritzv| (ascending)
        std::sort(index_res.begin(), index_res.begin() + unconverged, //sort ritzv values only 1st nev-locked
            [&](int a, int b) { return (resid[a] < resid[b]); });

	    if(locked == 0)
	    {
            std::sort(index.begin(), index.begin() + std::size_t(0.95*unconverged), //sort ritzv values only 1st nev-locked
                [&](int a, int b) { return (ritzv[index_res.data()[a]] < ritzv[index_res.data()[b]]); });
	    }
	    else
	    {
            std::sort(index.begin(), index.begin() + std::size_t(0.95*unconverged), //sort ritzv values only 1st nev-locked 
                [&](int a, int b) { return (ritzv[index_res.data()[a]] < ritzv[index_res.data()[b]]); }); //MIGHT BE WRONG HERE ACTUALLY. KEEP 0.95?

	    }
            
        for(std::size_t k = 0; k < unconverged; k++) {
            index_temp.data()[k] = index_res.data()[index.data()[k]];
        }

        std::swap(index_temp,index);


        // --------------- n_neg / n_pos (for purge and index) ---------------
        std::size_t n_neg = 0;
        for (std::size_t i = 0; i < unconverged; ++i)
            if (ritzv[i] < Base<T>(0))
                ++n_neg;
        std::size_t n_pos = unconverged - n_neg;

        // --------------- Outlier purge (Julia): ± pairs in boundary with |λ| ratio > threshold ---------------
        constexpr Base<T> PURGE_RATIO_THRESHOLD = 1.5;
        std::vector<std::size_t> reinit_indices;
        std::size_t npairs = std::min(nex, std::min(n_neg, n_pos));
        for (std::size_t k = 0; k < npairs; ++k)
        {
            // Last k-th negative index, first k-th positive index (boundary pairs)
            std::size_t i = n_neg - 1 - k;  // negative
            std::size_t j = n_neg + k;      // positive
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
            {
                std::ostringstream oss;
                oss << "  [purge] Reinitialized " << n_reinit
                    << " column(s) (outlier ± pairs, |λ| ratio > "
                    << static_cast<double>(PURGE_RATIO_THRESHOLD)
                    << "); excluding from bounds and locking this iter.\n";
                single->Output(oss.str());
            }
#endif
        }

        // --------------- Index for locking (Julia: neg first, then pos reverse; reinit at end) ----
        // Build index = [neg 0..n_neg-1, pos (unconverged-1)..n_neg], then move reinit to end.
        /*std::size_t ineg = 0;
        for (std::size_t i = 0; i < unconverged; ++i)
            if (ritzv[i] < Base<T>(0))
                index_temp[ineg++] = i;
        std::size_t ipos = 0;
        for (std::size_t i = 0; i < unconverged; ++i)
            if (ritzv[i] >= Base<T>(0))
                index_temp[n_neg + (ipos++)] = i;
        for (std::size_t i = 0; i < ipos / 2; ++i)
            std::swap(index_temp[n_neg + i], index_temp[n_neg + ipos - 1 - i]);
        // Move reinit indices to the end so they are excluded from locking (n_skip_tail).
        std::vector<bool> is_reinit(unconverged, false);
        for (std::size_t idx : reinit_indices)
            is_reinit[idx] = true;
        std::size_t write = 0;
        for (std::size_t k = 0; k < unconverged; ++k)
            if (!is_reinit[index_temp[k]])
                index[write++] = index_temp[k];
        for (std::size_t k = 0; k < unconverged; ++k)
            if (is_reinit[index_temp[k]])
                index[write++] = index_temp[k];
        (void)write;
        assert(write == unconverged);*/

        // --------------- Bounds update (Julia: new_mu_nevex, μ_1); exclude reinit ----
        if (iteration > 0 && unconverged > 0)
        {
            std::size_t n_valid = unconverged - n_reinit;
            if (n_valid > 0)
            {
                std::size_t idx_offset = static_cast<std::size_t>(
                    0.95 * static_cast<double>(nevex > n_found_pos ? nevex - n_found_pos : 0));
                std::size_t pos = (idx_offset < n_valid) ? (n_valid - 1 - idx_offset) : 0;
                std::size_t idx_new_mu = 0;
                for (std::size_t i = 0, count = 0; i < unconverged; ++i)
                {
                    /*if (is_reinit[i])
                        continue;*/
                    if (count == pos)
                    {
                        idx_new_mu = i;
                        break;
                    }
                    ++count;
                }
                Base<T> new_mu_nevex = ritzv[idx_new_mu] * ritzv[idx_new_mu] * config.GetDecayingRate();
                /*
                if (new_mu_nevex < lower)
                    lower = new_mu_nevex;
                std::size_t min_res_idx = unconverged;
                for (std::size_t i = 0; i < unconverged; ++i)
                {
                    if (is_reinit[i])
                        continue;
                    if (min_res_idx == unconverged || resid[i] < resid[min_res_idx])
                        min_res_idx = i;
                }
                if (min_res_idx < unconverged)
                {
                    Base<T> lam_min_res = ritzv[min_res_idx] * ritzv[min_res_idx];
                    if (lam_min_res > lambda_1)
                        lambda_1 = lam_min_res;
                }
                */
            }
        }

#ifdef CHASE_OUTPUT
        {
            // Same style as Solve: index, degrees, resid, residLast, ritzv (tab-separated).
            // Sort by ritzv so the same eigenpair stays in the same row across iterations;
            // then residLast in row k matches resid in row k from the previous iteration.
            std::vector<std::size_t> idx_neg_disp, idx_pos_disp;
            for (std::size_t i = 0; i < unconverged; ++i)
                if (ritzv[i] < Base<T>(0))
                    idx_neg_disp.push_back(i);
                else
                    idx_pos_disp.push_back(i);
            // λ<0: sort by ritzv ascending (most negative first)
            std::sort(idx_neg_disp.begin(), idx_neg_disp.end(),
                      [&](std::size_t a, std::size_t b) {
                          return ritzv[a] < ritzv[b];
                      });
            // λ>0: sort by ritzv ascending (smallest positive first)
            std::sort(idx_pos_disp.begin(), idx_pos_disp.end(),
                      [&](std::size_t a, std::size_t b) {
                          return ritzv[a] < ritzv[b];
                      });
            const std::size_t n_neg_disp = idx_neg_disp.size();
            const std::size_t n_pos_disp = idx_pos_disp.size();
#ifndef NDEBUG
            const std::size_t n_print_neg = n_neg_disp;
            const std::size_t n_print_pos = n_pos_disp;
            const std::size_t neg_start = 0;
            const std::size_t pos_start = 0;
#else
            const std::size_t n_print_neg = std::min(std::size_t(20), n_neg_disp);
            const std::size_t n_print_pos = std::min(std::size_t(20), n_pos_disp);
            // Print middle 20 of each block (symmetric about center)
            const std::size_t neg_start = (n_neg_disp <= n_print_neg)
                ? 0 : (n_neg_disp - n_print_neg) / 2;
            const std::size_t pos_start = (n_pos_disp <= n_print_pos)
                ? 0 : (n_pos_disp - n_print_pos) / 2;
#endif
            std::ostringstream oss;
            oss << "unconverged = " << "\t" << unconverged << "\t"
                << "new_converged" << "\t" << prev_new_converged << "\n";
            oss << "index\tdegrees\tresid\tresidLast\tritzv\n";
            
            for (std::size_t i = 0; i < unconverged; ++i)
            {
                oss << "x" << "\t" << degrees[index[i]] << "\t"
                    << resid[index[i]] << "\t" << residLast[i] << "\t" << ritzv[index[i]]
                    << "\n";
            }
/*
            for (std::size_t i = 0; i < n_print_neg; ++i)
            {
                std::size_t idx = neg_start + i;
                std::size_t p = idx_neg_disp[idx];
                int disp_idx = -static_cast<int>(n_neg_disp) + static_cast<int>(idx);
                oss << disp_idx << "\t" << degrees[p] << "\t"
                    << resid[p] << "\t" << residLast[] << "\t" << ritzv[p]
                    << "\n";
            }
            if (n_neg_disp > 0 && n_pos_disp > 0)
                oss << "--- λ < 0 | λ > 0 ---\n";
            for (std::size_t i = 0; i < n_print_pos; ++i)
            {
                std::size_t idx = pos_start + i;
                std::size_t p = idx_pos_disp[idx];
                int disp_idx = static_cast<int>(idx + 1);
                oss << disp_idx << "\t" << degrees[p] << "\t"
                    << resid[p] << "\t" << residLast[p] << "\t" << ritzv[p]
                    << "\n";
            }
*/
            single->Output(oss.str());
        }
#endif

        // --------------- Locking ----------------------------------------------
        std::size_t n_found_neg = locked - n_found_pos;  // Julia: lock neg first, then pos
        std::size_t new_converged = locking_pseudo(
            single, N, unconverged, nex, tol, index.data(), ritzv, resid,
            residLast, &early_locked_residuals, degrees, locked, iteration, nev,
            n_found_neg, n_found_pos, n_reinit);

        // Count how many newly locked are positive (Julia: care about n_found_pos < nev)
        std::size_t new_pos = 0;
        for (std::size_t i = 0; i < new_converged; ++i)
            if (ritzv[i] > Base<T>(0))
                ++new_pos;
        n_found_pos += new_pos;

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
                << "  locked = " << locked << "  n_found_pos = " << n_found_pos
                << "\n";
            single->Output(oss.str());
        }
#endif
        prev_new_converged = new_converged;

        ++iteration;
    }

    // --------------- Final output (Julia: copy positive locked, sort ascending) ---------
    // Reorder so first nev positions are positive locked eigenpairs sorted by eigenvalue.
    std::size_t n_reorder = locked + unconverged;
    if (n_reorder == 0)
        n_reorder = 1;
    std::vector<std::size_t> idx_pos_locked;
    for (std::size_t i = 0; i < locked; ++i)
        if (ritzv_[i] > Base<T>(0))
            idx_pos_locked.push_back(i);
    std::sort(idx_pos_locked.begin(), idx_pos_locked.end(),
              [&](std::size_t i, std::size_t j) {
                  return ritzv_[i] < ritzv_[j];
              });
    // Permutation: first positions = positive locked (sorted), then the rest
    std::vector<std::size_t> perm(n_reorder);
    std::size_t p = 0;
    for (std::size_t i : idx_pos_locked)
        perm[p++] = i;
    for (std::size_t i = 0; i < locked && p < n_reorder; ++i)
        if (ritzv_[i] <= Base<T>(0))
            perm[p++] = i;
    for (std::size_t i = locked; i < n_reorder; ++i)
        perm[p++] = i;
    std::vector<std::size_t> inv_perm(n_reorder);
    for (std::size_t i = 0; i < n_reorder; ++i)
        inv_perm[perm[i]] = i;
    std::vector<bool> visited(n_reorder, false);
    for (std::size_t i = 0; i < n_reorder; ++i)
    {
        if (visited[i] || inv_perm[i] == i)
            continue;
        std::size_t current = i;
        std::vector<std::size_t> cycle_indices;
        while (!visited[current])
        {
            visited[current] = true;
            cycle_indices.push_back(current);
            current = inv_perm[current];
        }
        // Content at cycle[k] goes to inv_perm[cycle[k]] = cycle[k+1]. Forward rotation.
        const std::size_t L = cycle_indices.size();
        Base<T> temp_ritz = ritzv_[cycle_indices[L - 1]];
        Base<T> temp_resid = resid_[cycle_indices[L - 1]];
        for (std::size_t k = L - 1; k >= 1; --k)
        {
            ritzv_[cycle_indices[k]] = ritzv_[cycle_indices[k - 1]];
            resid_[cycle_indices[k]] = resid_[cycle_indices[k - 1]];
            single->Swap(cycle_indices[k], cycle_indices[k - 1]);
        }
        ritzv_[cycle_indices[0]] = temp_ritz;
        resid_[cycle_indices[0]] = temp_resid;
    }

    single->set_early_locked_residuals(early_locked_residuals);
    single->End();
}

} // namespace chase
