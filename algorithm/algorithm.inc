// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
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
        if constexpr (std::is_same<Base<T>, float>::value) {
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
std::size_t Algorithm<T>::locking(ChaseBase<T>* single, std::size_t N,
                                  std::size_t unconverged, Base<T> tol,
                                  Base<T>* Lritzv, Base<T>* resid,
                                  Base<T>* residLast, std::size_t* degrees,
                                  std::size_t locked)
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

	if (resid[j] <= tol || (single->isSym() && resid[j] >= residLast[j]))
        {
		if(single->isSym() && resid[j] > tol && resid[j] >= residLast[j]){
#ifdef CHASE_OUTPUT
                	std::ostringstream oss;
                	oss << "locking unconverged pair is:" << resid[j]
                   	<< " was:" << residLast[j] << " tolerance is: " << tol
                   	<< " val: " << Lritzv[j] << "\n";
                	single->Output(oss.str());
#endif
		}

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
std::size_t Algorithm<T>::locking_quasi(
    ChaseBase<T>* single, std::size_t N,
    std::size_t unconverged, std::size_t nex, Base<T> tol,
    Base<T>* Lritzv, Base<T>* resid,
    std::size_t* degrees, std::size_t locked)
{
    SCOPED_NVTX_RANGE();

    // Build index vector for all unconverged eigenpairs
    std::vector<int> index(unconverged); //unconverged = nev+nex-locked
    std::iota(index.begin(), index.end(), 0);

    // 1. Sort all by residuals (ascending)
    std::sort(index.begin(), index.end(), //sort all nev + nex - locked residuals
        [&](int a, int b) { return resid[a] < resid[b]; });

    // 2. Sort first nev by |Lritzv| (ascending)
    std::sort(index.begin(), index.begin() + unconverged - nex, //sort ritzv values only 1st nev-locked
        [&](int a, int b) { return (Lritzv[a] < Lritzv[b]); });
    

    // 3. Locking step
    std::size_t converged = 0;
    for (std::size_t k = 0; k < unconverged - nex; ++k) {
        int j = index[k];
        if (resid[j] <= tol) {
            if (j != converged) {
                swap_kj(j, converged, resid);
                swap_kj(j, converged, Lritzv);
                swap_kj(j, converged, degrees);
                single->Swap(j + locked, converged + locked);
            }
            ++converged;
        }
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
std::size_t Algorithm<T>::lanczos(ChaseBase<T>* single, int N, int numvec, int m,
                                  int nevex, Base<T>* upperb, bool mode,
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

#ifdef CHASE_OUTPUT
    /*
    std::cout << "THETA:";
    for (std::size_t k = 0; k < numvec * m; ++k) {
      if( k % 5 == 0 ) std::cout << "\n";
      std::cout << Theta[k] << " ";
    }
    std::cout << "\n";
  */
#endif

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
    const auto G = [&](double x) -> double {
        return 0.5 * (1 + std::erf(x / sqrt(2 * sigma * sigma)));
    };

    std::size_t bound = m;
    if(single->isPseudoHerm())
    {
	//The spectrum of a Pseudo-Hermitian matrix is symmetric over the y-axis.
	bound /= 2; 
    }

    for (auto i = 0; i < numvec * bound-1; ++i)
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
            if (std::abs(curr - search) < std::abs(prev - search))
                lowerb = ThetaSorted[i];
            else
                lowerb = ThetaSorted[i-1];
            break;
        }
        prev = curr;
    }

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

                oss << "The input matrix is explicitly symmetric/Hermtian, ChASE can continue to proceed\n";

                single->Output(oss.str());
            }
#endif
        }else
        {
#ifdef CHASE_OUTPUT
            {
                std::ostringstream oss;

                oss << "!!! The input matrix is NOT explicitly symmetric/Hermtian, ChASE cannot continue to proceed\n";
                oss << "Plase consider to use function symmetrizeOrHermitianizeMatrix() to symmetrize/Hermitianize at first\n";

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
    if(random)
    {   
         single->QR(0, 1.0);
    }
    // --------------------------------- LANCZOS
    // ---------------------------------

    std::size_t lanczos_iter = std::min(nevex, std::min(N / 2, config.GetLanczosIter()));
    if(single->isPseudoHerm() && 2.0*(lanczos_iter / 2) < lanczos_iter)
    {
	config.SetLanczosIter(lanczos_iter-1);
	lanczos_iter = config.GetLanczosIter();
	/* We enforce the number of ritz values to be even since the spectrum is symmetric
	 * in the Pseudo-Hermitian case */

    }
    lanczos(single, N, num_lanczos,lanczos_iter,
	    nevex, &upperb, random, random ? ritzv : NULL);

    std::size_t locked = 0;    // Number of converged eigenpairs.
    std::size_t iteration = 0; // Current iteration.
    lowerb = *std::max_element(ritzv, ritzv + unconverged); 
    lambda = *std::min_element(ritzv_, ritzv_ + nevex);
    Base<T> lambda_temp;
    lowerb = lowerb * config.GetDecayingRate();
    if(single->isPseudoHerm())
    {
        if(upperb > 0)
        {
            upperb = upperb * config.GetUpperbScaleRate();
        }else
        {
            upperb = upperb / config.GetUpperbScaleRate();
        }
        
    }

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
	else if (single->isPseudoHerm() && cnt == unconverged)
	{
        	lowerb = std::min(ritzv[unconverged - 1], Base<T>(0.0));
	}

        if (single->isSym() && unconverged < nevex)
        {
            lambda = *std::min_element(ritzv_, ritzv_ + nevex);
        }
        else if (single->isPseudoHerm() && unconverged < nevex)
        {
            // Check first few eigenvalues (up to 5) and find one with reasonable residual
            lambda = *std::min(ritzv, ritzv + unconverged);
            //lambda = ritzv[std::distance(resid,std::min_element(resid, resid + unconverged))];

            lowerb = std::min(ritzv[unconverged - 1], Base<T>(0.0));
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
        //-------------------------------- DEGREES
        //--------------------------------
        if (config.DoOptimization() && iteration != 0)
        {
            deg = calc_degrees(single, N, unconverged, nex, upperb, lowerb, tol,
                               ritzv, resid, residLast, degrees, locked);
        }

        //------------------------------- FILTER -------------------------------
	
	filter(single, N, unconverged, deg, degrees, lambda,
                                lowerb, upperb);

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

	if(single->isSym()){
        	for (auto i = 0; i < unconverged; ++i)
            		residLast[i] = std::min(residLast[i], resid[i]);
	}else{
        	for (auto i = 0; i < unconverged; ++i)
            		residLast[i] = resid[i];
	}

        single->Resd(ritzv, resid, locked);

#ifdef CHASE_OUTPUT
        {
            std::ostringstream oss;
            oss << "degrees\tresid\tresidLast\tritzv\n";
            for (std::size_t k = 0; k < std::min<std::size_t>(unconverged, 40);
                 ++k)
                oss << degrees[k] << "\t" << resid[k] << "\t" << residLast[k]
                    << "\t" << ritzv[k] << "\n";

            single->Output(oss.str());
        }

#endif

        std::size_t new_converged;

        if(single->isSym()){
            new_converged = locking(single, N, unconverged - single->isSym() * nex, tol, ritzv, resid, residLast,
                        degrees, locked);
        }else
        {
            new_converged = locking_quasi(single, N, unconverged, nex, tol, ritzv, resid, degrees, locked);
        }
        // ---------------------------- Update pointers
        // ---------------------------- Since we double buffer we need the
        // entire locked portion in W and V
        single->Lock(new_converged);

        locked += new_converged;
        unconverged -= new_converged;

        resid += new_converged;
        residLast += new_converged;
        ritzv += new_converged;
        degrees += new_converged;

        iteration++;
    } // while ( converged < nev && iteration < omp_maxiter )

    //---------------------SORT-EIGENPAIRS-ACCORDING-TO-EIGENVALUES---------------
    // Create a permutation vector to track the desired final positions of elements
    std::vector<std::size_t> perm(nev);
    // Initialize permutation with identity mapping (0 to n-1)
    std::iota(perm.begin(), perm.end(), 0);
    // Sort the permutation indices based on eigenvalue ordering
    // This creates a mapping from current positions to desired positions
    std::sort(perm.begin(), perm.end(), [&](std::size_t i, std::size_t j) {
        return ritzv_[i] < ritzv_[j];
    }); 
   
    // Track which elements have been processed to avoid infinite loops
    std::vector<bool> visited(perm.size(), false);

    // Process each element in the permutation
    for (std::size_t i = 0; i < perm.size(); ++i) {
        if (visited[i] || perm[i] == i) continue;

        // Cycle decomposition start
        std::size_t current = i;
        chase::Base<T> temp_ritz = ritzv_[i];
        chase::Base<T> temp_resid = resid_[i];

        std::vector<std::size_t> cycle_indices;

        // Collect cycle indices
        while (!visited[current]) {
            visited[current] = true;
            cycle_indices.push_back(current);
            current = perm[current];
        }

        // Rotate ritzv_ and resid_ scalars along cycle
        for (size_t k = 0; k < cycle_indices.size() - 1; ++k) {
            ritzv_[cycle_indices[k]] = ritzv_[cycle_indices[k + 1]];
            resid_[cycle_indices[k]] = resid_[cycle_indices[k + 1]];
        }
        ritzv_[cycle_indices.back()] = temp_ritz;
        resid_[cycle_indices.back()] = temp_resid;

        // Rotate eigenvectors along cycle by swapping adjacent pairs
        // For a cycle of indices like [a, b, c, d], do swaps:
        // Swap(a,b), Swap(b,c), Swap(c,d)
        for (size_t k = 0; k < cycle_indices.size() - 1; ++k) {
            single->Swap(cycle_indices[k], cycle_indices[k + 1]);
        }
    }    

    single->End();

}
} // namespace chase
