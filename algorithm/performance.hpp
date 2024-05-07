/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2023, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#ifndef CHASE_ALGORITHM_PERFORMANCE_HPP
#define CHASE_ALGORITHM_PERFORMANCE_HPP

#ifndef NO_MPI
#include <mpi.h>
#endif

#include <chrono>
#include <iostream>
#include <vector>

#include "interface.hpp"
#include "types.hpp"

namespace chase
{
//! ChASE class for collecting data relative to FLOPs, timings, etc.
/*! The ChasePerfData class collects and handles information relative to the
   execution of the eigensolver. It collects information about
    - Number of subspace iterations
    - Number of filtered vectors
    - Timings of each main algorithmic procedure (Lanczos, Filter, etc.)
    - Number of FLOPs executed

    The number of iterations and filtered vectors can be used to
    monitor the behavior of the algorithm as it attempts to converge
    all the desired eigenpairs. The timings and number of FLOPs are
    use to measure performance, especially parallel performance. The
    timings are stored in a vector of objects derived by the class
    template `std::chrono::duration`.
 */
template <class T>
class ChasePerfData
{
#ifdef HAS_CUDA
    using TimePointType = cudaEvent_t;
#else
    using TimePointType = std::chrono::time_point<std::chrono::high_resolution_clock>;
#endif

public:
    ChasePerfData()
        : chase_iteration_count(0), chase_filtered_vecs(0), timings(7),
           start_points(7), end_points(7), chase_iter_blocksizes(0)
    {
    }

    enum TimePtrs
    {
        All,
        InitVecs,
        Lanczos,
        Filter,
        Qr,
        Rr,
        Resids_Locking
    };

    void Reset()
    {
        chase_iteration_count = 0;
        chase_filtered_vecs = 0;
        chase_iter_blocksizes.clear();
        for(size_t i=0; i<start_points.size(); ++i)
        {
            start_points[i].erase(start_points[i].begin(), start_points[i].end());
            end_points[i].erase(end_points[i].begin(), end_points[i].end());
        }

        std::fill(timings.begin(), timings.end(),
                  std::chrono::duration<double>());
    }

    //! Returns the number of total subspace iterations executed by ChASE.
    /*! The *S* in ChASE stands for *Subspace iteration*. The main
        engine under the hood of ChASE is a *loop* enveloping all the
        main routines executed by the code. Because of this structure,
        ChASE is a truly iterative algorithm based on subspace
        filtering. Counting the number of times such a loop is repeated
        gives a measure of the effectiveness of the algorithm and it is
        usually a non-linear function of the spectral distribution. For
        example, when using the flag ``approximate_ = 'true'`` to solve a
        sequence of eigenproblems, one can observe that the number of
        subspace iteration decreases as a function of sequences index.
        \return The total number of subspace iterations.
     */
    std::size_t get_iter_count() { return chase_iteration_count; }

    //! Returns the cumulative number of times each column vector is filtered by
    //! one degree.
    /*! The most computationally expensive routine of ChASE is the
        Chebyshev filter. Within the filter a matrix of vectors *V* is
        filtered with a varying degree each time a subspace iteration is
        executed. This counter return the total number of times each
        vector in *V* goes through a filtering step. For instance, when
        the flag `optim_ = false `, such a number roughly corresponds to
        rank(V) x degree x iter_count. When the `optim_` is set
        to `true` such a calculation is quite more complicated. Roughly
        speaking, this counter is useful to monitor the convergence
        ration of the filtered vectors and together with
        `get_iter_count` convey the effectiveness of the algorithm.
        \return Cumulative number of filtered vectors.
     */
    std::size_t get_filtered_vecs() { return chase_filtered_vecs; }

    std::vector<std::chrono::duration<double>> get_timings() { return timings; }

    //! Returns the total number of FLOPs executed by ChASE.
    /*! When measuring performance, it is fundamental to understand how
        many operations a routine executes against the total time to
        solutions. This counter returns the total amount of operations
        executed by ChASE and can be used to extract the performance of
        ChASE and compare it with theoretical peak performance of the
        platform where the code is executed.
        \param N Size of the eigenproblem matrix
        \return The total number of operations executed by ChASE.
     */
    std::size_t get_flops(std::size_t N)
    {
        std::size_t flop_count = 0;
	int factor = std::pow(4, int(sizeof(T) / sizeof(Base<T>)) - 1) ;
        for (auto block : chase_iter_blocksizes)
        {
            // QR //
	    // assume always to use cholqr-2
	    // syrk
	    flop_count += 2. * N * block * block;
	    //cholesky factorizatin
	    flop_count += 2. * block * block * block;
	    // dtrsm
	    flop_count += 2. * N * block * block;
            
	    // RR //

            // W = H*V
            // https://software.intel.com/en-us/mkl-developer-reference-fortran-2018-beta-gemm
            // 8MNK + 18MN
            // m = N, k = N, n = block
            flop_count += 2 * N * block * N;

            // A = W' * W
            // 8MNK + 18MN
            // m = block, k = N, n = block
            flop_count += 2 * block * block * N;

	    // https://en.wikipedia.org/wiki/Divide-and-conquer_eigenvalue_algorithm
	    // 4M^3
	    flop_count += 4 * block * block * block;

            // W = V*Z
            // 2MNK
            flop_count += 2 * N * block * block;

            // resid //
            // W = H*V
            flop_count += 2 * N * block * N;

            // V[:,i] - lambda W[:,i]
            // 3*block
            flop_count += 3 * block * N;

            // ||V[:,i]||
            // N
            flop_count += N * block;
        }

        // filter
        // 8MNK + 18MN
        flop_count +=
            2 * N * chase_filtered_vecs * N;

	flop_count *= factor;

        return flop_count / 1e9;
    }

    //! Returns the total number of FLOPs of the Chebyshev filter
    /*! Similar to `get_flops`, this counter return the total number of
        operations executed by the Chebyshev filter alone. Since the
        filter is the routine that executes, on average, 80% of the
        total FLOPs of ChASE, this counter is a good indicator of the
        performance of the entire algorithm. Because the filter executes
        almost exclusively BLAS-3 operations, this counter is quite
        useful to monitor how well the filter is close to the peak
        performance of the platform where ChASE is executed. This can be
        quite useful to fine tune the use of the computational resources
        used.
        \param N Size of the eigenproblem matrix
        \return The total number of operations executed by the polynomial
       filter.
     */
    std::size_t get_filter_flops(std::size_t N)
    {
        int factor = std::pow(4, int(sizeof(T) / sizeof(Base<T>)) - 1) ;

        return 2 * factor * N * chase_filtered_vecs * N /
               1e9;
    }

    void set_nprocs(int nProcs) { nprocs = nProcs; }

    void add_iter_count(std::size_t add) { chase_iteration_count += add; }

    void add_iter_blocksize(std::size_t nevex)
    {
        chase_iter_blocksizes.push_back(nevex);
    }

    void add_filtered_vecs(std::size_t add) { chase_filtered_vecs += add; }

    void start_clock(TimePtrs t)
    {
        start_points[t].push_back(getTimePoint());
    }

    void end_clock(TimePtrs t)
    {
        end_points[t].push_back(getTimePoint());
    }

    inline auto getTimePoint() -> TimePointType
    {
#ifdef HAS_CUDA
        cudaEvent_t event;
        cudaEventCreate(&event);
        cudaEventRecord(event);
    return event;
#else
    return std::chrono::high_resolution_clock::now();
#endif
    }

    //! Print function outputting counters and timings for all routines
    /*! It prints by default ( for N = 0) in the order,
        - size of the eigenproblem
        - total number of subspace iterations executed
        - total number of filtered vectors
        - time-to-solution of the following 6 main sections of the ChASE
       algorithm:
           1. Total time-to-solution
           2. Estimates of the spectral bounds based on Lanczos,
           3. Chebyshev filter,
           4. QR decomposition,
           5. Raleygh-Ritz procedure including the solution of the reduced dense
       problem,
           6. Computation of the eigenpairs residuals

        When the parameter `N` is set to be a number else than zero, the
        function returns total FLOPs and filter FLOPs, respectively.
        \param N Control parameter. By default equal to *0*.
     */
    void print(std::size_t N = 0)
    {
        this->calculateTimings();


        std::vector<std::string> output_names = {"MPI procs", "Iterations", "Vecs", "All", "Init Vecs", 
                                                 "Lanczos",   "Filter",     "QR",   "RR",  "Resid"};

        std::vector<std::string> all_values = {std::to_string(nprocs), 
                                               std::to_string(chase_iteration_count), 
                                               std::to_string(chase_filtered_vecs)};
        
        for (const auto& t : timings)
        {
            std::ostringstream stream;
            stream << std::scientific << std::setprecision(3) << t.count();
            all_values.push_back(stream.str());
        }

        if (N != 0)
        {
            std::size_t flops = get_flops(N);
            std::size_t filter_flops = get_filter_flops(N);

            output_names.push_back("GFLOPS: All");
            output_names.push_back("GFLOPS: Filter");

            std::ostringstream stream;
            stream << std::scientific << std::setprecision(3) << static_cast<double>(flops);
            all_values.push_back(stream.str());
            stream.str("");
            stream << std::scientific << std::setprecision(3) << static_cast<double>(filter_flops);
            all_values.push_back(stream.str());
        }

        printTable(output_names, 
                   all_values);
    }

private:
    inline std::chrono::duration<double> calculatePointTiming(TimePointType &start, TimePointType &stop) 
    {
#ifdef HAS_CUDA
        float tmp_milisec = 0;
        cudaEventElapsedTime(&tmp_milisec, start, stop);
        return  std::chrono::duration<double>(tmp_milisec) /1000.0;
#else
        return stop - start;
#endif
    }

    void calculateTimings()
    {
#ifdef HAS_CUDA
        cudaDeviceSynchronize();
#endif
        for(size_t it=0; it<start_points.size(); ++it)
        {
            size_t vec_size = start_points[it].size();
            for(size_t vecit = 0; vecit < vec_size; ++vecit)
            {
                timings[it] += calculatePointTiming(start_points[it][vecit], end_points[it][vecit]);
            }
        }
    }

    void printTable(const std::vector<std::string>& output_names, 
                    const std::vector<std::string>& all_values)
    {
        // Print table header and values
        std::cout << "| ";
        for (size_t i = 0; i < output_names.size(); ++i) {
            int max_width = std::max(output_names[i].size(), all_values[i].size());
            std::cout << std::setw(max_width) << output_names[i] << " | ";
        }
        std::cout << '\n';

        std::cout << "| ";
        for (size_t i = 0; i < all_values.size(); ++i) {
            int max_width = std::max(output_names[i].size(), all_values[i].size());
            std::cout << std::setw(max_width) << all_values[i] << " | ";
        }
        std::cout << '\n';
}

    std::size_t chase_iteration_count;
    std::size_t chase_filtered_vecs;
    std::vector<std::size_t> chase_iter_blocksizes;

    std::vector<std::chrono::duration<double>> timings;
    std::vector<std::vector<TimePointType>> start_points;
    std::vector<std::vector<TimePointType>> end_points;
    int nprocs;
};
//! A derived class used to extract performance and configuration data.
/*! This is a class derived from the Chase class which plays the
    role of interface for the kernels used by the library. All
    members of the Chase class are virtual functions. These
    functions are re-implemented in the PerformanceDecoratorChase
    class. All derived members that provide an interface to
    computational kernels are reimplemented by *decorating* the
    original function with time pointers which are members of the
    ChasePerfData class. All derived members that provide an
    interface to input or output data are called without any
    specific decoration. In addition to the virtual member of the
    Chase class, the PerformanceDecoratorChase class has also among
    its public members a reference to an object of type
    ChasePerfData. When using Chase to solve an eigenvalue problem,
    the members of the PerformanceDecoratorChase are called instead
    of the virtual functions members of the Chase class. In this
    way, all parameters and counters are automatically invoked and
    returned in the correct order.
    \see Chase
 */
template <class T>
class PerformanceDecoratorChase : public chase::Chase<T>
{
public:
    PerformanceDecoratorChase(Chase<T>* chase) : chase_(chase), perf_() {}

    void initVecs(bool random)
    {   
        perf_.start_clock(ChasePerfData<T>::TimePtrs::InitVecs);
        chase_->initVecs(random);
        perf_.end_clock(ChasePerfData<T>::TimePtrs::InitVecs);
    }

    void Shift(T c, bool isunshift = false)
    {
        if (isunshift)
            perf_.end_clock(ChasePerfData<T>::TimePtrs::Filter);
        else
            perf_.start_clock(ChasePerfData<T>::TimePtrs::Filter);

        chase_->Shift(c, isunshift);
    }
    void HEMM(std::size_t nev, T alpha, T beta, std::size_t offset)
    {
        chase_->HEMM(nev, alpha, beta, offset);
        perf_.add_filtered_vecs(nev);
    }

    void QR(std::size_t fixednev, Base<T> cond)
    {
        perf_.start_clock(ChasePerfData<T>::TimePtrs::Qr);
        chase_->QR(fixednev, cond);
        perf_.end_clock(ChasePerfData<T>::TimePtrs::Qr);
    }

    void RR(Base<T>* ritzv, std::size_t block)
    {
        perf_.start_clock(ChasePerfData<T>::TimePtrs::Rr);
        chase_->RR(ritzv, block);
        perf_.add_iter_blocksize(block);
        perf_.end_clock(ChasePerfData<T>::TimePtrs::Rr);
    }
    void Resd(Base<T>* ritzv, Base<T>* resd, std::size_t fixednev)
    {
        perf_.start_clock(ChasePerfData<T>::TimePtrs::Resids_Locking);
        chase_->Resd(ritzv, resd, fixednev);
        // We end with ->chase_->Lock()
    }
    void Lanczos(std::size_t m, Base<T>* upperb)
    {
        std::chrono::high_resolution_clock::time_point t1, t2;
        std::chrono::duration<double> elapsed;
        t1 = std::chrono::high_resolution_clock::now();
        perf_.start_clock(ChasePerfData<T>::TimePtrs::Lanczos);
        chase_->Lanczos(m, upperb);
        t2 = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        std::cout << "Lanczos timing = " << elapsed.count() << std::endl;
        perf_.end_clock(ChasePerfData<T>::TimePtrs::Lanczos);
    }
    void Lanczos(std::size_t M, std::size_t idx, Base<T>* upperb,
                 Base<T>* ritzv, Base<T>* Tau, Base<T>* ritzV)
    {
        perf_.start_clock(ChasePerfData<T>::TimePtrs::Lanczos);
        chase_->Lanczos(M, idx, upperb, ritzv, Tau, ritzV);
        perf_.end_clock(ChasePerfData<T>::TimePtrs::Lanczos);

    }
    void LanczosDos(std::size_t idx, std::size_t m, T* ritzVc)
    {
        perf_.start_clock(ChasePerfData<T>::TimePtrs::Lanczos);
        chase_->LanczosDos(idx, m, ritzVc);
        perf_.end_clock(ChasePerfData<T>::TimePtrs::Lanczos);
    }

    void Swap(std::size_t i, std::size_t j) { chase_->Swap(i, j); }
    void Lock(std::size_t new_converged)
    {
        chase_->Lock(new_converged);
        perf_.end_clock(ChasePerfData<T>::TimePtrs::Resids_Locking);
        perf_.add_iter_count(1);
    }

    bool checkSymmetryEasy()
    {
        return chase_->checkSymmetryEasy();
    }

    void symOrHermMatrix(char uplo)
    {
        chase_->symOrHermMatrix(uplo);
    }

    void Start()
    {
        chase_->Start();
        perf_.Reset();
        perf_.start_clock(ChasePerfData<T>::TimePtrs::All);
        perf_.set_nprocs(chase_->get_nprocs());
    }

    int get_nprocs() { return chase_->get_nprocs(); }

    void End()
    {
        chase_->End();
        perf_.end_clock(ChasePerfData<T>::TimePtrs::All);
    }

    std::size_t GetN() const { return chase_->GetN(); }
    std::size_t GetNev() { return chase_->GetNev(); }
    std::size_t GetNex() { return chase_->GetNex(); }
    Base<T>* GetRitzv() { return chase_->GetRitzv(); }
    Base<T>* GetResid() { return chase_->GetResid(); }
    ChaseConfig<T>& GetConfig() { return chase_->GetConfig(); }
    ChasePerfData<T>& GetPerfData() { return perf_; }

#ifdef CHASE_OUTPUT
    void Output(std::string str) { chase_->Output(str); }
#endif

private:
    Chase<T>* chase_;
    ChasePerfData<T> perf_;
};

} // namespace chase
#endif
