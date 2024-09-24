#include <iostream>
#include <vector>
#include <complex>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

#include "algorithm/performance.hpp"
#include "Impl/chase_seq_cpu.hpp"

using T = std::complex<double>;
using namespace chase;

int main()
{
    std::size_t N = 1200;
    std::size_t LDH = 1200;
    std::size_t nev = 80;
    std::size_t nex = 60;
    std::size_t idx_max = 3;
    Base<T> perturb = 1e-4;
    
    std::mt19937 gen(1337.0);
    std::normal_distribution<> d;
   
    std::cout << "ChASE example driver\n"
            << "Usage: ./driver \n";

    auto V = std::vector<T>(N * (nev + nex));
    auto Lambda = std::vector<Base<T>>(nev + nex);
    std::vector<T> H(N * LDH, T(0.0));

    chase::Impl::ChaseCPUSeq<T> single(N, nev, nex, H.data(), LDH, V.data(), N, Lambda.data());

    auto& config = single.GetConfig();
    config.SetTol(1e-10);
    config.SetDeg(20);
    config.SetOpt(true);
    config.SetApprox(false);

    std::cout << "Solving " << idx_max << " symmetrized Clement matrices ("
                << N << "x" << N
                << ") with element-wise random perturbation of " << perturb
                << '\n'
                << config;    

    // Generate Clement matrix
    for (auto i = 0; i < N; ++i)
    {
        H[i + N * i] = 0;
        if (i != N - 1)
            H[i + 1 + LDH * i] = std::sqrt(i * (N + 1 - i));
        if (i != N - 1)
            H[i + LDH * (i + 1)] = std::sqrt(i * (N + 1 - i));
    }

    for (auto idx = 0; idx < idx_max; ++idx)
    {

        std::cout << "Starting Problem #" << idx << "\n";
        if (config.UseApprox())
        {
            std::cout << "Using approximate solution\n";
        }
    
        PerformanceDecoratorChase<T> performanceDecorator(&single);
        chase::Solve(&performanceDecorator);

        performanceDecorator.GetPerfData().print();
        Base<T>* resid = single.GetResid();
        std::cout << "Finished Problem #" << idx << "\n";
        std::cout << "Printing first 5 eigenvalues and residuals\n";
        std::cout
            << "| Index |       Eigenvalue      |         Residual      |\n"
            << "|-------|-----------------------|-----------------------|"
                "\n";
        std::size_t width = 20;
        std::cout << std::setprecision(12);
        std::cout << std::setfill(' ');
        std::cout << std::scientific;
        std::cout << std::right;
        for (auto i = 0; i < std::min(std::size_t(5), nev); ++i)
            std::cout << "|  " << std::setw(4) << i + 1 << " | "
                        << std::setw(width) << Lambda[i] << "  | "
                        << std::setw(width) << resid[i] << "  |\n";
        std::cout << "\n\n\n";
        
        config.SetApprox(true);
        // Perturb Full Clement matrix
        for (std::size_t i = 1; i < N; ++i)
        {
            for (std::size_t j = 1; j < i; ++j)
            {
                T element_perturbation = T(d(gen), d(gen)) * perturb;
                H[j + LDH * i] += element_perturbation;
                H[i + LDH * j] += std::conj(element_perturbation);
            }
        }
    }
}

