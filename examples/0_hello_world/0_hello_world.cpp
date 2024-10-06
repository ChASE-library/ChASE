#include <iostream>
#include <vector>
#include <complex>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

#include "algorithm/performance.hpp"
#ifdef HAS_CUDA
#include "Impl/nccl/chase_nccl_gpu.hpp"
#else
#include "Impl/mpi/chase_mpi_cpu.hpp"
#endif

using T = std::complex<double>;
using namespace chase;

#ifdef HAS_CUDA
using ARCH = chase::platform::GPU;
typedef chase::Impl::ChaseNCCLGPU<T> ChaseImpl;
#else
using ARCH = chase::platform::CPU;
typedef chase::Impl::ChaseMPICPU<T> ChaseImpl;
#endif

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);  

    std::size_t N = 1200;
    std::size_t nev = 80;
    std::size_t nex = 60;
    std::size_t idx_max = 3;
    Base<T> perturb = 1e-4;
    
    std::mt19937 gen(1337.0);
    std::normal_distribution<> d;

    int dims_[2];
    dims_[0] = dims_[1] = 0;
    // MPI proc grid = dims[0] x dims[1]
    MPI_Dims_create(world_size, 2, dims_);

    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
        = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(dims_[0], dims_[1], MPI_COMM_WORLD);

    int *dims = mpi_grid.get()->get_dims();
    int *coords = mpi_grid.get()->get_coords();

    std::size_t m, n;

    if (N % dims[0] == 0)
    {
        m = N / dims[0];
    }
    else
    {
        m = std::min(N, N / dims[0] + 1);
    }

    if(coords[0] == dims[0] - 1)
    {
        m = N - (dims[0] - 1) * m;
    }

    if (N % dims[1] == 0)
    {
        n = N / dims[1];
    }
    else
    {
        n = std::min(N, N / dims[1] + 1);
    }

    if(coords[1] == dims[1] - 1)
    {
        n = N - (dims[1] - 1) * n;
    }
    

    if(world_rank == 0)
    {
        std::cout << "ChASE example driver\n"
                << "Usage: ./driver \n";
    }

    std::vector<T> H(m * n);
    auto V = std::vector<T>(m * (nev + nex));
    auto Lambda = std::vector<chase::Base<T>>(nev + nex);

    auto Hmat = chase::distMatrix::BlockBlockMatrix<T, ARCH>(m, n, m, H.data(), mpi_grid);    
    auto Vec = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, ARCH>(m, nev + nex, m, V.data(), mpi_grid);
    //redistribute Clement matrix into block-block distribution
    std::size_t *g_offs = Hmat.g_offs();

    for (auto i = 0; i < N; ++i)
    {
        if (i != N - 1)
        {
            auto x = i + 1 - g_offs[0];
            auto y = i - g_offs[1];
            if(x >= 0 && y >= 0 && x < m && y < n)
            {
                H[x + y * m] = std::sqrt(i * (N + 1 - i));
            }
        }
        if (i != N - 1)
        {
            auto x = i - g_offs[0];
            auto y = i + 1 - g_offs[1];
            if(x >= 0 && y >= 0 && x < m && y < n)
            {
                H[x + y * m] = std::sqrt(i * (N + 1 - i));
            }            
        }
    }   

#ifdef HAS_CUDA
        Hmat.H2D();
#endif    
    ChaseImpl single(nev, nex, &Hmat, &Vec, Lambda.data());

//#ifdef HAS_CUDA
//    chase::Impl::ChaseNCCLGPU<T> single(nev, nex, &Hmat, &Vec, Lambda.data());
//#else
//    chase::Impl::ChaseMPICPU<T> single(nev, nex, &Hmat, &Vec, Lambda.data());
//#endif
    //Setup configure for ChASE
    auto& config = single.GetConfig();
    //Tolerance for Eigenpair convergence
    config.SetTol(1e-10);
    //Initial filtering degree
    config.SetDeg(20);
    //Optimi(S)e degree
    config.SetOpt(true);
    config.SetMaxIter(25);

    if (world_rank == 0)
        std::cout << "Solving a symmetrized Clement matrices (" << N << "x" << N
                << ")"
                << '\n'
                << config;

    for (auto idx = 0; idx < idx_max; ++idx)
    {
        if (world_rank == 0)
        {
            std::cout << "Starting Problem #" << idx << "\n";
        }
        if (config.UseApprox())
        {
            if (world_rank == 0) std::cout << "Using approximate solution\n";
        }
        
        //Performance Decorator to meaure the performance of kernels of ChASE
        PerformanceDecoratorChase<T> performanceDecorator(&single);
        //Solve the eigenproblem
        chase::Solve(&performanceDecorator);

        //Output
        if (world_rank == 0)
        {
            performanceDecorator.GetPerfData().print();
            Base<T>* resid = single.GetResid();
            std::cout << "Finished Problem #1"
                    << "\n";
            std::cout << "Printing first 5 eigenvalues and residuals\n";
            std::cout
                << "| Index |       Eigenvalue      |         Residual      |\n"
                << "|-------|-----------------------|-----------------------|\n";
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
        }

        config.SetApprox(true);
        // Perturb Full Clement matrix
        for (std::size_t i = 1; i < N; ++i)
        {
            for (std::size_t j = 1; j < i; ++j)
            {
                T element_perturbation = T(d(gen), d(gen)) * perturb;                
                auto x = i - g_offs[0];
                auto y = j - g_offs[1];
                auto y_2 = i - g_offs[1];
                auto x_2 = j - g_offs[0];

                if(x >= 0 && y >= 0 && x < m && y < n)
                {
                    H[x + y * m] += element_perturbation;

                }

                if(x_2 >= 0 && y_2 >= 0 && x_2 < m && y_2 < n)
                {
                    H[x_2 + y_2 * m] += std::conj(element_perturbation);

                }
            }
        }
#ifdef HAS_CUDA        
        Hmat.H2D();
#endif        
    }
    
    MPI_Finalize();   
}
