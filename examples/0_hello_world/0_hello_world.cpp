#include <iostream>
#include <vector>
#include <complex>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

#include "algorithm/performance.hpp"
#include "Impl/chase_mpi_cpu.hpp"

using T = std::complex<double>;
using namespace chase;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);  

    std::size_t N = 1200;
    std::size_t nev = 120;
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

    auto Clement = chase::distMatrix::RedundantMatrix<T>(N, N, mpi_grid);
    std::vector<T> H(m * n);
    auto V = std::vector<T>(m * (nev + nex));
    auto Lambda = std::vector<chase::Base<T>>(nev + nex);

    //Generate Clement matrix
    for (auto i = 0; i < N; ++i)
    {
        Clement.l_data()[i + N * i] = 0;
        if (i != N - 1)
            Clement.l_data()[i + 1 + N * i] = std::sqrt(i * (N + 1 - i));
        if (i != N - 1)
            Clement.l_data()[i + N * (i + 1)] = std::sqrt(i * (N + 1 - i));
    }

    auto Hmat = chase::distMatrix::BlockBlockMatrix<T>(m, n, m, H.data(), mpi_grid);
    auto Vec = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(m, nev + nex, m, V.data(), mpi_grid);
    //redistribute Clement matrix into block-block distribution

    chase::Impl::ChaseMPICPU<T> single(nev, nex, &Hmat, &Vec, Lambda.data());

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

        Clement.redistributeImpl<chase::distMatrix::MatrixTypeTrait<decltype(Hmat)>::value>(&Hmat);

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
                Clement.l_data()[j + N * i] += element_perturbation;
                Clement.l_data()[i + N * j] += std::conj(element_perturbation);
            }
        }        
    }
    MPI_Finalize();   
}