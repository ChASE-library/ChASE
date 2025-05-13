// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <complex>
#include <iostream>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

#include "algorithm/performance.hpp"
#ifdef HAS_CUDA
#include "Impl/pchase_gpu/pchase_gpu.hpp"
#else
#include "Impl/pchase_cpu/pchase_cpu.hpp"
#endif

using T = std::complex<double>;
using namespace chase;

#ifdef HAS_CUDA
using ARCH = chase::platform::GPU;
using BackendType = chase::grid::backend::NCCL;
#else
using ARCH = chase::platform::CPU;
#endif

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int dims_[2];
    dims_[0] = dims_[1] = 0;

    MPI_Dims_create(world_size, 2, dims_);

    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        mpi_grid = std::make_shared<
            chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
            dims_[0], dims_[1], MPI_COMM_WORLD);

    size_t k = 1472;

    size_t N = 2 * k, nev = 200, nex = 100, mb = 40;

    int* dims = mpi_grid.get()->get_dims();
    int* coords = mpi_grid.get()->get_coords();

    if (world_rank == 0)
    {
        std::cout << "Matrix Size = " << N << std::endl;
        std::cout << "World  Size = " << world_size << std::endl;
        std::cout << "Dims        = " << dims[0] << " x " << dims[1]
                  << std::endl;
    }

    auto Hmat = chase::distMatrix::QuasiHermitianBlockBlockMatrix<T, ARCH>(
        N, N, mpi_grid);

    #ifdef HAS_CUDA
    	Hmat.allocate_cpu_data();
    #endif

/*
    auto Hmat = chase::distMatrix::QuasiHermitianBlockCyclicMatrix<T, ARCH>(
        N, N, mb, mb, mpi_grid);
*/
    Hmat.readFromBinaryFile(
	"../../../Data/Matrix/2x2x2_Silicon_QuasiHermitian.bin");
    //    "./tests/linalg/internal/BSE_matrices/cdouble_random_BSE.bin");

    auto Lambda = std::vector<chase::Base<T>>(nev + nex);
    
    auto Vec = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column, ARCH>(N, nev + nex, mpi_grid);

    /*auto Vec = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::column, ARCH>(N, nev + nex,mb, mpi_grid);
    */
    if (world_rank == 0)
    {
#ifdef HAS_CUDA
        std::cout
            << "======================== ChASE GPU ========================"
            << std::endl;
#else
        std::cout
            << "======================== ChASE CPU ========================"
            << std::endl;
#endif
    }

#ifdef HAS_CUDA
    Hmat.allocate_cpu_data();
    auto single =
        chase::Impl::pChASEGPU<decltype(Hmat), decltype(Vec), BackendType>(
            nev, nex, &Hmat, &Vec, Lambda.data());
#else
    auto single = chase::Impl::pChASECPU(nev, nex, &Hmat, &Vec, Lambda.data());
#endif

    single.initVecs(true);

    // Setup configure for ChASE
    auto& config = single.GetConfig();
    // Tolerance for Eigenpair convergence
    config.SetTol(1e-10);
    // Initial filtering degree
    config.SetDeg(10);
    // Optimi(S)e degree
    config.SetOpt(true);
    config.SetMaxIter(25);

    PerformanceDecoratorChase<T> performanceDecorator(&single);

    chase::Solve(&performanceDecorator);

    // Output
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

    MPI_Finalize();
}
