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

#ifdef HAS_NCCL 
#include "Impl/pchase_gpu/pchase_gpu.hpp"
#elif defined(HAS_CUDA)
#include "Impl/chase_gpu/chase_gpu.hpp"
#elif defined(USE_MPI)
#include "Impl/pchase_cpu/pchase_cpu.hpp"
#else
#include "Impl/chase_cpu/chase_cpu.hpp"
#endif

using T = std::complex<double>;
using namespace chase;

#if defined(HAS_CUDA) || defined(HAS_NCCL)
using ARCH = chase::platform::GPU;
#else
using ARCH = chase::platform::CPU;
#endif

#ifdef HAS_NCCL
using BackendType = chase::grid::backend::NCCL;
#endif

int main(int argc, char** argv)
{
    size_t k = 1472;
    size_t N = 2 * k, nev = 200, nex = 100;

#if defined(USE_MPI) || defined(HAS_NCCL)

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

    int* dims = mpi_grid.get()->get_dims();
    int* coords = mpi_grid.get()->get_coords();

    if (world_rank == 0)
    {

#endif
#if defined(USE_MPI) || defined(HAS_NCCL) //Parallel Implementation //A
    std::cout << "======================== Dist ChASE ";

#if defined(USE_BLOCKCYCLIC) 
    std::cout << "BlockCyclic ";
#else
    std::cout << "BlockBlock ";
#endif

#else
    std::cout << "======================== Seq  ChASE ";
#endif

#if defined(USE_PSEUDO_HERMITIAN) //C
    std::cout << "Pseudo-Hermitian ";
#else //C
    std::cout << "Hermitian ";
#endif //C

    std::cout << "Matrix ";

#ifdef HAS_CUDA//B
    std::cout <<  "on GPU ========================" << std::endl;
#else
    std::cout <<  "on CPU ========================" << std::endl;
#endif //B
#if defined(USE_MPI) || defined(HAS_NCCL)
    }

    MPI_Barrier(MPI_COMM_WORLD);
#endif
    auto Lambda = std::vector<chase::Base<T>>(nev + nex);

#if defined(USE_MPI) || defined(HAS_NCCL) 
    
#ifdef USE_BLOCKCYCLIC 
    std::size_t mb = 64;
    auto Vec = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::column, ARCH>(N, nev + nex, mb, mpi_grid);
#ifdef USE_PSEUDO_HERMITIAN 
    auto Hmat = chase::distMatrix::PseudoHermitianBlockCyclicMatrix<T, ARCH>(
        N, N, mb, mb, mpi_grid);
#else 
    auto Hmat = chase::distMatrix::BlockCyclicMatrix<T, ARCH>(
        N, N, mb, mb, mpi_grid);
#endif 
#else
    auto Vec = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column, ARCH>(N, nev + nex, mpi_grid);
#ifdef USE_PSEUDO_HERMITIAN
    auto Hmat = chase::distMatrix::PseudoHermitianBlockBlockMatrix<T, ARCH>(
        N, N, mpi_grid);
#else 
    auto Hmat = chase::distMatrix::BlockBlockMatrix<T, ARCH>(
        N, N, mpi_grid);
#endif 
#endif 
#ifdef HAS_NCCL
    auto single = chase::Impl::pChASEGPU<decltype(Hmat), decltype(Vec), BackendType>(
            nev, nex, &Hmat, &Vec, Lambda.data());
	
    Hmat.allocate_cpu_data();
#else
    auto single = chase::Impl::pChASECPU(nev, nex, &Hmat, &Vec, Lambda.data());
#endif 
    Hmat.readFromBinaryFile("../../../Data/Matrix/2x2x2_Silicon_QuasiHermitian.bin");

#ifdef HAS_NCCL
    Hmat.H2D();
#endif

#else 
    std::vector<T> V(N*(nev+nex));

#ifdef USE_PSEUDO_HERMITIAN 
    using MatrixType = chase::matrix::PseudoHermitianMatrix<T, ARCH>;
#else //B
    using MatrixType = chase::matrix::Matrix<T,ARCH>;
#endif //B
    
    auto Hmat = new MatrixType(N,N);

#ifdef HAS_CUDA 
    auto single = chase::Impl::ChASEGPU<T,MatrixType>(N, nev, nex, Hmat, V.data(), N, Lambda.data());
    	
    Hmat->allocate_cpu_data();
#else 
    auto single = chase::Impl::ChASECPU<T,MatrixType>(N, nev, nex, Hmat, V.data(), N, Lambda.data());
#endif
	
    Hmat->readFromBinaryFile("../../../Data/Matrix/2x2x2_Silicon_QuasiHermitian.bin");

#ifdef HAS_CUDA
    Hmat->H2D();
#endif

#endif

    std::cout << std::endl;
    
    
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
    config.SetLanczosIter(26);

    PerformanceDecoratorChase<T> performanceDecorator(&single);

    chase::Solve(&performanceDecorator);

#if defined(USE_MPI) || defined(HAS_NCCL)
    // Output
    if (world_rank == 0)
    {
#endif
        performanceDecorator.GetPerfData().print();
        Base<T>* resid = single.GetResid();
        std::cout << "Finished Problem #1"
                  << "\n";
        std::cout << "Printing first 10 eigenvalues and residuals\n";
        std::cout
            << "| Index |       Eigenvalue      |         Residual      |\n"
            << "|-------|-----------------------|-----------------------|\n";
        std::size_t width = 20;
        std::cout << std::setprecision(12);
        std::cout << std::setfill(' ');
        std::cout << std::scientific;
        std::cout << std::right;
        for (auto i = 0; i < std::min(std::size_t(10), nev); ++i)
            std::cout << "|  " << std::setw(4) << i + 1 << " | "
                      << std::setw(width) << Lambda[i] << "  | "
                      << std::setw(width) << resid[i] << "  |\n";
        std::cout << "\n\n\n";
#if defined(USE_MPI) || defined(HAS_NCCL)
    }

    MPI_Finalize();

#endif
}
