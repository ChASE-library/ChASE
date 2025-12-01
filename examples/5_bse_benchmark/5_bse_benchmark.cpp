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
#include "popl.hpp"

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

using namespace chase;

#if defined(HAS_CUDA) || defined(HAS_NCCL)
using ARCH = chase::platform::GPU;
#else
using ARCH = chase::platform::CPU;
#endif

#ifdef HAS_NCCL
using BackendType = chase::grid::backend::NCCL;
#endif

using namespace popl;

struct BSE_DriverProblemConfig
{
    std::size_t N; // Size of the Matrix
    std::size_t nev; // Number of sought after eigenvalues
    std::size_t nex; // Extra size of subspace
    std::size_t deg; // initial degree

    float tol_f; // desired tolerance
    double tol_d; // desired tolerance
    std::size_t maxIter; // maximum number of subspace iterations within ChASE.
    std::size_t maxDeg; // maximum value of the degree of the Chebyshev filter
    std::string opt; // enable optimisation of degree
    std::string mode; // Approx or Random mode
    std::size_t lanczosIter; // number of lanczos iterations
    std::size_t numLanczos; // number of lanczos vectors
    std::size_t blocksize; // blocksize
    float lowerb_decay; // number of processes
    std::size_t extraDeg; // added value of the degree of the Chebyshev filter

    std::string path_in; // path to the matrix input files
    bool isdouble;       
};

template <typename T>
int bse_solve(BSE_DriverProblemConfig& conf)
{
    std::size_t N = conf.N;
    std::size_t nev = conf.nev;
    std::size_t nex = conf.nex;
    std::size_t deg = conf.deg;
    std::size_t maxDeg = conf.maxDeg;
    std::size_t maxIter = conf.maxIter;
    std::size_t extraDeg = conf.extraDeg;
    chase::Base<T> tol; // desired tolerance
    std::string opt = conf.opt;
    std::string mode = conf.mode;
    std::size_t mb = conf.blocksize;
    float lowerb_decay = conf.lowerb_decay;
    if (conf.isdouble)
    {
        tol = conf.tol_d;
    }
    else
    {
        tol = conf.tol_f;
    }

    std::size_t lanczosIter = conf.lanczosIter;
    std::size_t numLanczos = conf.numLanczos;

    std::string path_in = conf.path_in;
    int world_rank = 0;

#if defined(USE_MPI) || defined(HAS_NCCL)
    //setup MPI grid
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
#endif

    if (world_rank == 0)
    {
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

#if defined(USE_QUASI_HERMITIAN) //C
    std::cout << "Quasi-Hermitian ";
#else //C
    std::cout << "Hermitian ";
#endif //C

    std::cout << "Matrix ";

#ifdef HAS_CUDA//B
    std::cout <<  "on GPU ========================" << std::endl;
#else
    std::cout <<  "on CPU ========================" << std::endl;
#endif //B
  
    }

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;
    
    auto Lambda = std::vector<chase::Base<T>>(nev + nex);

#if defined(USE_MPI) || defined(HAS_NCCL) 
    
#ifdef USE_BLOCKCYCLIC 
    auto Vec = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::column, ARCH>(N, nev + nex, mb, mpi_grid);
#ifdef USE_QUASI_HERMITIAN 
    auto Hmat = chase::distMatrix::QuasiHermitianBlockCyclicMatrix<T, ARCH>(
        N, N, mb, mb, mpi_grid);
#else 
    auto Hmat = chase::distMatrix::BlockCyclicMatrix<T, ARCH>(
        N, N, mb, mb, mpi_grid);
#endif 
#else
    auto Vec = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column, ARCH>(N, nev + nex, mpi_grid);
#ifdef USE_QUASI_HERMITIAN
    auto Hmat = chase::distMatrix::QuasiHermitianBlockBlockMatrix<T, ARCH>(
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

    start = std::chrono::high_resolution_clock::now();
    Hmat.readFromBinaryFile(path_in);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    if (world_rank == 0)
    {
        std::cout << "Time taken to read matrix: " << elapsed.count() << " seconds" << std::endl;
    }
#ifdef HAS_NCCL
    Hmat.H2D();
#endif

#else 
    std::vector<T> V(N*(nev+nex));

#ifdef USE_QUASI_HERMITIAN 
    using MatrixType = chase::matrix::QuasiHermitianMatrix<T, ARCH>;
#else //B
    using MatrixType = chase::matrix::Matrix<T,ARCH>;
#endif //B
    
    auto Hmat = new MatrixType(N, N);

#ifdef HAS_CUDA 
    auto single = chase::Impl::ChASEGPU<T,MatrixType>(N, nev, nex, Hmat, V.data(), N, Lambda.data());
    	
    Hmat->allocate_cpu_data();
#else 
    auto single = chase::Impl::ChASECPU<T,MatrixType>(N, nev, nex, Hmat, V.data(), N, Lambda.data());
#endif
    start = std::chrono::high_resolution_clock::now();
    Hmat->readFromBinaryFile(path_in);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    if (world_rank == 0)
    {
        std::cout << "Time taken to read matrix: " << elapsed.count() << " seconds" << std::endl;
    }

#ifdef HAS_CUDA
    Hmat->H2D();
#endif

#endif

#ifndef USE_QUASI_HERMITIAN
    //single.symOrHermMatrix('L');
/*	
#ifdef HAS_NCCL
    	//chase::linalg::internal::cuda_nccl::flipLowerHalfMatrixSign(Hmat);
	//Hmat.D2H();
#elif defined(HAS_CUDA)
    	//chase::linalg::internal::cuda::flipLowerHalfMatrixSign(Hmat);
	//Hmat->D2H();
#elif defined(USE_MPI)
    	chase::linalg::internal::cpu_mpi::flipLowerHalfMatrixSign(Hmat);
#else
    	chase::linalg::internal::cpu::flipLowerHalfMatrixSign(Hmat);
#endif*/
#endif

    if (world_rank == 0)
    {
        std::cout << "Starting ChASE" << std::endl;
    }

    auto& config = single.GetConfig();
    config.SetTol(tol);
    config.SetDeg(deg);
    config.SetOpt(opt == "S");
    config.SetMaxIter(maxIter);
    config.SetLanczosIter(lanczosIter);
    config.SetNumLanczos(numLanczos);
    config.SetMaxDeg(maxDeg);
    config.SetDegExtra(extraDeg);
    config.SetApprox(mode == "A");
    config.SetDecayingRate(lowerb_decay);
    config.SetClusterAwareDegrees(true);
    
    PerformanceDecoratorChase<T> performanceDecorator(&single);

    chase::Solve(&performanceDecorator);

    if (world_rank == 0)
    {
        performanceDecorator.GetPerfData().print(N,lanczosIter,numLanczos);
        Base<T>* resid = single.GetResid();
        std::cout << "Finished Problem #"
                  << "\n";
        std::cout << "Printing first ALL eigenvalues and residuals\n";
        std::cout
            << "| Index |       Eigenvalue      |         Residual      |\n"
            << "|-------|-----------------------|-----------------------|\n";
        std::size_t width = 20;
        std::cout << std::setprecision(12);
        std::cout << std::setfill(' ');
        std::cout << std::scientific;
        std::cout << std::right;
        for (auto i = 0; i < std::min(std::size_t(nev+nex), nev+nex); ++i)
            std::cout << "|  " << std::setw(4) << i + 1 << " | "
                      << std::setw(width) << Lambda[i] << "  | "
                      << std::setw(width) << resid[i] << "  |\n";
        std::cout << "\n\n\n";
    }
    return 0;
    
}

int main(int argc, char** argv)
{
#if defined(USE_MPI) || defined(HAS_NCCL)

    MPI_Init(&argc, &argv);
#endif

    BSE_DriverProblemConfig conf;

    popl::OptionParser desc("ChASE options");
    auto help_option = desc.add<Switch>("h", "help", "show this message");
    desc.add<Value<std::size_t>, Attribute::required>("", "n", "Size of the Input Matrix", 0, &conf.N);
    desc.add<Value<std::size_t>, Attribute::required>("", "nev", "Wanted Number of Eigenpairs", 0, &conf.nev);
    desc.add<Value<std::size_t>>("", "nex", "Extra Search Dimensions", 25, &conf.nex);
    desc.add<Value<std::size_t>>("", "deg", "Initial filtering degree", 20, &conf.deg);
    desc.add<Value<std::size_t>>("", "maxDeg", "Sets the maximum value of the degree of the Chebyshev filter", 36, &conf.maxDeg);
    desc.add<Value<std::size_t>>("", "extraDeg", "Increase the optimized degree of the Chebyshev filter by extraDeg", 0, &conf.extraDeg);
    desc.add<Value<std::string>>("", "opt", "Optimi(S)e degree, or do (N)ot optimise", "S", &conf.opt);
    desc.add<Value<std::string>>("", "mode", "Approximate (A) or Random (R)", "R", &conf.mode);
    desc.add<Value<std::size_t>>("", "lanczosIter", "Sets the number of Lanczos iterations executed by ChASE.", 26, &conf.lanczosIter);
    desc.add<Value<std::size_t>>("", "numLanczos", "Sets the number of stochastic vectors used for the spectral estimates in Lanczos", 4, &conf.numLanczos);
    desc.add<Value<float>>("", "tol_f", "Sets the desired tolerance for float precision", 1e-5, &conf.tol_f);
    desc.add<Value<double>>("", "tol_d", "Sets the desired tolerance for double precision", 1e-10, &conf.tol_d);
    desc.add<Value<std::size_t>>("", "maxIter", "Sets the maximum number of subspace iterations within ChASE.", 25, &conf.maxIter);
    desc.add<Value<bool>>("", "double", "Is matrix double valued, false indicates the single type", true, &conf.isdouble);    
    desc.add<Value<std::size_t>>("", "blocksize", "Sets the blocksize for the matrix", 64, &conf.blocksize);
    desc.add<Value<float>>("", "lowerb_decay", "Sets the decaying rate for the lower bound", 1.0, &conf.lowerb_decay);
    auto path_in_options = desc.add<Value<std::string>, Attribute::required>("", "path_in", "Path to the input matrix/matrices", "d", &conf.path_in);
    
    try
    {
        desc.parse(argc, argv);

        if (help_option->count() == 1)
        {
            std::cout << desc << "\n";
            return 0;
        }
    }
    catch (const popl::invalid_option& e)
    {
        if (help_option->count() == 1)
        {
            std::cout << desc << "\n";
            return 0;
        }
        std::cerr << "Invalid Option Exception: " << e.what() << "\n";
        std::cerr << "error:  ";
        if (e.error() == popl::invalid_option::Error::missing_argument)
            std::cerr << "missing_argument\n";
        else if (e.error() == popl::invalid_option::Error::invalid_argument)
            std::cerr << "invalid_argument\n";
        else if (e.error() == popl::invalid_option::Error::too_many_arguments)
            std::cerr << "too_many_arguments\n";
        else if (e.error() == popl::invalid_option::Error::missing_option)
            std::cerr << "missing_option\n";

        if (e.error() == popl::invalid_option::Error::missing_option)
        {
            std::string option_name(e.option()->name(popl::OptionName::short_name, true));
            if (option_name.empty())
                option_name = e.option()->name(popl::OptionName::long_name, true);
            std::cerr << "option: " << option_name << "\n";
        }
        else
        {
            std::cerr << "option: " << e.option()->name(e.what_name()) << "\n";
            std::cerr << "value:  " << e.value() << "\n";
        }
        return EXIT_FAILURE;
    }
    catch (const std::exception& e)
    {
        if (help_option->count() == 1)
        {
            std::cout << desc << "\n";
            return 0;
        }

        std::cerr << "Exception: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    conf.opt = toupper(conf.opt.at(0));
    conf.mode = toupper(conf.mode.at(0));

    if (conf.mode != "R" && conf.mode != "A")
    {
        std::cout << "Illegal value for mode: \"" << conf.mode << "\""
                  << std::endl
                  << "Legal values are R or A" << std::endl;
        return -1;
    }

    if (conf.opt != "N" && conf.opt != "S")
    {
        std::cout << "Illegal value for opt: " << conf.opt << std::endl
                  << "Legal values are N, S" << std::endl;
        return -1;
    }

    if (conf.isdouble)
    {
        bse_solve<std::complex<double>>(conf);
    }
    else
    {
        bse_solve<std::complex<float>>(conf);
    }

#if defined(USE_MPI) || defined (HAS_NCCL)
    
    MPI_Finalize();
#endif    
}
