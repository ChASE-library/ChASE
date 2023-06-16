/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2023, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "ChASE-MPI/chase_mpi.hpp"
#include "algorithm/performance.hpp"

#include "ChASE-MPI/impl/chase_mpidla_blaslapack_seq.hpp"
#include "ChASE-MPI/impl/chase_mpidla_blaslapack_seq_inplace.hpp"

#ifdef USE_MPI
#include "ChASE-MPI/impl/chase_mpidla_blaslapack.hpp"
#ifdef DRIVER_BUILD_MGPU
#include "ChASE-MPI/impl/chase_mpidla_mgpu.hpp"
#endif
#endif

using namespace chase;
using namespace chase::mpi;

namespace po = boost::program_options;
namespace bf = boost::filesystem;

template <typename T>
void readMatrix(T* H, std::string path_in, std::string spin, std::size_t kpoint,
                std::size_t index, std::string suffix, std::size_t size,
                bool legacy)
{
    std::ostringstream problem(std::ostringstream::ate);
    if (legacy)
        problem << path_in << "gmat  1 " << std::setw(2) << index << suffix;
    else
        problem << path_in << "mat_" << spin << "_" << std::setfill('0')
                << std::setw(2) << kpoint << "_" << std::setfill('0')
                << std::setw(2) << index << suffix;

    std::cout << problem.str() << std::endl;
    std::ifstream input(problem.str().c_str(), std::ios::binary);

    std::cout << problem.str().c_str() << " "
              << "---" << bf::file_size(problem.str().c_str()) << '\n';

    if (input.is_open())
    {
        input.read((char*)H, sizeof(T) * size);
    }
    else
    {
        throw std::string("error reading file: ") + problem.str();
    }
}

template <typename T>
void readMatrix(T* H, std::string path_in, std::string spin, std::size_t kpoint,
                std::size_t index, std::string suffix, std::size_t size,
                bool legacy, std::size_t xoff, std::size_t yoff,
                std::size_t xlen, std::size_t ylen)
{
    std::size_t N = std::sqrt(size);
    std::ostringstream problem(std::ostringstream::ate);

    int rank;

#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    rank = 0;
#endif

    if (legacy)
    {
        problem << path_in << "gmat  1 " << std::setw(2) << index << suffix;
    }
    else
    {
        problem << path_in << "mat_" << spin << "_" << std::setfill('0')
                << std::setw(2) << kpoint << "_" << std::setfill('0')
                << std::setw(2) << index << suffix;
    }

    if (rank == 0)
        std::cout << problem.str() << std::endl;

    std::size_t file_size = bf::file_size(problem.str().c_str());

    try
    {
        if (size * sizeof(T) != file_size)
        {
            throw std::logic_error(
                std::string("The given file : ") + problem.str() +
                std::string(" of size ") + std::to_string(file_size) +
                std::string(
                    " doesn't equals to the required size of matrix of size ") +
                std::to_string(size * sizeof(T)));
        }
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught " << typeid(e).name() << " : " << e.what()
                  << std::endl;
        return;
    }

    std::ifstream input(problem.str().c_str(), std::ios::binary);
    if (!input.is_open())
    {
        throw new std::logic_error(std::string("error reading file: ") +
                                   problem.str());
    }

    for (std::size_t y = 0; y < ylen; y++)
    {
        input.seekg(((xoff) + N * (yoff + y)) * sizeof(T));
        input.read(reinterpret_cast<char*>(H + xlen * y), xlen * sizeof(T));
    }
}

template <typename T>
void readMatrix(T* H, std::string path_in, std::string spin, std::size_t kpoint,
                std::size_t index, std::string suffix, std::size_t size,
                bool legacy, std::size_t m, std::size_t mblocks,
                std::size_t nblocks, std::size_t* r_offs, std::size_t* r_lens,
                std::size_t* r_offs_l, std::size_t* c_offs, std::size_t* c_lens,
                std::size_t* c_offs_l)
{
    std::size_t N = std::sqrt(size);
    std::ostringstream problem(std::ostringstream::ate);

    int rank;

#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    rank = 0;
#endif

    if (legacy)
    {
        problem << path_in << "gmat  1 " << std::setw(2) << index << suffix;
    }
    else
    {
        problem << path_in << "mat_" << spin << "_" << std::setfill('0')
                << std::setw(2) << kpoint << "_" << std::setfill('0')
                << std::setw(2) << index << suffix;
    }

    if (rank == 0)
        std::cout << problem.str() << std::endl;

    std::size_t file_size = bf::file_size(problem.str().c_str());

    try
    {
        if (size * sizeof(T) != file_size)
        {
            throw std::logic_error(
                std::string("The given file : ") + problem.str() +
                std::string(" of size ") + std::to_string(file_size) +
                std::string(
                    " doesn't equals to the required size of matrix of size ") +
                std::to_string(size * sizeof(T)));
        }
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught " << typeid(e).name() << " : " << e.what()
                  << std::endl;
        return;
    }

    std::ifstream input(problem.str().c_str(), std::ios::binary);
    if (!input.is_open())
    {
        throw new std::logic_error(std::string("error reading file: ") +
                                   problem.str());
    }

    for (std::size_t j = 0; j < nblocks; j++)
    {
        for (std::size_t i = 0; i < mblocks; i++)
        {
            for (std::size_t q = 0; q < c_lens[j]; q++)
            {
                input.seekg(((q + c_offs[j]) * N + r_offs[i]) * sizeof(T));
                input.read(reinterpret_cast<char*>(H + (q + c_offs_l[j]) * m +
                                                   r_offs_l[i]),
                           r_lens[i] * sizeof(T));
            }
        }
    }
}

template <typename T>
void readMatrix(T* H, std::string path_in, std::size_t size, std::size_t xoff,
                std::size_t yoff, std::size_t xlen, std::size_t ylen)
{
    std::size_t N = std::sqrt(size);
    std::ostringstream problem(std::ostringstream::ate);
    problem << path_in;
    int rank;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    rank = 0;
#endif

    if (rank == 0)
        std::cout << problem.str() << std::endl;

    std::size_t file_size = bf::file_size(problem.str().c_str());

    try
    {
        if (size * sizeof(T) != file_size)
        {
            throw std::logic_error(
                std::string("The given file : ") + problem.str() +
                std::string(" of size ") + std::to_string(file_size) +
                std::string(
                    " doesn't equals to the required size of matrix of size ") +
                std::to_string(size * sizeof(T)));
        }
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught " << typeid(e).name() << " : " << e.what()
                  << std::endl;
        return;
    }

    std::ifstream input(problem.str().c_str(), std::ios::binary);

    if (!input.is_open())
    {
        throw new std::logic_error(std::string("error reading file: ") +
                                   problem.str());
    }

    for (std::size_t y = 0; y < ylen; y++)
    {
        input.seekg(((xoff) + N * (yoff + y)) * sizeof(T));
        input.read(reinterpret_cast<char*>(H + xlen * y), xlen * sizeof(T));
    }
}

template <typename T>
void readMatrix(T* H, std::string path_in, std::size_t size, std::size_t m,
                std::size_t mblocks, std::size_t nblocks, std::size_t* r_offs,
                std::size_t* r_lens, std::size_t* r_offs_l, std::size_t* c_offs,
                std::size_t* c_lens, std::size_t* c_offs_l)
{

    std::size_t N = std::sqrt(size);
    std::ostringstream problem(std::ostringstream::ate);
    problem << path_in;

    int rank;

#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    rank = 0;
#endif

    if (rank == 0)
        std::cout << problem.str() << std::endl;

    std::size_t file_size = bf::file_size(problem.str().c_str());

    try
    {
        if (size * sizeof(T) != file_size)
        {
            throw std::logic_error(
                std::string("The given file : ") + problem.str() +
                std::string(" of size ") + std::to_string(file_size) +
                std::string(
                    " doesn't equals to the required size of matrix of size ") +
                std::to_string(size * sizeof(T)));
        }
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught " << typeid(e).name() << " : " << e.what()
                  << std::endl;
        return;
    }

    std::ifstream input(problem.str().c_str(), std::ios::binary);
    if (!input.is_open())
    {
        throw new std::logic_error(std::string("error reading file: ") +
                                   problem.str());
    }

    for (std::size_t j = 0; j < nblocks; j++)
    {
        for (std::size_t i = 0; i < mblocks; i++)
        {
            for (std::size_t q = 0; q < c_lens[j]; q++)
            {
                input.seekg(((q + c_offs[j]) * N + r_offs[i]) * sizeof(T));
                input.read(reinterpret_cast<char*>(H + (q + c_offs_l[j]) * m +
                                                   r_offs_l[i]),
                           r_lens[i] * sizeof(T));
            }
        }
    }
}

struct ChASE_DriverProblemConfig
{
    std::size_t N;   // Size of the Matrix
    std::size_t nev; // Number of sought after eigenvalues
    std::size_t nex; // Extra size of subspace
    std::size_t deg; // initial degree
    std::size_t bgn; // beginning of sequence
    std::size_t end; // end of sequence

    std::size_t maxIter; // maximum number of subspace iterations within ChASE.
    std::size_t maxDeg;  // maximum value of the degree of the Chebyshev filter
    double tol;          // desired tolerance
    bool sequence;       // handle this as a sequence?

    std::string path_in;   // path to the matrix input files
    std::string mode;      // Approx or Random mode
    std::string opt;       // enable optimisation of degree
    std::string arch;      // ??
    std::string path_eigp; // TODO
    std::string path_out;
    std::string path_name;

    std::size_t kpoint;
    bool legacy;
    std::string spin;

    bool iscomplex;
    bool isdouble;

    bool isMatGen;
    double dmax;

    std::size_t lanczosIter;
    std::size_t numLanczos;

#ifdef USE_BLOCK_CYCLIC
    std::size_t mbsize;
    std::size_t nbsize;
    int dim0;
    int dim1;
    int irsrc;
    int icsrc;
    std::string major;
#endif
};

template <typename T>
int do_chase(ChASE_DriverProblemConfig& conf)
{
    // todo due to legacy reasons we unpack the struct
    std::size_t N = conf.N;
    std::size_t nev = conf.nev;
    std::size_t nex = conf.nex;
    std::size_t deg = conf.deg;
    std::size_t bgn = conf.bgn;
    std::size_t end = conf.end;
    std::size_t maxDeg = conf.maxDeg;
    std::size_t maxIter = conf.maxIter;
    double dmax = conf.dmax;

    bool isMatGen = conf.isMatGen;

    double tol = conf.tol;
    bool sequence = conf.sequence;

    std::string path_in = conf.path_in;
    std::string mode = conf.mode;
    std::string opt = conf.opt;
    std::string arch;
    std::string path_eigp = conf.path_eigp;
    std::string path_out = conf.path_out;
    std::string path_name = conf.path_name;

    std::size_t lanczosIter = conf.lanczosIter;
    std::size_t numLanczos = conf.numLanczos;

    std::size_t kpoint = conf.kpoint;
    bool legacy = conf.legacy;
    std::string spin = conf.spin;
#ifdef USE_BLOCK_CYCLIC
    std::size_t mbsize = conf.mbsize;
    std::size_t nbsize = conf.nbsize;
    int dim0 = conf.dim0;
    int dim1 = conf.dim1;
    int irsrc = conf.irsrc;
    int icsrc = conf.irsrc;
    std::string major = conf.major;

    if (dim0 == 0 || dim1 == 0)
    {
        int dims[2];
        dims[0] = dims[1] = 0;
        int gsize;
        MPI_Comm_size(MPI_COMM_WORLD, &gsize);
        MPI_Dims_create(gsize, 2, dims);
        dim0 = dims[0];
        dim1 = dims[1];
    }
#endif
    int rank, size;

#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#else
    rank = 0;
    size = 1;
#endif

    //----------------------------------------------------------------------------
    std::cout << std::setprecision(16);

#ifdef USE_MPI
#if defined(DRIVER_BUILD_MGPU)
    typedef ChaseMpi<ChaseMpiDLAMultiGPU, T> CHASE;
#else
    typedef ChaseMpi<ChaseMpiDLABlaslapack, T> CHASE;
#endif // CUDA or not
#else
    typedef ChaseMpi<ChaseMpiDLABlaslapackSeq, T> CHASE;
#endif // seq ChASE

#ifdef USE_MPI
#ifdef USE_BLOCK_CYCLIC
    auto props = new ChaseMpiProperties<T>(
        N, mbsize, nbsize, nev, nex, dim0, dim1,
        const_cast<char*>(major.c_str()), irsrc, icsrc, MPI_COMM_WORLD);
#else
    auto props = new ChaseMpiProperties<T>(N, nev, nex, MPI_COMM_WORLD);
#endif
#endif

#ifdef USE_MPI
    auto m_ = props->get_m();
    auto n_ = props->get_n();
    auto ldh_ = props->get_ldh();
#else
    auto m_ = N;
    auto n_ = N;
    auto ldh_ = N;
#endif

    auto V__ = std::unique_ptr<T[]>(new T[m_ * (nev + nex)]);
    auto Lambda__ = std::unique_ptr<Base<T>[]>(new Base<T>[(nev + nex)]);
    auto H__ = std::unique_ptr<T[]>(new T[ldh_ * n_]);

    T* V = V__.get();
    Base<T>* Lambda = Lambda__.get();
    T *H = H__.get();

#if defined(USE_MPI)
#ifdef USE_BLOCK_CYCLIC
    CHASE single(props, H, ldh_, V, Lambda);
#else
    CHASE single(props, H, ldh_, V, Lambda);
#endif
#else
    CHASE single(N, nev, nex, H, ldh_, V, Lambda);
#endif
    ChaseConfig<T>& config = single.GetConfig();
    config.SetTol(tol);
    config.SetDeg(deg);
    config.SetOpt(opt == "S");
    config.SetLanczosIter(lanczosIter);
    config.SetNumLanczos(numLanczos);
    config.SetMaxDeg(maxDeg);
    config.SetMaxIter(maxIter);


    if (!sequence)
    {
        bgn = end = 1;
    }

    for (auto i = bgn; i <= end; ++i)
    {
        if (i == bgn || !sequence)
        {
            for (int j = 0; j < (nev + nex); j++)
            {
                Lambda[j] = 0.0;
            }
            
        }
        else
        {
            config.SetApprox(true);
        }

#ifdef USE_MPI
#ifdef USE_BLOCK_CYCLIC
        /*local block number = mblocks x nblocks*/
        std::size_t mblocks = props->get_mblocks();
        std::size_t nblocks = props->get_nblocks();

        /*local matrix size = m x n*/
        std::size_t m = props->get_m();
        std::size_t n = props->get_n();

        /*global and local offset/length of each block of block-cyclic data*/
        std::size_t *r_offs, *c_offs, *r_lens, *c_lens, *r_offs_l, *c_offs_l;

        props->get_offs_lens(r_offs, r_lens, r_offs_l, c_offs, c_lens,
                             c_offs_l);
#else
        std::size_t xoff;
        std::size_t yoff;
        std::size_t xlen;
        std::size_t ylen;

        props->get_off(&xoff, &yoff, &xlen, &ylen);
#endif

#else
        std::size_t xoff = 0;
        std::size_t yoff = 0;
        std::size_t xlen = N;
        std::size_t ylen = N;

#endif
        std::chrono::high_resolution_clock::time_point start, end;
        std::chrono::duration<double> elapsed;

        start = std::chrono::high_resolution_clock::now();

#ifdef USE_NSIGHT
        nvtxRangePushA("MatrixIO");
#endif

        if (rank == 0)
            std::cout << "start reading matrix\n";
#ifdef USE_BLOCK_CYCLIC
        if (sequence)
        {
            readMatrix(H, path_in, spin, kpoint, i, ".bin", N * N, legacy, m,
                       mblocks, nblocks, r_offs, r_lens, r_offs_l, c_offs,
                       c_lens, c_offs_l);
        }
        else
        {
            readMatrix(H, path_in, N * N, m, mblocks, nblocks, r_offs, r_lens,
                       r_offs_l, c_offs, c_lens, c_offs_l);
        }
#else
        if (sequence)
        {
            readMatrix(H, path_in, spin, kpoint, i, ".bin", N * N, legacy, xoff,
                       yoff, xlen, ylen);
        }
        else
        {
            if(isMatGen){
                Base<T> epsilon = 1e-4;
                Base<T>* eigenv = new Base<T>[N];
                for (std::size_t i = 0; i < ylen; i++) {
                    for (std::size_t j = 0; j < xlen; j++) {
                        if (xoff + j == (i + yoff)) {
                            H[i * xlen + j] =  dmax * (epsilon + (Base<T>)(xoff + j) * (1.0 - epsilon) / (Base<T>)N);
                        }else{
                            H[i * xlen + j] = T(0.0);
                        }
                    }
                }
            }else{
                readMatrix(H, path_in, N * N, xoff, yoff, xlen, ylen);
            }
        }
#endif

#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        end = std::chrono::high_resolution_clock::now();

        elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);

        if (rank == 0)
            std::cout << "matrix are loaded in " << elapsed.count()
                      << " seconds" << std::endl;

#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
        PerformanceDecoratorChase<T> performanceDecorator(&single);
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        chase::Solve(&performanceDecorator);
        // chase::Solve(&single);
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        if (rank == 0)
        {
            std::cout << " ChASE timings: "
                      << "\n";
            performanceDecorator.GetPerfData().print(N);
#ifdef PRINT_EIGENVALUES
            Base<T>* resid = single.GetResid();
            std::cout << "Finished Problem \n";
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
#endif
        }
    }

    return 0;
}

int main(int argc, char* argv[])
{

#ifdef USE_MPI
    MPI_Init(&argc, &argv);
#endif

    ChASE_DriverProblemConfig conf;

    po::options_description desc("ChASE Options");

    desc.add_options()(                                                   //
        "help,h",                                                         //
        "show this message"                                               //
        )(                                                                //
        "n", po::value<std::size_t>(&conf.N)->required(),                 //
        "Size of the Input Matrix"                                        //
        )(                                                                //
        "double", po::value<bool>(&conf.isdouble)->default_value(true),   //
        "Is matrix double valued, false indicates the single type"        //
        )(                                                                //
        "complex", po::value<bool>(&conf.iscomplex)->default_value(true), //
        "Matrix is complex, false indicated the real matrix"              //
        )(                                                                //
        "nev", po::value<std::size_t>(&conf.nev)->required(),             //
        "Wanted Number of Eigenpairs"                                     //
        )(                                                                //
        "nex", po::value<std::size_t>(&conf.nex)->default_value(25),      //
        "Extra Search Dimensions"                                         //
        )(                                                                //
        "deg", po::value<std::size_t>(&conf.deg)->default_value(20),      //
        "Initial filtering degree"                                        //
        )("maxDeg",
          po::value<std::size_t>(&conf.maxDeg)->default_value(36), //
          "Sets the maximum value of the degree of the Chebyshev filter")(
        "maxIter",
        po::value<std::size_t>(&conf.maxIter)->default_value(25), //
        "Sets the value of the maximum number of subspace iterations"
        "within ChASE")(                                                  //
        "bgn", po::value<std::size_t>(&conf.bgn)->default_value(2),       //
        "Start ell"                                                       //
        )(                                                                //
        "end", po::value<std::size_t>(&conf.end)->default_value(2),       //
        "End ell"                                                         //
        )(                                                                //
        "spin", po::value<std::string>(&conf.spin)->default_value("d"),   //
        "spin"                                                            //
        )(                                                                //
        "kpoint", po::value<std::size_t>(&conf.kpoint)->default_value(0), //
        "kpoint"                                                          //
        )(                                                                //
        "tol", po::value<double>(&conf.tol)->default_value(1e-10),        //
        "Tolerance for Eigenpair convergence"                             //
        )(                                                                //
        "path_in", po::value<std::string>(&conf.path_in)->required(),     //
        "Path to the input matrix/matrices"                               //
        )(                                                                //
        "mode", po::value<std::string>(&conf.mode)->default_value("A"),   //
        "valid values are R(andom) or A(pproximate)"                      //
        )(                                                                //
        "opt", po::value<std::string>(&conf.opt)->default_value("S"),     //
        "Optimi(S)e degree, or do (N)ot optimise"                         //
        )(                                                                //
        "path_eigp", po::value<std::string>(&conf.path_eigp),             //
        "Path to approximate solutions, only required when mode"          //
        "is Approximate, otherwise not used"                              //
        )(                                                                //
        "sequence",
        po::value<bool>(&conf.sequence)->default_value(false),            //
        "Treat as sequence of Problems. Previous ChASE solution is used," //
        "when available"                                                  //
        )("lanczosIter",
          po::value<std::size_t>(&conf.lanczosIter)->default_value(25),
          "Sets the number of Lanczos iterations executed by ChASE.")(
        "numLanczos",
        po::value<std::size_t>(&conf.numLanczos)->default_value(4),
        " Sets the number of stochastic vectors used for the spectral "
        "estimates"
        "in Lanczos")(
        "isMatGen", po::value<bool>(&conf.isMatGen)->default_value(false), //
        "generating a matrix in place"
        )(
        "dmax", po::value<double>(&conf.dmax)->default_value(100),        //
        "for generating the spectrum"                             //
        )                
#ifdef USE_BLOCK_CYCLIC
        ( //
            "mbsize",
            po::value<std::size_t>(&conf.mbsize)->default_value(400), //
            "block size for the row"                                  //
            )(                                                        //
            "nbsize",
            po::value<std::size_t>(&conf.nbsize)->default_value(400), //
            "block size for the column"                               //
            )(                                                        //
            "dim0", po::value<int>(&conf.dim0)->default_value(0),     //
            "row number of MPI proc grid"                             //
            )(                                                        //
            "dim1", po::value<int>(&conf.dim1)->default_value(0),     //
            "column number of MPI proc grid"                          //
            )(                                                        //
            "irsrc", po::value<int>(&conf.irsrc)->default_value(0),   //
            "The process row over which the first row of matrix is"   //
            "distributed."                                            //
            )(                                                        //
            "icsrc", po::value<int>(&conf.icsrc)->default_value(0),   //
            "The process column over which the first column of the array A "
            "is"           //
            "distributed." //
            )(             //
            "major",
            po::value<std::string>(&conf.mode)->default_value("C"),      //
            "Major of MPI proc grid, valid values are R(ow) or C(olumn)" //
            )
#endif
            ("legacy",
             po::value<bool>(&conf.legacy)->default_value(false), //
             "Use legacy naming scheme?");                        //

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // print help
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }

    po::notify(vm);
    conf.mode = toupper(conf.mode.at(0));
    conf.opt = toupper(conf.opt.at(0));

    if (conf.bgn > conf.end)
    {
        std::cout << "Begin must be smaller than End!" << std::endl;
        return -1;
    }

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

    if (conf.path_eigp.empty() && conf.mode == "A")
    {
        std::cout << "eigp is required when mode is " << conf.mode << std::endl;
        return -1;
    }

    if(conf.isMatGen){
        conf.iscomplex = false;
    }

    if (conf.isdouble)
    {
        if (conf.iscomplex)
        {
            do_chase<std::complex<double>>(conf);
        }
        else
        {
            do_chase<double>(conf);
        }
    }
    else
    {
        std::cout << "single not implemented\n";
    }

#ifdef USE_MPI
    MPI_Finalize();
#else
    return 0;
#endif
}
