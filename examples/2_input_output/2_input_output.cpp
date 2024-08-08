/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2023, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "popl.hpp"

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

using namespace popl;

std::size_t GetFileSize(std::string path_in)
{
    std::ifstream file(path_in, std::ios::binary | std::ios::ate);
    return file.tellg();
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

	if(!isMatGen)
	{
            if (rank == 0)
            	std::cout << "start reading matrix\n";

            std::ostringstream problem(std::ostringstream::ate);
            if(sequence){
            	if(legacy)
            	{
                    problem << path_in << "gmat  1 " << std::setw(2) << i << ".bin";
            	}
            	else
            	{
                	problem << path_in << "mat_" << spin << "_" << std::setfill('0')
                        	<< std::setw(2) << kpoint << "_" << std::setfill('0')
                        	<< std::setw(2) << i << ".bin";
            	}
            }
            else
            {
            	problem << path_in;
            }

            if (rank == 0)
            	std::cout << "Reading matrix: "<< problem.str() << std::endl;


	    std::size_t file_size = GetFileSize(problem.str());

            //check the input file size
            try
            {
                if (N * N * sizeof(T) != file_size)
                {
                    throw std::logic_error(
                        std::string("The given file : ") + problem.str() +
                        std::string(" of size ") + std::to_string(file_size) +
                        std::string(
                        	" doesn't equals to the required size of matrix of size ") +
                    std::to_string(N * N * sizeof(T)));
            	}
           }
           catch (std::exception& e)
           {
           	 std::cerr << "Caught " << typeid(e).name() << " : " << e.what()
                    << std::endl;
            	return 1;
           }
#ifdef USE_MPI
#ifdef USE_BLOCK_CYCLIC
	   props->readHamiltonianBlockCyclicDist(problem.str(), H);
#else
	   props->readHamiltonianBlockDist(problem.str(), H);	   
#endif		
#else
           std::ifstream input(problem.str().c_str(), std::ios::binary);
           if (input.is_open())
           {
                input.read((char*)H, sizeof(T) * N * N);
           }
           else
           {
                throw std::string("error reading file: ") + problem.str();
           }
#endif		
	}
	else
	{
#ifdef USE_BLOCK_CYCLIC
	    std::cerr << 
		"Generating test matrix on the fly is not supported for block-cyclic distribution implementation" 
	    	<< std::endl;
#else		
            if (rank == 0)
                std::cout << "start generating matrix\n";

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
#endif    	    
	}

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

    popl::OptionParser desc("ChASE options");
    auto help_option = desc.add<Switch>("h", "help", "show this message");
    desc.add<Value<std::size_t>, Attribute::required>("", "n", "Size of the Input Matrix", 0, &conf.N);
    desc.add<Value<bool>>("", "double", "Is matrix double valued, false indicates the single type", true, &conf.isdouble);
    desc.add<Value<bool>>("", "complex", "Matrix is complex, false indicated the real matrix", true, &conf.iscomplex);
    desc.add<Value<std::size_t>, Attribute::required>("", "nev", "Wanted Number of Eigenpairs", 0, &conf.nev);
    desc.add<Value<std::size_t>>("", "nex", "Extra Search Dimensions", 25, &conf.nex);
    desc.add<Value<std::size_t>>("", "deg", "Initial filtering degree", 20, &conf.deg);
    desc.add<Value<std::size_t>>("", "maxDeg", "Sets the maximum value of the degree of the Chebyshev filter", 36, &conf.maxDeg);
    desc.add<Value<std::size_t>>("", "maxIter", "Sets the value of the maximum number of subspace iterations\nwithin ChASE", 25, &conf.maxIter);
    desc.add<Value<std::size_t>>("", "bgn", "Start ell", 2, &conf.bgn);
    desc.add<Value<std::size_t>>("", "end", "End ell", 2, &conf.end);
	desc.add<Value<std::string>>("", "spin", "spin", "d", &conf.spin);
    desc.add<Value<std::size_t>>("", "kpoint", "kpoint", 0, &conf.kpoint);
    desc.add<Value<double>>("", "tol", "Tolerance for Eigenpair convergence", 1e-10, &conf.tol);
    auto path_in_options = desc.add<Value<std::string>, Attribute::required>("", "path_in", "Path to the input matrix/matrices", "d", &conf.path_in);
    desc.add<Value<std::string>>("", "mode", "valid values are R(andom) or A(pproximate)", "A", &conf.mode);
    desc.add<Value<std::string>>("", "opt", "Optimi(S)e degree, or do (N)ot optimise", "S", &conf.opt);
    desc.add<Value<std::string>>("", "path_eigp", "Path to approximate solutions, only required when mode\nis Approximate, otherwise not used" , "", &conf.path_eigp);
    desc.add<Value<bool>>("", "sequence", "Treat as sequence of Problems. Previous ChASE solution is used, when available", false, &conf.sequence);
    desc.add<Value<std::size_t>>("", "lanczosIter", "Sets the number of Lanczos iterations executed by ChASE.", 25, &conf.lanczosIter);
    desc.add<Value<std::size_t>>("", "numLanczos", "Sets the number of stochastic vectors used for the spectral estimates in Lanczos", 4, &conf.numLanczos);
    auto isMatGen_options = desc.add<Value<bool>>("", "isMatGen", "generating a matrix in place", false, &conf.isMatGen);
    desc.add<Value<double>>("", "dmax", "Tolerance for Eigenpair convergence", 100, &conf.dmax);
#ifdef USE_BLOCK_CYCLIC
        desc.add<Value<std::size_t>>("", "mbsize", "block size for the row", 400, &conf.mbsize);
        desc.add<Value<std::size_t>>("", "nbsize", "block size for the column", 400, &conf.nbsize);
        desc.add<Value<int>>("", "dim0", "row number of MPI proc grid", 0, &conf.dim0);
        desc.add<Value<int>>("", "dim1", "column number of MPI proc grid", 0, &conf.dim1);
        desc.add<Value<int>>("", "irsrc", "The process row over which the first row of matrix is distributed.", 0, &conf.irsrc);
        desc.add<Value<int>>("", "icsrc", "The process column over which the first column of the array A\nis\n distributed." , 0, &conf.icsrc);
        desc.add<Value<std::string>>("", "major", "Major of MPI proc grid, valid values are R(ow) or C(olumn)" , "C", &conf.mode);
#endif
    desc.add<Value<bool>>("", "legacy", "Use legacy naming scheme?", false, &conf.legacy);

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
		if (e.error() == invalid_option::Error::missing_argument)
			std::cerr << "missing_argument\n";
		else if (e.error() == invalid_option::Error::invalid_argument)
			std::cerr << "invalid_argument\n";
		else if (e.error() == invalid_option::Error::too_many_arguments)
			std::cerr << "too_many_arguments\n";
		else if (e.error() == invalid_option::Error::missing_option)
			std::cerr << "missing_option\n";

		if (e.error() == invalid_option::Error::missing_option)
		{
			std::string option_name(e.option()->name(OptionName::short_name, true));
			if (option_name.empty())
				option_name = e.option()->name(OptionName::long_name, true);
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
