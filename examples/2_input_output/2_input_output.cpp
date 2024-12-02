#include <iostream>
#include <vector>
#include <complex>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>
#include "popl.hpp"
#include <omp.h>

#include "algorithm/performance.hpp"

#ifdef HAS_NCCL
#include "Impl/pchase_gpu/pchase_gpu.hpp"
using ARCH = chase::platform::GPU;
#elif defined(USE_MPI)
#include "Impl/pchase_cpu/pchase_cpu.hpp"
using ARCH = chase::platform::CPU;
#else
#include "Impl/chase_cpu/chase_cpu.hpp"
#ifdef HAS_CUDA
#include "Impl/chase_gpu/chase_gpu.hpp"
#endif
#endif

using namespace popl;

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
};

template <typename T>
int do_chase(ChASE_DriverProblemConfig& conf)
{
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

    std::cout << std::setprecision(16);
    auto Lambda__ = std::unique_ptr<chase::Base<T>[]>(new chase::Base<T>[(nev + nex)]);
    chase::Base<T>* Lambda = Lambda__.get();
    int grank = 0;

#if defined(USE_MPI) || defined(HAS_NCCL)
    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
        = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

#ifdef USE_BLOCKCYCLIC
    std::size_t blocksize = 64;
    auto Hmat = chase::distMatrix::BlockCyclicMatrix<T, ARCH>(N, N, blocksize, blocksize, mpi_grid);  
    auto Vec = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column, ARCH>(N, nev + nex, blocksize, mpi_grid);     
#else
    auto Hmat = chase::distMatrix::BlockBlockMatrix<T, ARCH>(N, N, mpi_grid);  
    auto Vec = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, ARCH>(N, nev + nex, mpi_grid);  
#endif

    T *H;
#if defined(HAS_NCCL)
    Hmat.allocate_cpu_data();
    H = Hmat.cpu_data();
#elif USE_MPI
    H = Hmat.l_data();
#endif
#ifdef USE_MPI
    auto single = chase::Impl::pChASECPU(nev, nex, &Hmat, &Vec, Lambda);
#elif HAS_NCCL
    auto single = chase::Impl::pChASEGPU(nev, nex, &Hmat, &Vec, Lambda);
#endif
#else
    auto V__ = std::unique_ptr<T[]>(new T[N * (nev + nex)]);
    auto H__ = std::unique_ptr<T[]>(new T[N * N]);

    T* V = V__.get();
    T *H = H__.get();   
#ifdef HAS_CUDA
    auto single = chase::Impl::ChASEGPU(N, nev, nex, H, N, V, N, Lambda);
#else
    auto single = chase::Impl::ChASECPU(N, nev, nex, H, N, V, N, Lambda);
#endif    
#endif

    chase::ChaseConfig<T>& config = single.GetConfig();
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

        std::chrono::high_resolution_clock::time_point start, end;
        std::chrono::duration<double> elapsed;

        start = std::chrono::high_resolution_clock::now();
  
        if(!isMatGen)
        {
            if(grank == 0)
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
            
            if(grank == 0)
                std::cout << "Reading matrix: "<< problem.str() << std::endl;
            
            single.loadProblemFromFile(problem.str());
        }
        else
        {
 	        if(grank == 0)
                std::cout << "start generating matrix\n";

            std::size_t xoff, yoff, xlen, ylen, ld;

#if defined(USE_MPI) || defined(HAS_NCCL)
#ifdef USE_BLOCKCYCLIC
            throw std::runtime_error("Matrix Generation mode is not supported for block cyclic matrix");
#else 
            std::size_t *g_offs = Hmat.g_offs();
            xoff = g_offs[0];
            yoff = g_offs[1];
            xlen = Hmat.l_rows();
            ylen = Hmat.l_cols();
            ld = Hmat.l_ld();
#endif            
#else
            xoff = 0;
            yoff = 0;
            xlen = N;
            ylen = N;
            ld = N;
#endif
            chase::Base<T> epsilon = 1e-4;
            chase::Base<T>* eigenv = new chase::Base<T>[N];
            
            for (std::size_t i = 0; i < ylen; i++) {
                for (std::size_t j = 0; j < xlen; j++) {
                    if (xoff + j == (i + yoff)) {
                        H[i * ld + j] =  dmax * (epsilon + (chase::Base<T>)(xoff + j) * (1.0 - epsilon) / (chase::Base<T>)N);
                    }else{
                        H[i * ld + j] = T(0.0);
                    }
                }
            }   	    
        }

        end = std::chrono::high_resolution_clock::now();

        elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);


        if(grank == 0)
        {
            std::cout << "matrix are loaded in " << elapsed.count()
                    << " seconds" << std::endl;
        }

        if(!single.checkSymmetryEasy())
        {
            single.symOrHermMatrix('L');
        }

        chase::PerformanceDecoratorChase<T> performanceDecorator(&single);

        chase::Solve(&performanceDecorator);
        
        if(grank == 0)
        {        
            std::cout << " ChASE timings: "
                        << "\n";
            performanceDecorator.GetPerfData().print(N);
#ifdef PRINT_EIGENVALUES
            chase::Base<T>* resid = single.GetResid();
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
#if defined(USE_MPI) || defined (HAS_NCCL)
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
    desc.add<Value<std::string>>("", "mode", "valid values are R(andom) or A(pproximate)", "R", &conf.mode);
    desc.add<Value<std::string>>("", "opt", "Optimi(S)e degree, or do (N)ot optimise", "S", &conf.opt);
    desc.add<Value<std::string>>("", "path_eigp", "Path to approximate solutions, only required when mode\nis Approximate, otherwise not used" , "", &conf.path_eigp);
    desc.add<Value<bool>>("", "sequence", "Treat as sequence of Problems. Previous ChASE solution is used, when available", false, &conf.sequence);
    desc.add<Value<std::size_t>>("", "lanczosIter", "Sets the number of Lanczos iterations executed by ChASE.", 25, &conf.lanczosIter);
    desc.add<Value<std::size_t>>("", "numLanczos", "Sets the number of stochastic vectors used for the spectral estimates in Lanczos", 4, &conf.numLanczos);
    auto isMatGen_options = desc.add<Value<bool>>("", "isMatGen", "generating a matrix in place", false, &conf.isMatGen);
    desc.add<Value<double>>("", "dmax", "Tolerance for Eigenpair convergence", 100, &conf.dmax);
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

#if defined(USE_MPI) || defined (HAS_NCCL)
    MPI_Finalize();
#endif

    return 0;
}