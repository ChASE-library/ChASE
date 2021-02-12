#include <boost/program_options.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <time.h>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include "scalapack_templates.hpp"

#include "algorithm/performance.hpp"
#include "ChASE-MPI/chase_mpi.hpp"

#include "ChASE-MPI/impl/chase_mpihemm_blas_seq.hpp"
#include "ChASE-MPI/impl/chase_mpihemm_blas_seq_inplace.hpp"
#include "ChASE-MPI/impl/chase_mpihemm_blas.hpp"

const int i_zero = 0, i_one = 1;
const std::size_t sze_one = 1;

typedef std::size_t DESC[ 9 ];

namespace po = boost::program_options;
using namespace::chase;
using namespace::chase::mpi;


template <typename T>
std::vector<T> generateEyeMat(std::size_t N){
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) std::cout << "]> Generating an Identity matrix as S" << std::endl;

  std::vector<T> S(N * N, T(0.0));
  for (std::size_t i = 0; i < N; i++) {
    S[i + i * N] = 1.0;
  }

  return S;
}

template <typename T>
std::vector<T> generateClementMat(std::size_t N){
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) std::cout << "]> Generating a Clement matrix as H" << std::endl;

  std::vector<T> C(N * N, T(0.0));
  for (auto i = 0; i < N; ++i) {
    C[i + N * i] = 0;
    if (i != N - 1) C[i + 1 + N * i] = std::sqrt(i * (N + 1 - i));
    if (i != N - 1) C[i + N * (i + 1)] = std::sqrt(i * (N + 1 - i));
  }

  return C;
}

//write eigenvalues: path_out/eigenvalues_index.bin
//write eigenvetors: path_out/eigenvectors_index.bin
template <typename T>
void writeMatrix(T *H, std::string path_out, std::string prefix, 
		       std::size_t index, std::string suffix, std::size_t size)
{
  std::ostringstream problem(std::ostringstream::ate);
  problem << path_out << prefix << "_" << index << suffix;

  std::cout << "]> writing ";
  std::cout << prefix << " of size into ";
  std::cout << problem.str() << std::endl;

  auto outfile = std::fstream(problem.str().c_str(), std::ios::out | std::ios::binary);

  outfile.write((char*)&H[0], size * sizeof(T));

  outfile.close();

}

//read eigenvalues: path_in/eigenvalues_index.bin
//read eigenvetors: path_in/eigenvectors_index.bin
template <typename T>
void readMatrix(T *H, std::string path_in, std::string prefix,
                       std::size_t index, std::string suffix, std::size_t size)
{
  std::ostringstream problem(std::ostringstream::ate);
  problem << path_in << prefix << "_" << index << suffix;

  std::cout << "]> reading ";
  std::cout << prefix << " of size from ";
  std::cout << problem.str() << std::endl;

  auto infile = std::fstream(problem.str().c_str(), std::ios::binary);

  infile.write((char*)&H[0], size * sizeof(T));

  infile.close();

}


template <typename T>
void readMatrix(T* H, std::string path_in, std::string prefix,
                std::size_t index, std::string suffix, std::size_t size, 
		std::size_t m, std::size_t mblocks, std::size_t nblocks,
            	std::size_t* r_offs, std::size_t* r_lens, std::size_t* r_offs_l, 
		std::size_t* c_offs, std::size_t* c_lens, std::size_t* c_offs_l)
{

  std::size_t N = std::sqrt(size);
  std::ostringstream problem(std::ostringstream::ate);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  problem << path_in << prefix << "  1 " << std::setw(2) << index << suffix;
  if (rank == 0) std::cout << "]> Loading " << problem.str() << std::endl;

  std::ifstream input(problem.str().c_str(), std::ios::binary);

  if (!input.is_open()) {
    throw new std::logic_error(std::string("error reading file: ") +
                               problem.str());
  }

  for(std::size_t j = 0; j < nblocks; j++){
      for(std::size_t i = 0; i < mblocks; i++){
          for(std::size_t q = 0; q < c_lens[j]; q++){
	      input.seekg(((q + c_offs[j]) * N + r_offs[i])* sizeof(T));
	      input.read(reinterpret_cast<char*>(H + (q + c_offs_l[j]) * m + r_offs_l[i]), r_lens[i] * sizeof(T));
	  }
      }
  }
}

using T = std::complex<double>;

typedef ChaseMpi<ChaseMpiHemmBlas, T> CHASE;

struct ChASE_DriverProblemConfig {
  std::size_t N;    // Size of the Matrix
  std::size_t nev;  // Number of sought after eigenvalues
  std::size_t nex;  // Extra size of subspace
  std::size_t deg;  // initial degree
  std::size_t bgn;  // beginning of sequence
  std::size_t end;  // end of sequence

  double tol;     // desired tolerance
  bool sequence;  // handle this as a sequence?

  std::string path_in;    // path to the matrix input files
  std::string mode;       // Approx or Random mode
  std::string opt;        // enable optimisation of degree
  std::string arch;       // ??
  std::string path_eigp;  // TODO
  std::string path_out;
  std::string path_name;

  bool complex;
  bool isdouble;
  
  std::size_t mbsize;
  std::size_t nbsize;
  int dim0;
  int dim1;
  int irsrc;
  int icsrc;
  std::string major;
};

template <typename T>
int do_chase_gev(ChASE_DriverProblemConfig& conf) {

  std::size_t N = conf.N;
  std::size_t nev = conf.nev;
  std::size_t nex = conf.nex;
  std::size_t deg = conf.deg;
  std::size_t bgn = conf.bgn;
  std::size_t end = conf.end;

  double tol = conf.tol;
  bool sequence = conf.sequence;

  std::string path_in = conf.path_in;
  std::string mode = conf.mode;
  std::string opt = conf.opt;
  std::string arch;
  std::string path_eigp = conf.path_eigp;
  std::string path_out = conf.path_out;
  std::string path_name = conf.path_name;

  std::size_t mbsize = conf.mbsize;
  std::size_t nbsize = conf.nbsize;
  int dim0 = conf.dim0;
  int dim1 = conf.dim1;
  int irsrc = conf.irsrc;
  int icsrc = conf.irsrc;
  std::string major = conf.major;

  if(dim0 == 0 || dim1 == 0){
    int dims[2];
    dims[0] = dims[1] = 0;
    int gsize;
    MPI_Comm_size(MPI_COMM_WORLD, &gsize);    
    MPI_Dims_create(gsize, 2, dims);
    dim0 = dims[0];
    dim1 = dims[1];    
  }

  int rank, size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::cout << std::setprecision(16);

  if (rank == 0){
    std::cout << "\nChASE example driver for a sequence of Generalized "
	<<"Eigenproblems."
	<< std::endl;
  }

  // Timing variables
  std::chrono::high_resolution_clock::time_point st, ed, st_loc;
  std::size_t prb_nb = end - bgn + 1;
  std::chrono::duration<double> elapsed[6][prb_nb];

  //Scalapack part
  //Initalize Scalapack environment
  int myproc, nprocs;
  blacs_pinfo( &myproc, &nprocs );
  int ictxt;
  int val;
  blacs_get( &ictxt, &i_zero, &val );
  blacs_gridinit( &ictxt, major.at(0), &dim0, &dim1 );
  int myrow, mycol;
  blacs_gridinfo( &ictxt, &dim0, &dim1, &myrow, &mycol);

  //get local size of matrix = N_loc_r x N_loc_c
  std::size_t N_loc_r, N_loc_c;

  N_loc_r = numroc( &N, &mbsize, &myrow, &irsrc, &dim0 );
  N_loc_c = numroc( &N, &nbsize, &mycol, &icsrc, &dim1 );

  //for column major matrix, the leading dimension
  std::size_t lld_loc = std::max(N_loc_r, (std::size_t)1);

  //construct scalapack matrix descriptor 
  DESC   desc;
  int    info;

  descinit( desc, &N, &N, &mbsize, &nbsize, &irsrc, &irsrc, &ictxt, &lld_loc, &info );

  //ChASE part
  //eigenpairs of standard eigenproblem
  auto V__ = std::unique_ptr<T[]>(new T[N * (nev + nex)]);
  auto Lambda__ = std::unique_ptr<Base<T>[]>(new Base<T>[(nev + nex)]);

  T* V = V__.get();
  Base<T>* Lambda = Lambda__.get();

  //Setup ChASE environment for a standard eigenproblem
  CHASE single(new ChaseMpiProperties<T>(N, mbsize, nbsize, nev, nex, dim0, 
	dim1, const_cast<char*>(major.c_str()), irsrc, icsrc, MPI_COMM_WORLD), V, Lambda);

  ChaseConfig<T>& config = single.GetConfig();
  config.SetTol(tol);
  config.SetDeg(deg);
  config.SetOpt(opt == "S");
  config.SetMaxIter(100);

  if (rank == 0)
    std::cout << "\n"
              << config;

  std::mt19937 gen(1337.0);
  std::normal_distribution<> d;

  T* matrix = single.GetMatrixPtr();

  // Using ChASE-MPI functionalities to get some additional information 
  // on the block cyclic data layout which faciliates the implementation
  /*local block number = mblocks x nblocks*/
  std::size_t mblocks = single.get_mblocks(); 
  std::size_t nblocks = single.get_nblocks();

  /*local matrix size = m x n*/
  std::size_t m = single.get_m(); // should = N_loc_r
  std::size_t n = single.get_n(); // should = N_loc_c

  /*global and local offset/length of each block of block-cyclic data*/
  std::size_t *r_offs, *c_offs, *r_lens, *c_lens, *r_offs_l, *c_offs_l;

  single.get_offs_lens(r_offs, r_lens, r_offs_l, c_offs, c_lens, c_offs_l);

  Base<T> scale; //for t_psy(he)gst in Scalapack

  // GEV: H * X = LAMBDA * S * X, in which H and S are local matrices
  T *H = new T [N_loc_r * N_loc_c];
  T *S = new T [N_loc_r * N_loc_c];

  //sequence of problem
  for(auto idx = bgn; idx <= end; ++idx){

    if (rank == 0) {
      std::cout << "]> Starting Problem #" << idx << "\n";
    }
    
    st = std::chrono::high_resolution_clock::now();
    st_loc = std::chrono::high_resolution_clock::now();

    // if the path of matrices are empty, generate H as a Clement matrix
    // and generate S as an Identity matrix
    if(path_in.empty())
    {
      /*Generate Clement matrix*/
      std::vector<T> HH = generateClementMat<T>(N);
      /*Generate an Identify matrix*/
      std::vector<T> SS = generateEyeMat<T>(N);
      //redistribute into HH and SS into block cyclic layout
      for(std::size_t j = 0; j < nblocks; j++){
        for(std::size_t i = 0; i < mblocks; i++){
          for(std::size_t q = 0; q < c_lens[j]; q++){
	    for(std::size_t p = 0; p < r_lens[i]; p++){
	      H[(q + c_offs_l[j]) * m + p + r_offs_l[i]] = HH[(q + c_offs[j]) * N + p + r_offs[i]];
              S[(q + c_offs_l[j]) * m + p + r_offs_l[i]] = SS[(q + c_offs[j]) * N + p + r_offs[i]];
	    }
	  }
        }
      } 
    }else //if path of matrices are given, load them into H and S by parallel IO
    {
      //read matrix H from local
      readMatrix(H, path_in, "hmat", idx, ".bin", N * N, m, mblocks, nblocks,
         r_offs, r_lens, r_offs_l, c_offs, c_lens, c_offs_l);
      //read matrix S from local
      readMatrix(S, path_in, "smat", idx, ".bin", N * N, m, mblocks, nblocks,
         r_offs, r_lens, r_offs_l, c_offs, c_lens, c_offs_l);
    }

    ed = std::chrono::high_resolution_clock::now();
    elapsed[0][idx - 1] = std::chrono::duration_cast<std::chrono::duration<double>>(ed - st_loc);
 
    st_loc = std::chrono::high_resolution_clock::now();

    // Transform to standard problem using SCALAPACK
    // Cholesky Factorization of S = L * L^T, S is overwritten by L
    t_ppotrf<T>('U', N, S, sze_one, sze_one, desc);

    ed = std::chrono::high_resolution_clock::now();
    elapsed[1][idx - 1] = std::chrono::duration_cast<std::chrono::duration<double>>(ed - st_loc);

    st_loc = std::chrono::high_resolution_clock::now();

    // Reduce H * X = eig * S ( X to the standard from H' * X' = eig * X'
    // with H' = L^{-1} * H * (L^T)^{-1}
    t_psyhegst<T>(i_one, 'U', N, H, sze_one, sze_one, desc, S, sze_one, sze_one, desc, &scale);

    ed = std::chrono::high_resolution_clock::now();
    elapsed[2][idx - 1] = std::chrono::duration_cast<std::chrono::duration<double>>(ed - st_loc);

    // Copy H into single.matrix()
    std::memcpy(matrix, H, m * n * sizeof(T));

    // for the first problem, generate randomly the initial guess of V
    if (idx == bgn) {
      config.SetApprox(false);
      //random generated initial guess of V
      for (std::size_t i = 0; i < N * (nev + nex); ++i) {
        V[i] = T(d(gen), d(gen));
      }
    }else{
      //use the eigenpairs from last problem
      config.SetApprox(true);
    }

    PerformanceDecoratorChase<T> performanceDecorator(&single);

    MPI_Barrier(MPI_COMM_WORLD);

    st_loc = std::chrono::high_resolution_clock::now();

    // ChASE to solve the standard eigenproblem H' * X' = eig * X'
    chase::Solve(&performanceDecorator);

    MPI_Barrier(MPI_COMM_WORLD);

    ed = std::chrono::high_resolution_clock::now();
    elapsed[3][idx - 1] = std::chrono::duration_cast<std::chrono::duration<double>>(ed - st_loc);

    if (rank == 0) {
      std::cout << "\n]> Output of ChASE for Problem #" << idx << "\n";
      performanceDecorator.GetPerfData().print();
      Base<T>* resid = single.GetResid();
      std::cout << "]> Printing first 5 eigenvalues and residuals\n";
      std::cout
          << "| Index |       Eigenvalue      |         Residual      |\n"
          << "|-------|-----------------------|-----------------------|\n";
      std::size_t width = 20;
      std::cout << std::setprecision(12);
      std::cout << std::setfill(' ');
      std::cout << std::scientific;
      std::cout << std::right;
      for (auto i = 0; i < std::min(std::size_t(5), nev); ++i)
        std::cout << "|  " << std::setw(4) << i + 1 << " | " << std::setw(width)
                  << Lambda[i] << "  | " << std::setw(width) << resid[i]
                  << "  |\n";
      std::cout << "\n";
    }


    //Scalapack part
    //In ChASE, the eigenvectors V is stored rebundantly on each proc
    //In order to recover the generalized eigenvectors by Scalapack, it should be
    //redistributed into block-cyclic format. We re-use H to restore V.
    //this part is in parallel implicitly
    
    st_loc = std::chrono::high_resolution_clock::now();

    for(std::size_t j = 0; j < nblocks; j++){
      for(std::size_t i = 0; i < mblocks; i++){
        for(std::size_t q = 0; q < c_lens[j]; q++){
	  for(std::size_t p = 0; p < r_lens[i]; p++){
	    if((q + c_offs[j]) * N + p + r_offs[i] < (nev + nex) * N){
	      H[(q + c_offs_l[j]) * m + p + r_offs_l[i]] = V[(q + c_offs[j]) * N + p + r_offs[i]];
	    }
	  }
	}
      }
    }

    //Now the first (nev+nex) columns of H (a global view) is overwritten by V   
    //Recover the genealized eigenvectors X by solving X' = L^T * X

    t_ptrtrs<T>('U','N','N', N, nev + nex, S, sze_one, sze_one, desc, H, sze_one, sze_one, desc);

    ed = std::chrono::high_resolution_clock::now();
    elapsed[4][idx - 1] = std::chrono::duration_cast<std::chrono::duration<double>>(ed - st_loc);
    elapsed[5][idx - 1] = std::chrono::duration_cast<std::chrono::duration<double>>(ed - st);

    //copy V_dist -> V;
    //restore eigenvectors from scalapack form to ChASE form

    if (rank == 0) {
      std::cout << "]> Finished Problem #" << idx << "\n";
      std::cout << "**********************"  << std::endl;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if(rank == 0){
    std::cout << "\nSUMMARY : Time (s)" << "\n";
    std::cout
          << "---------------------------------------------------------------------------------------------------------------------------------------------------------------\n"
          << "|  Index|             Parallel IO|  Cholesky Factorization|    Transfer to Standard|            ChASE Solver|          Back Transform|                     All|\n"
          << "|-------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|";
    std::cout << std::endl;

    std::cout << std::setprecision(6);
    std::cout << std::setfill(' ');
    std::cout << std::scientific;
    for(auto idx = bgn - 1; idx < end; ++idx){
      std::cout << "|" << std::setw(7) << idx + 1 << "|"; 
      for(auto i = 0; i < 6; i++){
	std::cout << std::setw(24) << elapsed[i][idx].count() << "|";
      }
      std::cout << std::endl;
    }  

    std::cout
            << "---------------------------------------------------------------------------------------------------------------------------------------------------------------"
	    << std::endl;
  }

  return 0;
}

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  ChASE_DriverProblemConfig conf;

  po::options_description desc("ChASE Options");

  desc.add_options()(                                                     //
      "help,h",                                                           //
      "show this message"                                                 //
      )(                                                                  //
      "n", po::value<std::size_t>(&conf.N)->required(),                   //
      "Size of the Input Matrix"                                          //
      )(                                                                  //
      "double", po::value<bool>(&conf.isdouble)->default_value(true),     //
      "Is matrix complex double valued, false indicates the single type"  //
      )(                                                                  //
      "complex", po::value<bool>(&conf.complex)->default_value(true),     //
      "Matrix is complex valued"                                          //
      )(                                                                  //
      "nev", po::value<std::size_t>(&conf.nev)->required(),               //
      "Wanted Number of Eigenpairs"                                       //
      )(                                                                  //
      "nex", po::value<std::size_t>(&conf.nex)->default_value(25),        //
      "Extra Search Dimensions"                                           //
      )(                                                                  //
      "deg", po::value<std::size_t>(&conf.deg)->default_value(20),        //
      "Initial filtering degree"                                          //
      )(                                                                  //
      "bgn", po::value<std::size_t>(&conf.bgn)->default_value(1),         //
      "Start ell"                                                         //
      )(                                                                  //
      "end", po::value<std::size_t>(&conf.end)->default_value(1),         //
      "End ell"                                                           //
      )(                                                                  //
      "tol", po::value<double>(&conf.tol)->default_value(1e-10),          //
      "Tolerance for Eigenpair convergence"                               //
      )(                                                                  //
      "path_in", po::value<std::string>(&conf.path_in)->default_value(""),//
      "Path to the input matrix/matrices"                                 //
      )(                                                                  //
      "mode", po::value<std::string>(&conf.mode)->default_value("A"),     //
      "valid values are R(andom) or A(pproximate)"                        //
      )(                                                                  //
      "opt", po::value<std::string>(&conf.opt)->default_value("S"),       //
      "Optimi(S)e degree, or do (N)ot optimise"                           //
      )(                                                                  //
      "path_eigp", po::value<std::string>(&conf.path_eigp),               //
      "Path to approximate solutions, only required when mode"            //
      "is Approximate, otherwise not used"                                //
      )(                                                                  //
      "sequence", po::value<bool>(&conf.sequence)->default_value(false),  //
      "Treat as sequence of Problems. Previous ChASE solution is used,"   //
      "when available"                                                    //
      )(                                                                  //
      "mbsize", po::value<std::size_t>(&conf.mbsize)->default_value(50),  //
      "block size for the row"                                            //
      )(                                                                  //
      "nbsize", po::value<std::size_t>(&conf.nbsize)->default_value(50),  //
      "block size for the column"                                         //
      )(                                                                  //
      "dim0", po::value<int>(&conf.dim0)->default_value(0),               //
      "row number of MPI proc grid"                                       //
      )(                                                                  //
      "dim1", po::value<int>(&conf.dim1)->default_value(0),               //
      "column number of MPI proc grid"                                    //
      )(								  //
      "irsrc", po::value<int>(&conf.irsrc)->default_value(0),             //
      "The process row over which the first row of matrix is"             //
      "distributed."                                                      //
      )(                                                                  //
      "icsrc", po::value<int>(&conf.icsrc)->default_value(0),             //
      "The process column over which the first column of the array A is"  //
      "distributed."                                                      //
      )(                                                                  //
      "major", po::value<std::string>(&conf.major)->default_value("C"),    //
      "Major of MPI proc grid, valid values are R(ow) or C(olumn)"        //
      );

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);

  // print help
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  po::notify(vm);
  conf.mode = toupper(conf.mode.at(0));
  conf.opt = toupper(conf.opt.at(0));

  if (conf.bgn > conf.end) {
    std::cout << "Begin must be smaller than End!" << std::endl;
    return -1;
  }

  if (conf.mode != "R" && conf.mode != "A") {
    std::cout << "Illegal value for mode: \"" << conf.mode << "\"" << std::endl
              << "Legal values are R or A" << std::endl;
    return -1;
  }

  if (conf.opt != "N" && conf.opt != "S") {
    std::cout << "Illegal value for opt: " << conf.opt << std::endl
              << "Legal values are N, S" << std::endl;
    return -1;
  }

  if (conf.path_eigp.empty() && conf.mode == "A") {
    std::cout << "eigp is required when mode is " << conf.mode << std::endl;
    return -1;
  }

  if (conf.isdouble) {
    do_chase_gev<std::complex<double>>(conf);
  } else {
    std::cout << "single not implemented\n";
  }

  MPI_Finalize();

}

