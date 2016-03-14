#include "../include/chase.h"
#include "../include/testresult.h"
#include "../include/lanczos.h"

#include <random>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

template<typename T>
void readMatrix( T *H, std::string path_in, int index,
                 std::string suffix, int size )
{
  std::ostringstream problem(std::ostringstream::ate);
  problem << path_in << "gmat  1 " << std::setw(2) << index << suffix;

  std::ifstream input( problem.str().c_str(), std::ios::binary );

  if(input.is_open())
  {
    input.read((char *) H, sizeof(T) * size);
  } else {
    throw std::string("error reading file: ") + problem.str();
  }
}

int main(int argc, char* argv[])
{

  int N;
  int nev;
  int nex;
  int deg;
  int bgn;
  int end;

  double tol;
  bool sequence;

  std::string path_in;
  std::string mode;
  std::string opt;
  std::string arch;
  std::string path_eigp;
  std::string path_out;
  std::string path_name;

  po::options_description desc("ChASE Options");
  desc.add_options()
    ("help,h", "show this message")
    ("n", po::value<int>(&N)->required(), "Size of the Input Matrix")
    ("nev", po::value<int>(&nev)->required(), "nev")
    ("nex", po::value<int>(&nex)->default_value(25), "nex")
    ("deg", po::value<int>(&deg)->default_value(20), "deg")
    ("bgn", po::value<int>(&bgn)->default_value(2), "TODO")
    ("end", po::value<int>(&end)->default_value(2), "TODO")
    ("tol", po::value<double>(&tol)->default_value(1e-10), "TODO")
    ("path_in", po::value<std::string>(&path_in)->required(), "TODO")
    ("mode", po::value<std::string>(&mode)->default_value("A"), "valid values are R[andom] or A[pproximate]")
    ("opt", po::value<std::string>(&opt)->default_value("S"), "TODO")
    ("path_eigp", po::value<std::string>(&path_eigp), "TODO")
    ("sequence", po::value<bool>(&sequence)->default_value(false), "TODO")
    ;

  std::string testName;
  po::options_description testOP("Test options");
  testOP.add_options()
    ("write", "Write Profile")
    ("name", po::value<std::string>(&testName)->required(), "Name of the testing profile")
    ;

  desc.add(testOP);

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    std::cout
      << desc << std::endl;
    return 1;
  }

  try {
    po::notify(vm);
  }
  catch( std::exception &e )
  {
    std::cout
      << e.what() << std::endl
      << std::endl
      << desc << std::endl;
    return -1;
  }

  mode = toupper(mode.at(0));
  opt = toupper(opt.at(0));

  std::cout << path_in << " " << path_eigp << std::endl;


  if( bgn > end )
  {
    std::cout << "Begin must be smaller than End!" << std::endl;
    return -1;
  }

  if( mode != "R" && mode != "A" )
  {
    std::cout
      << "Illegal value for mode: \"" << mode << "\"" << std::endl
      << "Legal values are R or A" << std::endl;
    return -1;
  }

  if( opt != "N" && opt != "S" && opt != "M" )
  {
    std::cout
      << "Illegal value for opt: " << opt << std::endl
      << "Legal values are N, S, M" << std::endl;
    return -1;
  }

  if( path_eigp.empty() && mode == "A" )
  {
    std::cout << "eigp is required when mode is " << mode << std::endl;
    // TODO verify that eigp is a valid path
    return -1;
  }

  //----------------------------------------------------------------------------

  HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceScheduleYield));
  magma_init();


  // -----Validation of test----
  bool testResultWrite;
  if( vm.count("write") )
    testResultWrite = CHASE_TESTRESULT_WRITE;
  else
    testResultWrite = CHASE_TESTRESULT_COMPARE;

  TestResult TR(
    testResultWrite,
    testName,
    N,
    nev,
    nex,
    deg,
    tol,
    mode[0],
    opt[0],
    sequence
    );

  const int nevex = nev + nex; // Block size for the algorithm.

  // Matrix representing the generalized eigenvalue problem.
  MKL_Complex16 *H = new MKL_Complex16[N*N];
  // Matrix which stores the approximate eigenvectors
  MKL_Complex16 *V = new MKL_Complex16[N*nevex];
  // Matrix which stores the eigenvectors
  MKL_Complex16 *W = new MKL_Complex16[N*nevex];
  // eigenvalues
  double * Lambda = new double[nevex];
  int *degrees  = new int[nevex];

  //----------------------------------------------------------------------------
  //std::random_device rd;
  std::mt19937 gen(2342.0);
  std::normal_distribution<> d;

  for (int i = bgn; i <= end ; ++i)
  {
    if( i == bgn || !sequence )
    {
      /*
        if (path_eigp == "_" && int_mode == OMP_APPROX && i == bgn )
        { // APPROX. No approximate pairs given.
        //-------------------------SOLVE-PREVIOUS-PROBLEM-------------------------
        app = ".bin"; // Read the matrix of the previous problem.
        myreadwrite<MKL_Complex16>(H, path_in.c_str(), app.c_str(), i-1, N*N, 'r');

        // Solve the previous problem, store the eigenpairs in V and Lambda.
        ZHEEVR("V", "I", "L", &N, H, &N, &vl, &vu, &il, &iu, &tol,
        &notneeded, Lambda, V, &N, isuppz, zmem, &lzmem, dmem, &ldmem,
        imem, &limem, &INFO);

        //------------------------------------------------------------------------
        // In next iteration the solutions to this one will be used as approximations.
        path_eigp = path_out;
        }
        else */
      if (mode[0] == CHASE_MODE_APPROX)
      { // APPROX. Approximate eigenpairs given.
        //-----------------------READ-APPROXIMATE-EIGENPAIRS----------------------
        readMatrix( V, path_eigp, i-1, ".vct", N*nevex );
        readMatrix( Lambda, path_eigp, i-1, ".vls", nevex );
        //------------------------------------------------------------------------
      }
      else
      { // RANDOM.
        // Randomize V.
        for( std::size_t i=0; i < N*nevex; ++i)
        {
          V[i] = std::complex<double>( d(gen), d(gen) );
        }
        // Set Lambda to zeros. ( Lambda = zeros(N,1) )
        for(int j=0; j<nevex; j++)
          Lambda[j]=0.0;
      }
    }
    else
    {
      // when doing a sequence and we are not in the first iteration, we use the
      // previous solution
      mode = "A";
    }

    readMatrix( H, path_in, i, ".bin", N*N);


    // test lanczos
    {
      double upperb;
      double *ritzv_;
      MKL_Complex16 *V_;
      int num_its = 10;
      int numvecs = 4;
      if( mode[0] == CHASE_MODE_RANDOM )
      {
        ritzv_ = new double[nevex];
        V_ = new MKL_Complex16[num_its*N];
        num_its = 40;
      }

      lanczos( H, N, numvecs, num_its, nevex, &upperb,
               mode[0] == CHASE_MODE_RANDOM,
               ritzv_, V_);

      if( mode[0] == CHASE_MODE_RANDOM )
      {
        double lambda = * std::min_element( ritzv_, ritzv_ + nevex );
        double lowerb = * std::max_element( ritzv_, ritzv_ + nevex );
        TR.registerValue( i, "lambda1", lambda );
        TR.registerValue( i, "lowerb", lowerb );
        delete[] ritzv_;
        delete[] V_;
      }

      TR.registerValue( i, "upperb", upperb );
    }

    //------------------------------SOLVE-CURRENT-PROBLEM-----------------------
    chase(H, N, V, W, Lambda, nev, nex, deg, degrees, tol, mode[0], opt[0]);
    //--------------------------------------------------------------------------

    int iterations = get_iter_count();
    int filteredVecs = get_filtered_vecs();

    TR.registerValue( i, "filteredVecs", filteredVecs );
    TR.registerValue( i, "iterations", iterations );

    //-------------------------- Calculate Residuals ---------------------------
    memcpy(W, V, sizeof(MKL_Complex16)*N*nev);
    MKL_Complex16 one(1.0);
    MKL_Complex16 zero(0.0);
    MKL_Complex16 eigval;
    int iOne = 1;
    for(int ttz = 0; ttz<nev;ttz++){
      eigval = -1.0 * Lambda[ttz];
      cblas_zscal( N, &eigval, W+ttz*N, 1);
    }
    cblas_zhemm(
      CblasColMajor,
      CblasLeft,
      CblasLower,
      N, nev, &one, H, N, V, N, &one, W, N);
    double norm = LAPACKE_zlange( LAPACK_COL_MAJOR, 'M', N, nev, W, N);
    TR.registerValue( i, "resd", norm);


    // Check eigenvector orthogonality
    MKL_Complex16 *unity = new MKL_Complex16[nev*nev];
    MKL_Complex16 neg_one(-1.0);
    for(int ttz = 0; ttz < nev; ttz++){
      for(int tty = 0; tty < nev; tty++){
        if(ttz == tty) unity[nev*ttz+tty] = 1.0;
        else unity[nev*ttz+tty] = 0.0;
      }
    }

    cblas_zgemm(
      CblasColMajor,
      CblasConjTrans, CblasNoTrans,
      nev, nev, N,
      &one,
      V, N,
      V, N,
      &neg_one,
      unity, nev
      );
    double norm2 = LAPACKE_zlange( LAPACK_COL_MAJOR, 'M', nev, nev, unity, nev);
    TR.registerValue( i, "orth", norm2);

    delete[] unity;

    std::cout << "resd: " << norm << "\torth:" << norm2 << std::endl;

    print_timings();
    reset_clock();
    std::cout << "Filtered Vectors\t\t" << filteredVecs << std::endl;
  } // for(int i = bgn; i <= end; ++i)


  TR.done();
  magma_finalize();

  delete[] H;
  delete[] V; delete[] W;
  delete[] Lambda;
  //delete[] zmem; delete[] dmem; delete[] imem;
  //delete[] isuppz;

  return 0;
}
