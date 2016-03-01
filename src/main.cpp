#include "../include/chfsi.h"
#include "../include/testresult.h"

#include <random>
#include <boost/program_options.hpp>
#include <mkl.h> // todo: use lapacke and cblas

namespace po = boost::program_options;

template<typename T>
void readMatrix( T *H, std::string path_in, int index,
                 std::string suffix, int size )
{
  std::ostringstream problem(std::ostringstream::ate);
  problem << path_in << "gmat  1 " << setw(2) << index << suffix;

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

  string path_in;
  string mode;
  string opt;
  string arch;
  string path_eigp;
  string path_out;
  string path_name;

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
    ("path_in", po::value<string>(&path_in)->required(), "TODO")
    ("mode", po::value<string>(&mode)->default_value("A"), "valid values are R[andom] or A[pproximate]")
    ("opt", po::value<string>(&opt)->default_value("S"), "TODO")
    ("path_eigp", po::value<string>(&path_eigp), "TODO")
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

  int nevex = nev + nex; // Block size for the algorithm.

  // Matrix representing the generalized eigenvalue problem.
  MKL_Complex16 *H = new MKL_Complex16[N*N];
  // Matrix which stores the approximate eigenvectors (INPUT to chfsi).
  MKL_Complex16 *V = new MKL_Complex16[N*nevex];
  // Matrix which stores the eigenvectors (OUTPUT of the chfsi function).
  MKL_Complex16 *W = new MKL_Complex16[N*nevex];
  // eigenvalues
  double * Lambda = new double[nevex];

  //----------------------------------------------------------------------------
  //----------------------------------------------------------------------------
  //std::random_device rd;
  std::mt19937 gen(2342.0);
  std::normal_distribution<> d;

  for (int i = bgn; i <= end ; ++i)
  {
    if( i == bgn || !sequence )
    {
      // I don't think I want to support this at the moment:
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
        /*
          VSLStreamStatePtr randomStream;
          vslNewStream(&randomStream, VSL_BRNG_MT19937, 677);
          vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, randomStream, 2*N*nevex, (double*)V, 0.0, 1.0);
          vslDeleteStream(&randomStream);
        */

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

    //------------------------------SOLVE-CURRENT-PROBLEM-----------------------
    readMatrix( H, path_in, i, ".bin", N*N);
    chfsi(H, N, V, W, Lambda, nev, nex, deg, tol, mode[0], opt[0]);
    //--------------------------------------------------------------------------

    int iterations;
    int filteredVecs;
    get_iteration(&iterations); // Get the number of iterations.
    get_filteredVecs(&filteredVecs);

    TR.registerValue( i, "filteredVecs", filteredVecs );
    TR.registerValue( i, "iterations", iterations );

    //-------------------------- Calculate Residuals ---------------------------
    memcpy(V, W, sizeof(MKL_Complex16)*N*nev);
    MKL_Complex16 one(1.0);
    MKL_Complex16 zero(0.0);
    MKL_Complex16 eigval;
    const int* iOne = new int(1);
    for(int ttz = 0; ttz<nev;ttz++){
      eigval = -1.0 * Lambda[ttz];
      zscal(&N,&eigval,W+ttz*N, iOne);
    }
    zhemm("L","L", &N, &nev, &one, H, &N, V, &N, &one, W, &N);
    double norm = zlange("M", &N, &nev, W, &N, NULL);
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

    zgemm("C", "N", &nev, &nev, &N, &one, V, &N, V, &N, &neg_one, unity, &nev);
    double norm2 = zlange("M", &nev, &nev, unity, &nev, NULL);
    TR.registerValue( i, "orth", norm2);

  } // for(int i = bgn; i <= end; ++i)


  TR.done();

  delete[] H;
  delete[] V; delete[] W;
  delete[] Lambda;
  //delete[] zmem; delete[] dmem; delete[] imem;
  //delete[] isuppz;

  return 0;
}
