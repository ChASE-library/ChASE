#include "chase_blas.hpp"
#include "testresult.hpp"
//#include "../include/lanczos.h"


#include <random>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

template<typename T>
void readMatrix( T *H, std::string path_in, std::string spin, std::size_t kpoint, std::size_t index,
                 std::string suffix, std::size_t size, bool legacy )
{
  std::ostringstream problem(std::ostringstream::ate);
  if(legacy)
    problem << path_in << "gmat  1 " << std::setw(2) << index << suffix;
  else
    problem << path_in << "mat_" << spin << "_" << std::setfill('0') << std::setw(2)
            << kpoint << "_" << std::setfill('0') << std::setw(2) << index << suffix;

  std::cout << problem.str() << std::endl;
  std::ifstream input( problem.str().c_str(), std::ios::binary );
  if(input.is_open()) {
    input.read((char *) H, sizeof(T) * size);
  } else {
    throw std::string("error reading file: ") + problem.str();
  }
}

typedef std::complex< double > T;

int main(int argc, char* argv[])
{

  std::size_t N;
  std::size_t nev;
  std::size_t nex;
  std::size_t deg;
  std::size_t bgn;
  std::size_t end;

  double tol;
  bool sequence;

  std::string path_in;
  std::string mode;
  std::string opt;
  std::string arch;
  std::string path_eigp;
  std::string path_out;
  std::string path_name;

  std::size_t kpoint;
  bool legacy;
  std::string spin;

  po::options_description desc("ChASE Options");
  desc.add_options()
    ("help,h", "show this message")
    ("n", po::value<std::size_t>(&N)->required(), "Size of the Input Matrix")
    ("nev", po::value<std::size_t>(&nev)->required(), "Wanted Number of Eigenpairs")
    ("nex", po::value<std::size_t>(&nex)->default_value(25), "Extra Search Dimensions")
    ("deg", po::value<std::size_t>(&deg)->default_value(20), "Initial filtering degree")
    ("bgn", po::value<std::size_t>(&bgn)->default_value(2), "Start ell")
    ("end", po::value<std::size_t>(&end)->default_value(2), "End ell")
    ("spin", po::value<std::string>(&spin)->default_value("d"), "spin")
    ("kpoint", po::value<std::size_t>(&kpoint)->default_value(0), "kpoint")
    ("tol", po::value<double>(&tol)->default_value(1e-10), "Tolerance for Eigenpair convergence")
    ("path_in", po::value<std::string>(&path_in)->required(), "Path to the input matrix/matrices")
    ("mode", po::value<std::string>(&mode)->default_value("A"), "valid values are R(andom) or A(pproximate)")
    ("opt", po::value<std::string>(&opt)->default_value("S"), "Optimi(S)e degree, or do (N)ot optimise")
    ("path_eigp", po::value<std::string>(&path_eigp), "Path to approximate solutions, only required when mode is Approximate, otherwise not used")
    ("sequence", po::value<bool>(&sequence)->default_value(false), "Treat as sequence of Problems. Previous ChASE solution is used, when available")
    ("legacy", po::value<bool>(&legacy)->default_value(false), "Use legacy naming scheme?")
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

  if( opt != "N" && opt != "S" )
  {
    std::cout
      << "Illegal value for opt: " << opt << std::endl
      << "Legal values are N, S" << std::endl;
    return -1;
  }

  if( path_eigp.empty() && mode == "A" )
  {
    std::cout << "eigp is required when mode is " << mode << std::endl;
    // TODO verify that eigp is a valid path
    return -1;
  }

  // -----Validation of test----
  bool testResultCompare;
  if( vm.count("write") )
    testResultCompare = false;
  else
    testResultCompare = true;

  TestResult TR(
    testResultCompare,
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

  //----------------------------------------------------------------------------

  ChASE_Config config( N, nev, nex );
  config.setTol( tol );
  config.setDeg( deg );
  config.setOpt( opt == "S" );

  ChASE_Blas< T > *single = new ChASE_Blas< T >( config );

  //std::random_device rd;
  std::mt19937 gen(2342.0);
  std::normal_distribution<> d;

  T *V = single->getVectorsPtr();
  T *H = single->getMatrixPtr();
  Base< T > * Lambda = single->getRitzv();

  // the example matrices are stored in std::complex< double >
  // so we read them as such and then case them
  std::complex< double > *_H = new MKL_Complex16[N*N];

  for (auto i = bgn; i <= end ; ++i)
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
      if (mode[0] == 'A')
      { // APPROX. Approximate eigenpairs given.
        //-----------------------READ-APPROXIMATE-EIGENPAIRS----------------------
        readMatrix( V, path_eigp, spin, kpoint, i-1, ".vct", N*(nev+nex), legacy );
        readMatrix( Lambda, path_eigp, spin, kpoint, i-1, ".vls", (nev+nex), legacy );
        //------------------------------------------------------------------------
      }
      else
      { // RANDOM.
        // Randomize V.
        for( std::size_t i=0; i < N*(nev+nex); ++i)
        {
          V[i] = T( d(gen), d(gen) );
        }
        // Set Lambda to zeros. ( Lambda = zeros(N,1) )
        for(int j=0; j<(nev+nex); j++)
          Lambda[j]=0.0;
      }
    }
    else
    {
      // when doing a sequence and we are not in the first iteration, we use the
      // previous solution
      mode = "A";
    }
    readMatrix( _H, path_in, spin, kpoint, i, ".bin", N*N, legacy);
    // the input is complex double so we cast to T
    for( std::size_t idx = 0; idx < N*N; ++idx )
      H[idx] = _H[idx];

    Base< T > normH = std::max(t_lange('1', N, N, H, N), Base< T >(1.0) );
    single->setNorm( normH );

    //------------------------------SOLVE-CURRENT-PROBLEM-----------------------
    single->solve();
    //--------------------------------------------------------------------------

    ChASE_PerfData perf = single->getPerfData();

    std::size_t iterations = perf.get_iter_count();
    std::size_t filteredVecs = perf.get_filtered_vecs();

    TR.registerValue( i, "filteredVecs", filteredVecs );
    TR.registerValue( i, "iterations", iterations );

    //-------------------------- Calculate Residuals ---------------------------
    Base<T> resd = single->residual();
    Base<T> orth = single->orthogonality();

    TR.registerValue( i, "resd", resd);
    TR.registerValue( i, "orth", orth);

    perf.print();

    if( resd > nev*normH*tol || orth > 1e-14 )
      throw new std::exception();

  } // for(int i = bgn; i <= end; ++i)


  TR.done();

#ifdef PRINT_EIGENVALUES
  std::cout << "Eigenvalues: " << std::endl;
  for(int zzt = 0; zzt < nev; zzt++)
    std::cout << std::setprecision(16) << Lambda[zzt] << std::endl;
  std::cout << "End of eigenvalues. " << std::endl;
#endif


  delete single;
  delete[] _H;

  return 0;
}
