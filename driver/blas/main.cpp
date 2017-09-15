/* -*- Mode: C++; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set ts=8 sts=2 et sw=2 tw=80: */
/* TODO License */

// #include "chase_blas.hpp"
#include <boost/program_options.hpp>
#include <random>
#include "chase_blas_factory.hpp"
#include "testresult.hpp"

// typedef std::complex<double> T;

/*
extern "C" {
template <typename T>
void chase_write_hdf5(MPI_Comm comm, T* H, size_t N);
template <typename T>
void chase_read_matrix(MPI_Comm comm, std::size_t xoff, std::size_t yoff,
                       std::size_t xlen, std::size_t ylen, T* H);
}
*/
namespace po = boost::program_options;

template <typename T>
T getRandomT(std::function<double(void)> f);

template <>
double getRandomT(std::function<double(void)> f) {
  return double(f());
}

template <>
float getRandomT(std::function<double(void)> f) {
  return float(f());
}

template <>
std::complex<double> getRandomT(std::function<double(void)> f) {
  return std::complex<double>(f(), f());
}

template <>
std::complex<float> getRandomT(std::function<double(void)> f) {
  return std::complex<float>(f(), f());
}

template <typename T>
void readMatrix(T* H, std::string path_in, std::string spin, std::size_t kpoint,
                std::size_t index, std::string suffix, std::size_t size,
                bool legacy) {
  std::ostringstream problem(std::ostringstream::ate);
  if (legacy)
    problem << path_in << "gmat  1 " << std::setw(2) << index << suffix;
  else
    problem << path_in << "mat_" << spin << "_" << std::setfill('0')
            << std::setw(2) << kpoint << "_" << std::setfill('0')
            << std::setw(2) << index << suffix;

  std::cout << problem.str() << std::endl;
  std::ifstream input(problem.str().c_str(), std::ios::binary);
  if (input.is_open()) {
    input.read((char*)H, sizeof(T) * size);
  } else {
    throw std::string("error reading file: ") + problem.str();
  }
}

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

  std::size_t kpoint;
  bool legacy;
  std::string spin;

  bool complex;
  bool isdouble;
};

template <typename T>
int do_chase(ChASE_DriverProblemConfig& conf, TestResult& TR) {
  // todo due to legacy reasons we unpack the struct
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

  std::size_t kpoint = conf.kpoint;
  bool legacy = conf.legacy;
  std::string spin = conf.spin;

  //----------------------------------------------------------------------------
  std::cout << std::setprecision(16);

  ChASE_Config<T> config(N, nev, nex);
  config.setTol(tol);
  config.setDeg(deg);
  config.setOpt(opt == "S");
  config.setApprox(mode == "A");

  auto V__ = std::unique_ptr<T[]>(new T[N * (nev + nex)]);
  auto Lambda__ = std::unique_ptr<Base<T>[]>(new Base<T>[(nev + nex)]);

  T* V = V__.get();
  Base<T>* Lambda = Lambda__.get();

  // ChASE_Blas<T>* single = new ChASE_Blas<T>(config);
  // std::unique_ptr<ChASE_Blas<T>> single_ =
  //     ChASEFactory<T>::constructChASE(config, NULL, NULL,
  //     MPI_COMM_WORLD);
  std::unique_ptr<ChASE_Blas<T>> single_ = ChASEFactory<T>::constructChASE(
      config, nullptr, V, Lambda, MPI_COMM_WORLD);

  ChASE_Blas<T>* single = single_.get();

  // std::random_device rd;
  std::mt19937 gen(2342.0);
  std::normal_distribution<> d;

  T* H = single->getMatrixPtr();

  // the example matrices are stored in std::complex< double >
  // so we read them as such and then case them
  // std::complex<double>* _H = new MKL_Complex16[N * N];

  for (auto i = bgn; i <= end; ++i) {
    if (i == bgn || !sequence) {
      /*
  if (path_eigp == "_" && int_mode == OMP_APPROX && i == bgn )
  { // APPROX. No approximate pairs given.
  //-------------------------SOLVE-PREVIOUS-PROBLEM-------------------------
  app = ".bin"; // Read the matrix of the previous problem.
  myreadwrite<MKL_Complex16>(H, path_in.c_str(), app.c_str(), i-1, N*N,
  'r');

  // Solve the previous problem, store the eigenpairs in V and Lambda.
  ZHEEVR("V", "I", "L", &N, H, &N, &vl, &vu, &il, &iu, &tol,
  &notneeded, Lambda, V, &N, isuppz, zmem, &lzmem, dmem, &ldmem,
  imem, &limem, &INFO);

  //------------------------------------------------------------------------
  // In next iteration the solutions to this one will be used as
  approximations.
  path_eigp = path_out;
  }
  else */
      if (mode[0] == 'A') {  // APPROX. Approximate eigenpairs given.
        //-----------------------READ-APPROXIMATE-EIGENPAIRS--------------------
        readMatrix(V, path_eigp, spin, kpoint, i - 1, ".vct", N * (nev + nex),
                   legacy);
        readMatrix(Lambda, path_eigp, spin, kpoint, i - 1, ".vls", (nev + nex),
                   legacy);
        //----------------------------------------------------------------------
      } else {  // RANDOM.
        // Randomize V.
        for (std::size_t i = 0; i < N * (nev + nex); ++i) {
          // V[i] = T(d(gen)); // TODO
          V[i] = getRandomT<T>([&]() { return d(gen); });
        }
        // Set Lambda to zeros. ( Lambda = zeros(N,1) )
        for (int j = 0; j < (nev + nex); j++) Lambda[j] = 0.0;
      }
    } else {
      // when doing a sequence and we are not in the first iteration, we use the
      // previous solution
      mode = "A";
    }

    CHASE_INT xoff;
    CHASE_INT yoff;
    CHASE_INT xlen;
    CHASE_INT ylen;

    // int size = 1;
    // MPI_Comm_size(MPI_COMM_WORLD, &size);

    // if (size > 1) {
    //     std::cout << "reading from hdf5\n";
    //     single->get_off(&xoff, &yoff, &xlen, &ylen);
    //     chase_read_matrix(MPI_COMM_WORLD, xoff, yoff, xlen, ylen, H);
    // } else {
    std::cout << "reading from plain file\n";
    readMatrix(H, path_in, spin, kpoint, i, ".bin", N * N, legacy);
    //}

    // the input is complex double so we cast to T
    // for (std::size_t idx = 0; idx < N * N; ++idx) H[idx] = _H[idx];

    Base<T> normH;
    normH = std::max(t_lange('1', N, N, H, N), Base<T>(1.0));

    // normH = 9;
    //    std::cout << "setting norm to static value " << normH << "\n";

    single->setNorm(normH);

    //------------------------------SOLVE-CURRENT-PROBLEM-----------------------
    single->solve();
    //--------------------------------------------------------------------------

    ChASE_PerfData perf = single->getPerfData();

    std::size_t iterations = perf.get_iter_count();
    std::size_t filteredVecs = perf.get_filtered_vecs();

    TR.registerValue(i, "filteredVecs", filteredVecs);
    TR.registerValue(i, "iterations", iterations);

    //-------------------------- Calculate Residuals ---------------------------
    Base<T> resd = single->residual();
    Base<T> orth = single->orthogonality();

    TR.registerValue(i, "resd", resd);
    TR.registerValue(i, "orth", orth);

    perf.print();

    if (resd > nev * normH * tol ||
        orth > (std::numeric_limits<Base<T>>::epsilon() * 100)) {
      std::cout << "resd:" << resd << "(" << nev * normH * tol << ")\n"
                << "orth: " << orth << "("
                << std::numeric_limits<Base<T>>::epsilon() * 100 << "\n";
      throw new std::exception();
    }

  }  // for(int i = bgn; i <= end; ++i)

#ifdef PRINT_EIGENVALUES
  std::cout << "Eigenvalues: " << std::endl;
  for (int zzt = 0; zzt < nev; zzt++)
    std::cout << std::setprecision(16) << Lambda[zzt] << std::endl;
  std::cout << "End of eigenvalues. " << std::endl;
#endif

  // delete single;
  // delete[] _H;

  return 0;
}

int main(int argc, char* argv[]) {
#ifdef HAS_MPI
  MPI_Init(NULL, NULL);
#endif

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
      "bgn", po::value<std::size_t>(&conf.bgn)->default_value(2),         //
      "Start ell"                                                         //
      )(                                                                  //
      "end", po::value<std::size_t>(&conf.end)->default_value(2),         //
      "End ell"                                                           //
      )(                                                                  //
      "spin", po::value<std::string>(&conf.spin)->default_value("d"),     //
      "spin"                                                              //
      )(                                                                  //
      "kpoint", po::value<std::size_t>(&conf.kpoint)->default_value(0),   //
      "kpoint"                                                            //
      )(                                                                  //
      "tol", po::value<double>(&conf.tol)->default_value(1e-10),          //
      "Tolerance for Eigenpair convergence"                               //
      )(                                                                  //
      "path_in", po::value<std::string>(&conf.path_in)->required(),       //
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
      "legacy", po::value<bool>(&conf.legacy)->default_value(false),      //
      "Use legacy naming scheme?");                                       //

  std::string testName;
  po::options_description testOP("Test options");
  testOP.add_options()(                                       //
      "write",                                                //
      "Write Profile"                                         //
      )(                                                      //
      "name", po::value<std::string>(&testName)->required(),  //
      "Name of the testing profile");                         //

  desc.add(testOP);

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);

  // print help
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  po::notify(vm);

  // some normalization
  conf.mode = toupper(conf.mode.at(0));
  conf.opt = toupper(conf.opt.at(0));

  // Additional Error checks
  // TODO this should be a member of struct
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
    // TODO verify that eigp is a valid path
    return -1;
  }

  // -----Validation of test----
  bool testResultCompare;
  if (vm.count("write"))
    testResultCompare = false;
  else
    testResultCompare = true;

  TestResult TR(testResultCompare, testName, conf.N, conf.nev, conf.nex,
                conf.deg, conf.tol, conf.mode[0], conf.opt[0], conf.sequence);

  //   if (conf.complex) {
  if (conf.isdouble)
    do_chase<std::complex<double>>(conf, TR);
  else  // single
    do_chase<std::complex<float>>(conf, TR);
  // } else {
  //   if (conf.isdouble)
  //     do_chase<double>(conf, TR);
  //   else  // single
  //     do_chase<float>(conf, TR);
  // }

  TR.done();

#ifdef HAS_MPI
  MPI_Finalize();
#endif
  return 0;
}
