#include <chrono>
#include <complex>

using std::chrono::nanoseconds;
using std::chrono::duration_cast;

#include "genera/elemental/ChASE_Elemental.h"
#include "genera/matrixfree/blas_templates.h"

using T = El::Complex<double>;

int main(int argc, char* argv[]) {
  El::Initialize(argc, argv);
  El::mpi::Comm comm = El::mpi::COMM_WORLD;
  El::Grid grid{comm, 1};

  const El::Int N = El::Input<El::Int>("--n", "matrix width");
  const El::Int nev = El::Input<El::Int>("--nev", "EPs");
  const El::Int nex = El::Input<El::Int>("--nex", "additional subspace");
  const std::string path = El::Input<std::string>("--path", "path");
  const El::Int bgn = El::Input<El::Int>("--bgn", "bgn seq");
  const El::Int end = El::Input<El::Int>("--end", "end seq");
  const bool opt = El::Input<bool>("--opt", "optimize?", true);
  El::ProcessInput();

  //*
  // std::size_t N = 3893;
  // std::size_t nev = 256;
  // std::size_t nex = 100;

  El::DistMatrix<El::Complex<double>> H{grid};
  El::Gaussian(H, N, N);

  ChaseConfig<std::complex<double>> config__{N, nev, nex};
  config__.setOpt(opt);
  ChaseConfig<El::Complex<double>>* config =
      reinterpret_cast<ChaseConfig<El::Complex<double>>*>(&config__);

  ElementalChase<El::Complex<double>> single{*config, H};

  for (std::size_t it = bgn; it <= end; ++it) {
    ///////////// READ MATRIX ////////
    std::ostringstream oss;
    oss << path << "/gmat  1 " << std::setw(2) << it << ".bin";
    // El::Read(H, std::string(
    //                 "/homeb/slai/slai00/MATrix/NaCl/size4k/bin/gmat  1
    //                 1.bin"),
    //          El::BINARY_FLAT, true);
    El::Read(H, oss.str(), El::BINARY_FLAT);

    /*/
      std::size_t N = 10;
      std::size_t nev = 2;
      std::size_t nex = 2;

      El::DistMatrix<El::Complex<double>> H{grid};
      El::Ones(H, N, N);

      ChaseConfig<std::complex<double>> config__{N, nev, nex};
      ChaseConfig<El::Complex<double>>* config =
      reinterpret_cast<ChaseConfig<El::Complex<double>>*>(&config__);
    //*/

    single.Solve();

    Base<T> largest_norm = 0;
    El::DistMatrix<Base<T>, El::STAR, El::STAR> ritzv{nev + nex, 1, H.Grid()};

    El::DistMatrix<El::Complex<double>>& V = single.GetV();
    El::DistMatrix<El::Complex<double>> W{V};

    Base<T>* ritzv_ = single.GetRitzv();
    for (std::size_t i = 0; i < nev + nex; ++i) ritzv.Set(i, 0, ritzv_[i]);

    El::DiagonalScale(El::RIGHT, El::NORMAL, ritzv, W);
    El::Hemm(El::LEFT, El::LOWER, T(1.0), H, V, T(-1.0), W);

    for (std::size_t i = 0; i < nev; ++i) {
      auto wi = El::View(W, 0, i, N, 1);

      // || H x - lambda x || / ( max( ||H||, |lambda| ) )
      // norm_2 <- || H x - lambda x ||
      Base<T> norm_2 = El::Nrm2(wi);
      Base<T> norm = norm_2 / (std::max(single.GetNorm(), 1.0));

      largest_norm = std::max(norm, largest_norm);

      // if (H.Grid().Rank() == 0) {
      //   if (i % 10 == 0) std::cout << "\n";
      //   std::cout << norm_2 / (std::max(single.GetNorm(), 1.0)) << " ";
      // }
    }

    config__.setApprox(true);

    {
      //      El::HermitianEigCtrl<T> ctrl;
      // El::HermitianEigSubset<double> subset;
      // subset.indexSubset = true;
      // subset.lowerIndex = 0;
      // subset.upperIndex = nev + nex - 1;

      // El::DistMatrix<double, El::STAR, El::STAR> Lambda{nev + nex, 1, grid};
      El::DistMatrix<double, El::VR, El::STAR> Lambda{nev + nex, nev + nex,
                                                      grid};
      El::DistMatrix<T> VV{N, nex + nev, grid};

      auto t1 = std::chrono::high_resolution_clock::now();
      El::HermitianEig<T>(El::LOWER, H, Lambda, VV, 0, nev + nex - 1,
                          El::ASCENDING);
      auto t2 = std::chrono::high_resolution_clock::now();

      El::Axpy(Base<T>(-1.0), Lambda, ritzv);

      if (H.Grid().Rank() == 0) {
        auto ritzv_nev = El::View(ritzv, 0, 0, nev, 1);
        std::chrono::duration<double> diff = t2 - t1;
        std::cout << "ELEMENTAL: " << diff.count() << " " << largest_norm
                  << " / "
                  << std::real(static_cast<T>(El::Max(ritzv_nev).value))
                  << "\n";
      }
    }
  }
  return 0;
}
