#include <complex>
#include <memory>
#include <random>
#include <vector>

/*include ChASE headers*/
#include "algorithm/performance.hpp"
#include "ChASE-MPI/chase_mpi.hpp"

#include "ChASE-MPI/impl/chase_mpihemm_blas.hpp"
#include "ChASE-MPI/impl/chase_mpihemm_blas_seq.hpp"
#include "ChASE-MPI/impl/chase_mpihemm_blas_seq_inplace.hpp"

using T = std::complex<double>;
using namespace chase;
using namespace chase::mpi;

/*use ChASE-MPI without GPU support*/
typedef ChaseMpi<ChaseMpiHemmBlas, T> CHASE;

int main(int argc, char** argv)
{
  MPI_Init(NULL, NULL);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::size_t N = 1001; //problem size
  std::size_t nev = 100; //number of eigenpairs to be computed
  std::size_t nex = 20; //extra searching space

  std::mt19937 gen(1337.0);
  std::normal_distribution<> d;

  auto V = std::vector<T>(N * (nev + nex)); //eigevectors
  auto Lambda = std::vector<Base<T>>(nev + nex); //eigenvalues

  /*construct eigenproblem to be solved*/
  CHASE single(new ChaseMpiProperties<T>(N, nev, nex, MPI_COMM_SELF), V.data(),
               Lambda.data());

  /*Setup configure for ChASE*/
  auto& config = single.GetConfig();
  /*Tolerance for Eigenpair convergence*/
  config.SetTol(1e-10);
  /*Initial filtering degree*/
  config.SetDeg(20);
  /*Optimi(S)e degree*/
  config.SetOpt(true);


  if (rank == 0)
    std::cout << "Solving a symmetrized Clement matrices (" << N
              << "x" << N << ")" << '\n'
              << config;

  /*randomize V*/
  for (std::size_t i = 0; i < N * (nev + nex); ++i) {
    V[i] = T(d(gen), d(gen));
  }

  std::size_t xoff, yoff, xlen, ylen;

  /*Get Offset and length of block of H on each node*/
  single.GetOff(&xoff, &yoff, &xlen, &ylen);

  std::vector<T> H(N * N, T(0.0));

  /*Generate Clement matrix*/
  for (auto i = 0; i < N; ++i) {
    H[i + N * i] = 0;
    if (i != N - 1) H[i + 1 + N * i] = std::sqrt(i * (N + 1 - i));
    if (i != N - 1) H[i + N * (i + 1)] = std::sqrt(i * (N + 1 - i));
  }

  if (rank == 0) {
      std::cout << "Starting Problem #1" << "\n";
  }

  /*Load different blocks of H to each node*/
  for (std::size_t x = 0; x < xlen; x++) {
    for (std::size_t y = 0; y < ylen; y++) {
      single.GetMatrixPtr()[x + xlen * y] = H.at((xoff + x) * N + (yoff + y));
    }
  }

  /*Performance Decorator to meaure the performance of kernels of ChASE*/
  PerformanceDecoratorChase<T> performanceDecorator(&single);
  /*Solve the eigenproblem*/
  chase::Solve(&performanceDecorator);
  /*Output*/
  if (rank == 0) {
    performanceDecorator.GetPerfData().print();
    Base<T>* resid = single.GetResid();
    std::cout << "Finished Problem #1" << "\n";
    std::cout << "Printing first 5 eigenvalues and residuals\n";
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
    std::cout << "\n\n\n";
    }

    MPI_Finalize();
}


