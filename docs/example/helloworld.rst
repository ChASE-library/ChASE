C++ Code
----------

.. code-block:: c++
    
    #include <complex>
    #include <memory>
    #include <random>
    #include <vector>
    #include <iostream>     // std::cout
    #include <fstream>      // std::ifstream

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

    int main(int argc, char* argv[]) {

      MPI_Init(&argc, &argv);
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      std::size_t N = 1001; //problem size
      std::size_t nev = 100; //number of eigenpairs to be computed
      std::size_t nex = 25; //extra searching space

      auto V = std::vector<T>(N * (nev + nex));  //eigevectors
      auto Lambda = std::vector<Base<T>>(nev + nex); //eigenvalues

      /*construct eigenproblem to be solved*/
      CHASE single(new ChaseMpiProperties<T>(N, nev, nex, MPI_COMM_WORLD), V.data(), Lambda.data());

      /*Setup configure for ChASE*/
      auto& config = single.GetConfig();

      /*Tolerance for Eigenpair convergence*/
      config.SetTol(1.0e-10);
      /*Initial filtering degree*/
      config.SetDeg(20);
      /*Optimi(S)e degree*/
      config.SetOpt(True);

      std::mt19937 gen(1337.0);
      std::normal_distribution<> d;
    
      /*randomize V*/
      for (std::size_t i = 0; i < N * (nev + nex); ++i) {
        V[i] = T(d(gen), d(gen));
      }

      /*Get Offset and length of block of H on each node*/
      std::size_t xoff, yoff, xlen, ylen;
      single.GetOff(&xoff, &yoff, &xlen, &ylen);

      std::vector<T> H(N * N, T(0.0)); /*Hermitian matrix*/

      /*Generate Clement matrix*/
      for (auto i = 0; i < N; ++i) {
        H[i + N * i] = 0;
        if (i != N - 1) H[i + 1 + N * i] = std::sqrt(i * (N + 1 - i));
        if (i != N - 1) H[i + N * (i + 1)] = std::sqrt(i * (N + 1 - i));
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
      if(rank == 0){
        std::cout << " ChASE timings: " << "\n";
        /*Print the timings of different kernels of ChASE*/
        performanceDecorator.GetPerfData().print();
        std::cout << "Finished Problem \n";
        /*Print first five eigenvalues and the related residuals*/
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

      return 0;

    }


Output
---------

.. code-block:: bash

    ChASE timings:
    | Size  | Iterations | Vecs   |  All       | Lanczos    | Filter     | QR         | RR         | Resid      |
    |     1 |          9 |  18700 |   0.726643 |   0.215907 |   0.313806 |  0.0701111 |  0.0685877 |  0.0267064 |
   Finished Problem
   Printing first 5 eigenvalues and residuals
   | Index |       Eigenvalue      |         Residual      |
   |-------|-----------------------|-----------------------|
   |     1 |  -1.001999052554e+03  |   5.031097871974e-05  |
   |     2 |  -9.999980537536e+02  |   6.316669642834e-02  |
   |     3 |  -9.979970534604e+02  |   8.933118471008e-02  |
   |     4 |  -9.959960516707e+02  |   1.094079046013e-01  |
   |     5 |  -9.939950483800e+02  |   1.263333629940e-01  |
