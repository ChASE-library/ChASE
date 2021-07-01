C++ Code
----------

This example shows the way to construct ChASE with both block
distribution and block-cyclic distribution of the Hermtian matrix.

.. code-block:: c++
    
    #include <complex>
    #include <memory>
    #include <random>
    #include <vector>

    /*include ChASE headers*/
    #include "algorithm/performance.hpp"
    #include "ChASE-MPI/chase_mpi.hpp"
    /*include ChASE-MPI interface for Pure-CPU distributed-memory systems*/
    #include "ChASE-MPI/impl/chase_mpidla_blaslapack.hpp"

    using T = std::complex<double>;
    using namespace chase;
    using namespace chase::mpi;

    /*use ChASE-MPI without GPU support*/
    typedef ChaseMpi<ChaseMpiDLABlaslapack, T> CHASE;
    /*
    // For ChASE-MPI without MPI
    typedef ChaseMpi<ChaseMpiDLABlaslapackSeq, T> CHASE;
    // For ChASE-MPI with MPI + multiGPUs
    typedef ChaseMpi<ChaseMpiDLAMultiGPU, T> CHASE;
    */

    int main(int argc, char** argv)
    {
      MPI_Init(NULL, NULL);
      int rank = 0, size;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);

      std::size_t N = 1001; //problem size
      std::size_t nev = 40; //number of eigenpairs to be computed
      std::size_t nex = 20; //extra searching space

    #ifdef USE_BLOCK_CYCLIC
      /*parameters of block-cyclic data layout*/
      std::size_t NB = 50; //block size for block-cyclic data layout
      int dims[2]; 
      dims[0] = dims[1] = 0;
      //MPI proc grid = dims[0] x dims[1]
      MPI_Dims_create(size, 2, dims);
      int irsrc = 0; 
      int icsrc = 0;
    #endif
  
      std::mt19937 gen(1337.0);
      std::normal_distribution<> d;

      auto V = std::vector<T>(N * (nev + nex)); //eigevectors
      auto Lambda = std::vector<Base<T>>(nev + nex); //eigenvalues

      /*construct eigenproblem to be solved*/
    #ifdef USE_BLOCK_CYCLIC
      /*Use block-cyclic layout to distribute matrix*/
      CHASE single(new ChaseMpiProperties<T>(N, NB, NB, nev, nex, dims[0], dims[1], (char *)"C", irsrc, icsrc, MPI_COMM_WORLD), 
            V.data(), Lambda.data());
    #else
      /*Use block layout to distribute matrix*/
      CHASE single(new ChaseMpiProperties<T>(N, nev, nex, MPI_COMM_WORLD), V.data(),
                 Lambda.data());
    #endif

      /*Setup configuration for ChASE*/
      auto& config = single.GetConfig();
      /*Tolerance for Eigenpair convergence*/
      config.SetTol(1e-10);
      /*Initial filtering degree*/
      config.SetDeg(20);
      /*Optimi(S)e degree*/
      config.SetOpt(true);
      config.SetMaxIter(25);

      if (rank == 0)
        std::cout << "Solving a symmetrized Clement matrices (" << N
                  << "x" << N << ")"
    #ifdef USE_BLOCK_CYCLIC       
              << " with block-cyclic data layout: " << NB << "x" << NB 
    #endif
        << '\n'       
              << config;

      /*randomize V as the initial guess*/
      for (std::size_t i = 0; i < N * (nev + nex); ++i) {
        V[i] = T(d(gen), d(gen));
      }

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

      std::cout << std::setprecision(16);

    #ifdef USE_BLOCK_CYCLIC
      /*local block number = mblocks x nblocks*/
      std::size_t mblocks = single.get_mblocks();
      std::size_t nblocks = single.get_nblocks();

      /*local matrix size = m x n*/
      std::size_t m = single.get_m();
      std::size_t n = single.get_n();

      /*global and local offset/length of each block of block-cyclic data*/
      std::size_t *r_offs, *c_offs, *r_lens, *c_lens, *r_offs_l, *c_offs_l;
      single.get_offs_lens(r_offs, r_lens, r_offs_l, c_offs, c_lens, c_offs_l);

      /*distribute Clement matrix into block cyclic data layout */
      for(std::size_t j = 0; j < nblocks; j++){
          for(std::size_t i = 0; i < mblocks; i++){
              for(std::size_t q = 0; q < c_lens[j]; q++){
                  for(std::size_t p = 0; p < r_lens[i]; p++){
                      single.GetMatrixPtr()[(q + c_offs_l[j]) * m + p + r_offs_l[i]] = H[(q + c_offs[j]) * N + p + r_offs[i]];
                  }
              }
          }
      }

    #else  
      std::size_t xoff, yoff, xlen, ylen;

      /*Get Offset and length of block of H on each node*/
      single.GetOff(&xoff, &yoff, &xlen, &ylen);

      /*Load different blocks of H to each node*/
      for (std::size_t x = 0; x < xlen; x++) {
        for (std::size_t y = 0; y < ylen; y++) {
          single.GetMatrixPtr()[x + xlen * y] = H.at((xoff + x) * N + (yoff + y));
        }
      }
    #endif

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



Execution
----------

The execution of this example through the command line is:

.. code-block:: sh

    mpirun -np ${NPROCS} ./0_hello_world/0_hello_world


For the execution of this example with **Block-Cyclic Distribution**,
it can be done as:

.. code-block:: sh

    mpirun -np ${NPROCS} ./0_hello_world/0_hello_world_block_cyclic


Output
---------

The output of this example gives the timings of important kernels of ChASE and prints the first
5 computed eigenvalues and related residuals.

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
