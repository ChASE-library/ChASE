C++ Code
---------

.. code-block:: cpp

  /*...
    The initialization part is quite similar as the previous example.
    ...
  */

  /*Generate Clement matrix for the 1st eigenproblem*/
  for (auto i = 0; i < N; ++i) {
    H[i + N * i] = 0;
    if (i != N - 1) H[i + 1 + N * i] = std::sqrt(i * (N + 1 - i));
    if (i != N - 1) H[i + N * (i + 1)] = std::sqrt(i * (N + 1 - i));
  }

  std::size_t idx_max = 3; /*The number of problem to be solved in sequence*/
  /*For the 1st problem, we don't have approximated eigenvalues to be reused*/
  config.SetApprox(false);

  for (auto idx = 0; idx < idx_max; ++idx) {
    if (rank == 0) {
      std::cout << "Starting Problem #" << idx << "\n";
      if (config.UseApprox()) {
        std::cout << "Using approximate solution\n";
      }
    }

 #ifdef USE_BLOCK_CYCLIC
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
      std::cout << "Finished Problem #" << idx << "\n";
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

      config.SetApprox(true);

      /*Generate next Clement matrix by the perturbation of elements for the next problem*/
      for (std::size_t i = 1; i < N; ++i) {
        for (std::size_t j = 1; j < i; ++j) {
          T element_perturbation = T(d(gen), d(gen)) * perturb;
          H[j + N * i] += element_perturbation;
          H[i + N * j] += std::conj(element_perturbation);
        }
      }

      MPI_Barrier(MPI_COMM_WORLD);
   }

  /*...
    ...
    ...
  */


Output
-------

.. code-block:: bash

    Starting Problem #0
     | Size  | Iterations | Vecs   |  All       | Lanczos    | Filter     | QR         | RR         | Resid      |
     |     1 |          5 |  10974 |    0.31693 |  0.0665169 |   0.155455 |  0.0416761 |  0.0369509 |  0.0150942 |
    Finished Problem #0
    Printing first 5 eigenvalues and residuals
    | Index |       Eigenvalue      |         Residual      |
    |-------|-----------------------|-----------------------|
    |     1 |  -1.001000000000e+03  |   3.103509700217e-11  |
    |     2 |  -9.990000000000e+02  |   4.384023033610e-11  |
    |     3 |  -9.970000000000e+02  |   4.223319943235e-11  |
    |     4 |  -9.950000000000e+02  |   5.236648653823e-11  |
    |     5 |  -9.930000000000e+02  |   4.694707763186e-11  |



    Starting Problem #1
    Using approximate solution
    | Size  | Iterations | Vecs   |  All       | Lanczos    | Filter     | QR         | RR         | Resid      |
    |     1 |          3 |   5716 | 1.486899e-01 | 6.437470e-03 | 9.449926e-02 | 2.086782e-02 | 1.743995e-02 | 8.172003e-03 |
    Finished Problem #1
    Printing first 5 eigenvalues and residuals
    | Index |       Eigenvalue      |         Residual      |
    |-------|-----------------------|-----------------------|
    |     1 |  -1.001000026082e+03  |   1.213937398026e-11  |
    |     2 |  -9.989999626846e+02  |   1.290405645729e-11  |
    |     3 |  -9.970000007482e+02  |   1.392145504287e-11  |
    |     4 |  -9.949999584251e+02  |   1.541421315367e-11  |
    |     5 |  -9.930002191627e+02  |   1.685610611985e-11  |



    Starting Problem #2
    Using approximate solution
    | Size  | Iterations | Vecs   |  All       | Lanczos    | Filter     | QR         | RR         | Resid      |
    |     1 |          3 |   5716 | 1.827815e-01 | 7.925692e-03 | 9.755376e-02 | 3.922733e-02 | 2.912766e-02 | 7.646011e-03 |
    Finished Problem #2
    Printing first 5 eigenvalues and residuals
    | Index |       Eigenvalue      |         Residual      |
    |-------|-----------------------|-----------------------|
    |     1 |  -1.000999977886e+03  |   1.216241128129e-11  |
    |     2 |  -9.989998757359e+02  |   1.249352158615e-11  |
    |     3 |  -9.970000504574e+02  |   1.434239777145e-11  |
    |     4 |  -9.949999357134e+02  |   1.533918527688e-11  |
    |     5 |  -9.930001634377e+02  |   1.682910938179e-11  |
