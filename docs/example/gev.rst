ScaLAPACK interface
--------------------

In this example, firstly we give an interface for the selected ScaLAPACK routines.
More details of implementation, please visit
`4_gev <https://gitlab.version.fz-juelich.de/SLai/ChASE/-/tree/cmake/examples/3_installation>`_.


.. code:: c++

  /*Returns the number of processes available for use.*/
  void blacs_pinfo(int *mypnum, int *nprocs);
  /*You call the BLACS_GET routine when you want the values the BLACS are using for internal defaults.*/
  void blacs_get(int *icontxt, const int *what, int *val );
  /*Assigns available processes into BLACS process grid.*/
  void blacs_gridinit(int *icontxt, const char layout, const int *nprow, const int *npcol);
  /*Returns information on the current grid.*/
  void blacs_gridinfo(int *icontxt, int *nprow, int *npcol, int *myprow, int *mypcol);
  /*This function computes the local number of rows or columns of a block-cyclically 
    distributed matrix contained in a process row or process column, respectively, 
    indicated by the calling sequence argument iproc.*/
  std::size_t numroc(std::size_t *n, std::size_t *nb, int *iproc, const int *isrcproc, int *nprocs);
  /*This subroutine initializes a type-1 array descriptor with error checking.*/
  void descinit(std::size_t *desc, std::size_t *m, std::size_t *n, std::size_t *mb, std::size_t *nb,
                const int *irsrc, const int *icsrc, int *ictxt, std::size_t *lld, int *info);
  /*Cholesky Factorization*/
  template <typename T>
  void t_ppotrf(const char uplo, const std::size_t n, T *a, const std::size_t ia, 
                const std::size_t ja, std::size_t *desc_a);
  /*Reduce generalized eigenproblem to a standard one*/
  template <typename T>
  void t_psyhegst(const int ibtype, const char uplo,const std::size_t n, T *a, const std::size_t ia,
                  const std::size_t ja, std::size_t *desc_a, const T *b, const std::size_t ib,
                  const std::size_t jb, std::size_t *desc_b, Base<T> *scale);
  /*Solve a triangular linear system*/
  template <typename T>
  void t_ptrtrs(const char uplo, const char trans, const char diag, const std::size_t n,
                const std::size_t nhs, T *a,  const std::size_t ia, const std::size_t ja, 
                std::size_t *desc_a, T *b, const std::size_t ib, const std::size_t jb, 
                std::size_t *desc_b);


Choleksy Factorization + ChASE
-------------------------------

ChASE parameters can be configured as follows:

.. code:: c++

  /*...
    ...
    ...
  */
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

  //read matrix H from local
  readMatrix(H, path_in, "hmat", idx, ".bin", N * N, m, mblocks, nblocks,
         r_offs, r_lens, r_offs_l, c_offs, c_lens, c_offs_l);

  //read matrix S from local
  readMatrix(S, path_in, "smat", idx, ".bin", N * N, m, mblocks, nblocks,
         r_offs, r_lens, r_offs_l, c_offs, c_lens, c_offs_l);

  // Transform to standard problem using SCALAPACK
  // Cholesky Factorization of S = L * L^T, S is overwritten by L
  t_ppotrf<T>('U', N, S, sze_one, sze_one, desc);

  // Reduce H * X = eig * S ( X to the standard from H' * X' = eig * X'
  // with H' = L^{-1} * H * (L^T)^{-1}
  t_psyhegst<T>(i_one, 'U', N, H, sze_one, sze_one, desc, S, sze_one, sze_one, desc, &scale);

  // Copy H into single.matrix()
  std::memcpy(matrix, H, m * n * sizeof(T));

  config.SetApprox(false);
  //random generated initial guess of V
  for (std::size_t i = 0; i < N * (nev + nex); ++i) {
    V[i] = T(d(gen), d(gen));
  }

  PerformanceDecoratorChase<T> performanceDecorator(&single);

  // ChASE to solve the standard eigenproblem H' * X' = eig * X'
  chase::Solve(&performanceDecorator);

  //Scalapack part
  //In ChASE, the eigenvectors V is stored rebundantly on each proc
  //In order to recover the generalized eigenvectors by Scalapack, it should be
  //redistributed into block-cyclic format. We re-use H to restore V.
  //this part is in parallel implicitly
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

  /*...
    ...
    ...
  */


