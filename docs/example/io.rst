ChASE Object and Configuration 
------------------------------

ChASE parameters can be configured as follows:

.. code:: c++
  
  typedef ChaseMpi<ChaseMpiDLABlaslapack, T> CHASE;
  /* Construction of ChASE object */
  CHASE single(new ChaseMpiProperties<T>(N, nev, nex, MPI_COMM_WORLD), V.data(),
               Lambda.data());

  /* Get the class of configuration from ChASE object */
  auto& config = single.GetConfig();

  /* Parameter configurations */
  config.SetTol(conf.tol);
  config.SetDeg(conf.deg);
  config.SetOpt(true);
  ...

.. note::

    For all the provided public funtions of ChASE configuration, please refer to :ref:`configuration_object`.


Parallel IO 
------------

The code block below shows the implementation of parallel I/O
which is able to load a matrix in parallel from local memory. It can be
reused by the users to implement their own applications.

Block Distribution
^^^^^^^^^^^^^^^^^^^

This is an example to load a matrix from local into block distribution data layout.

.. code:: c++

  template <typename T>
  void readMatrix(T* H, /*The pointer to store the local part of matrix on each MPI rank*/
                  std::string path_in, /*The path to load binary file of matrix*/
                  std::size_t size, /*size = N * N, in which N is the size of matrix to be loaded*/
                  std::size_t xoff, 
                  std::size_t yoff, 
                  std::size_t xlen, 
                  std::size_t ylen)
  {
    std::size_t N = std::sqrt(size);
    std::ostringstream problem(std::ostringstream::ate);
    problem << path_in;

    std::cout << problem.str() << std::endl;
    std::ifstream input(problem.str().c_str(), std::ios::binary);
    if (!input.is_open()) {
      throw new std::logic_error(std::string("error reading file: ") +
                                 problem.str());
    }

    for (std::size_t y = 0; y < ylen; y++) {
      input.seekg(((xoff) + N * (yoff + y)) * sizeof(T));
      input.read(reinterpret_cast<char*>(H + xlen * y), xlen * sizeof(T));
    }
  }

For the parameters **xoff**, **yoff**, **xlen** and **ylen**, they can 
be obtained by the member function ``GetOff`` of :ref:`ChaseMpi` class as follows.


.. code:: c++

  std::size_t xoff;
  std::size_t yoff;
  std::size_t xlen;
  std::size_t ylen;

  single.GetOff(&xoff, &yoff, &xlen, &ylen);


Block-Cyclic Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is an example to load a matrix from local into block-cyclic distribution data layout.


.. code:: c++

  template <typename T>
  void readMatrix(T* H, /*The pointer to store the local part of matrix on each MPI rank*/
                  std::string path_in, /*The path to load binary file of matrix*/
                  std::size_t size, /*size = N * N, in which N is the size of matrix to be loaded*/
                  std::size_t m, 
                  std::size_t mblocks, 
                  std::size_t nblocks,
                  std::size_t* r_offs, 
                  std::size_t* r_lens, 
                  std::size_t* r_offs_l,
                  std::size_t* c_offs, 
                  std::size_t* c_lens, 
                  std::size_t* c_offs_l){

    std::size_t N = std::sqrt(size);
    std::ostringstream problem(std::ostringstream::ate);
    problem << path_in;

    std::cout << problem.str() << std::endl;

    std::ifstream input(problem.str().c_str(), std::ios::binary);
    if (!input.is_open()) {
      throw new std::logic_error(std::string("error reading file: ") +
                                 problem.str());
    }

    for(std::size_t j = 0; j < nblocks; j++){
      for(std::size_t i = 0; i < mblocks; i++){
        for(std::size_t q = 0; q < c_lens[j]; q++){
            input.seekg(((q + c_offs[j]) * N + r_offs[i])* sizeof(T));
            input.read(reinterpret_cast<char*>(H + (q + c_offs_l[j]) * m + r_offs_l[i]), r_lens[i] * sizeof(T));
        }
      }
    }
  }


For the parameters **m**, **mblocks**, **nblocks**, **r_offs**, **r_lens**, **r_offs_l**, 
**c_offs**, **c_lens** and **c_offs_l**, 
they can be obtained by the member functions ``get_mblocks``, ``get_nblocks``, 
``get_m``, ``get_n``, and ``get_offs_lens``  of :ref:`ChaseMpi` class as follows.


.. code:: c++

  /*local block number = mblocks x nblocks*/
  std::size_t mblocks = single.get_mblocks();
  std::size_t nblocks = single.get_nblocks();

  /*local matrix size = m x n*/
  std::size_t m = single.get_m();
  std::size_t n = single.get_n();

  /*global and local offset/length of each block of block-cyclic data*/
  std::size_t *r_offs, *c_offs, *r_lens, *c_lens, *r_offs_l, *c_offs_l;

  single.get_offs_lens(r_offs, r_lens, r_offs_l, c_offs, c_lens, c_offs_l);



Execution
----------

This examples provides multiple implementation of ChASE targeting different computing architectures with different data distribution scheme:

1. ``2_input_output_seq``: sequential ChASE without MPI support, and **Block Distribution**.

2. ``2_input_output``: ChASE for distributed memory systems with MPI support (pure CPUs),  and **Block Distribution**.

3. ``2_input_output_block_cyclic``: ChASE for distributed memory systems with MPI support (pure CPUs),  and **Block-Cyclic Distribution**.

4. ``2_input_output_mgpu``: ChASE for distributed memory systems with MPI support (with GPUs),  and **Block Distribution**.

5. ``2_input_output_mgpu_block_cyclic``: ChASE for distributed memory systems with MPI support (with GPUs),  and **Block-Cyclic Distribution**.


This example uses `Boost` for parsing the parameters, thus the required parameters and configuration can be gotten by the `help` flag::

  ./2_input_output/2_input_output -h


Solving single problem
^^^^^^^^^^^^^^^^^^^^^^^^

Here we utilize ``2_input_output`` as an example to illustrate the way to use ChASE to solve single eigenproblem with loading external matrix.

The execution of this example through the command line is::

  mpirun -np ${NPROCS} ./2_input_output/2_input_output --path_in=${MATRIX_BINARY_FILE} --n=${RANK_OF_MATRIX} --nev=${NB_of_WANTED_EIGENPAIRS} --nex=${EXTERNAL_SEARCHNING_SPACE} --mode=R


Solving a sequence of problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we also utilize ``2_input_output`` as an example to illustrate the way to use ChASE to solve a sequence of eigenproblems with loading external matrix.

The execution of this example through the command line is::

  mpirun -np ${NPROCS} ./2_input_output/2_input_output --path_in=${DIRECTORY_STORE_MATRICES} --n=${RANK_OF_MATRIX} --nev=${NB_of_WANTED_EIGENPAIRS} --nex=${EXTERNAL_SEARCHNING_SPACE} --legacy=true --mode=R --bgn=2 --end=10 --sequence=true

In this example, for each physical system, a number (``N``) of matrices are available, which should be solved in sequence. 
All the matrices are named ``gmat/ /1/ ell``, with ``ell`` varying from ``1`` to ``N``. The flag ``--legacy=true`` enables
searching the matrices conforming this naming policies in the directory ``${DIRECTORY_STORE_MATRICES}``.

In the execution example above, the first matrix to be solved is ``gmat/ /1/ 2``, the the last matrix  to be solved is ``gmat/ /1/ 10``.

Parameters of configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

========================= ===================================================================================================
Parameter (default value) Description
========================= ===================================================================================================
  -h [ --help ]           show this message
  --n arg                 Size of the Input Matrix
  --double arg (=1)       Is matrix complex double valued, false indicates the
                          single type
  --complex arg (=1)      Matrix is complex valued
  --nev arg               Wanted Number of Eigenpairs
  --nex arg (=25)         Extra Search Dimensions
  --deg arg (=20)         Initial filtering degree
  --bgn arg (=2)          Start ell
  --end arg (=2)          End ell
  --tol arg (=1e-10)      Tolerance for Eigenpair convergence
  --path_in arg           Path to the input matrix/matrices
  --mode arg (=A)         valid values are R(andom) or A(pproximate)
  --opt arg (=S)          Optimi(S)e degree, or do (N)ot optimise
  --path_eigp arg         Path to approximate solutions, only required when
                          mode is Approximate, otherwise not used
  --sequence arg (=0)     Treat as sequence of Problems. Previous ChASE solution
                          is used,when available
  --mbsize arg (=400)     block size for the row, it only matters for **Block-Cyclic Distribution**.
  --nbsize arg (=400)     block size for the column, it only matters for **Block-Cyclic Distribution**.
  --dim0 arg (=0)         row number of MPI proc grid, it only matters for **Block-Cyclic Distribution**.
  --dim1 arg (=0)         column number of MPI proc grid, it only matters for **Block-Cyclic Distribution**.
  --irsrc arg (=0)        The process row over which the first row of matrix
                          is distributed. It only matters for **Block-Cyclic Distribution**.
  --icsrc arg (=0)        The process column over which the first column of the
                          array A isdistributed. It only matters for **Block-Cyclic Distribution**.
  --major arg (=C)        Major of MPI proc grid, valid values are R(ow) or
                          C(olumn). It only matters for **Block-Cyclic Distribution**.
  --legacy arg (=0)       Use legacy naming scheme?
========================= ===================================================================================================


.. note:: 
  We have generated a few number of matrices defining (sequences of) eigenproblems from multiple material science simulation codes, if you want to
  test with these matrices, please feel free to contact us.


.. note::
  For the fine tuning of more parameters in ChASE, please visit :ref:`configuration_object`, in which we provide a class
  to set up all the parameters of eigensolvers. For the suggestion of selecting values of parameters, please visit :ref:`parameters_and_config`.