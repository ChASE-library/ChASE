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





