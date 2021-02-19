C++ Code
----------

The code block below shows the implementation of parallel I/O
which is able to load a matrix in parallel from local memory. It can be
reused by the users to implement their own applications.

.. code:: c++

  template <typename T>
  void readMatrix(T* H, std::string path_in, std::size_t size,
                  std::size_t xoff, std::size_t yoff,
                  std::size_t xlen, std::size_t ylen) {
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


ChASE parameters can be configured as follows:

.. code:: c++
  
  /* Construction of ChASE object */
  CHASE single(new ChaseMpiProperties<T>(N, nev, nex, MPI_COMM_WORLD), V.data(),
               Lambda.data());

  /* Get the class of configuration from ChASE object */
  auto& config = single.GetConfig();

  /* Parameter configurations */
  config.SetTol(conf.tol);
  config.SetDeg(conf.deg);
  config.SetOpt(true);


.. note::

    For all the provided public funtions of ChASE configuration, please refer to :ref:`configuration_object`.




