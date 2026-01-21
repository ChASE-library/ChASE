************************************
How to use ChASE
************************************

Use ChASE as a standalone solver
=====================================

ChASE has multiple versions for both shared-memory and distributed-memory
systems, with or without GPU supports. This section helps users use ChASE to solve
their own problem on preferred architectures from scratch.

In order to use ChASE, the first header file should be included is ``ChASE-MPI/chase_mpi.hpp``, which is a common interface of ChASE solver. This header provides multiple constructors of
class ``ChaseMpi``, targeting different computing architectures.

.. note:: 

  It is named as ``ChASE-MPI``, but for current release
  version of ChASE, it should be included no matter shared-memory or distributed-memory versions of ChASE is selected to be used. 

.. note::
  
  For all versions of ChASE, they share a same interface for the solving step, parameter configuration and performance decoration.

Shared-Memory ChASE
----------------------------------

Include headers
^^^^^^^^^^^^^^^^^^^^^^^
The shared-memory version of ChASE can be built with or without the support of
Nvidia GPU. If GPU support is enabled, only 1 GPU card would be used.

- In order to use shared-memory version of ChASE with only CPU support, it is also necessary to include header ``ChASE-MPI/impl/chase_mpidla_blaslapack_seq.hpp``. This header file provides an implementation of the templated class ``ChaseMpiDLABlaslapackSeq`` which provides the implementations of required dense linear algebra operations. Its template type determines the scalar type that the user would like to work with.

- In order to use shared-memory version of ChASE with GPU support, another header file ``ChASE-MPI/impl/chase_mpidla_cuda_seq.hpp`` should be included. This header file provides an implementation of the templated class ``ChaseMpiDLACudaSeq`` which provides the implementations of required dense linear algebra operations. Its template 
type determines the scalar type that the user would like to work with.

ChASE solver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A ChASE solver, which is in fact an instance of ``ChaseMpi`` should be constructed, before solving the eigenprolem. ``ChaseMpi`` is templated with 2 classes: ``template <typename> class MF`` and ``class T``. The ``T`` is to determine the scalar type that the user
would like to work with. For shared-memory ChASE ``template <typename> class MF`` should be
either ``ChaseMpiDLACudaSeq<T>`` or ``ChaseMpiDLABlaslapackSeq<T>``.


With the combination of the templates ``MF`` and ``T``, different instances of ``ChaseMpi`` 
can be constructed targeting different architectures and scalar types.

For examples, if the user wants to use ChASE to solve an Hermitian matrix with double precision on GPU, an instance of ``ChaseMpi`` should be constructed as follows

.. code-block:: c++

  //N: global size of matrix to be diagonalized
  //nev: number of eigenpairs to be computed
  //nex: external searching space size
  //buffer for storing eigevectors
  auto V = std::vector<std::complex<double>>(N * (nev + nex));
  //buffer for storing computed ritz values
  auto Lambda = std::vector<double>(nev + nex);
  //buffer for storing Hermitian matrix to be diagonalized
  std::vector<std::complex<double>> H(N * N);

  ChaseMpi<ChaseMpiDLACudaSeq, std::complex<double>> solver(N, nev, nex, V.data(), Lambda.data(), H.data());


For the details of APIs, please visit :ref:`ChaseMpi`.


Distributed-Memory ChASE
----------------------------------

Include headers
^^^^^^^^^^^^^^^^^^^^^^^
The distributed-memory version of ChASE can be built with or without the support for
Nvidia GPUs. If GPU support is enabled, it supports only 1 GPU per MPI rank.

- **CPU version**: it is also necessary to include header ``ChASE-MPI/impl/chase_mpidla_blaslapack.hpp``. This header file provides an implementation of a templated class ``ChaseMpiDLABlaslapack`` which provides the implementations of required dense linear algebra operations. Its template type determines the scalar type that the user would like to work with.

- **GPU version**: another header file ``ChASE-MPI/impl/chase_mpidla_mgpu.hpp`` should be included. This header file provides an implementation of a templated class ``ChaseMpiDLAMultiGPU`` which provides the implementations of required dense linear algebra operations. Its template type determines the scalar type that the user would like to work with.


MPI working context
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unlike shared-memory ChASE, for distributed-memory ChASE, it is necessary to initialize a
working MPI communicator for it. A class ``ChaseMpiProperties`` is designed which is able
to construct a 2D MPI grid environment based on user's configuration. Multiple constructors of this class are available:

- a constructor for **Block Distribution** with user-customized 2D MPI grid

- a constructor for **Block Distribution** with 2D MPI grid determined internally by ChASE (as square as possible)

- a constructor for **Block-Cyclic Distribution** with user-customized 2D MPI grid

.. note::
  
  Apart from the setup of 2D MPI grid, this class allocates also the temporary buffers for ChASE and provides some utilities for facilitating the communications.

An example for the constructor for **Block Distribution** with 2D MPI grid determined internally is given as follows

.. code-block:: c++

  auto props = new ChaseMpiProperties<std::complex<double>>(N, nev, nex, MPI_COMM_WORLD);   

in which the input arguments are for: global matrix size, number of eigenpairs to compute,
external searching space size, and working MPI communicator, respectively. A 2D MPI grid will be
internally by ChASE which is as square as possible.

An example for the constructor for **Block Distribution** with user-customized 2D MPI grid is given as follows

.. code-block:: c++
  
  auto props = new ChaseMpiProperties<T>(N, nev, nex, m, n, dims[0], dims[1], (char*)"C", MPI_COMM_WORLD);

 
in which the input arguments are for: global matrix size, number of eigenpairs to compute,
external searching space size, the row number of local block of matrix, the column number of local block of matrix, row number of 2D MPI grid, column number of 2D MPI grid, the grid major of 2D MPI grid ('C' refers to column major), and working MPI communicator, respectively.


An example for the constructor for **Block-Cyclic Distribution** with user-customized 2D MPI grid is given as follows

.. code-block:: c++
  
  auto props = new ChaseMpiProperties<T>(N, NB, NB, nev, nex, dims[0], dims[1], (char*)"C", irscr, icsrc, MPI_COMM_WORLD);

 
in which the input arguments are for: global matrix size, the block factor of block-cyclic distribution for the 1st and 2nd dimension of 2D MPI grid, number of eigenpairs to compute,
external searching space size, the row number of local block of matrix, the column number of local block of matrix, row number of 2D MPI grid, column number of 2D MPI grid, the grid major of 2D MPI grid ('C' refers to column major), process row/column over which the first row/column of the global matrix is distributed, and working MPI communicator, respectively.

ChASE solver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Same as for shared-memory version of ChASE, the class ``ChaseMpi`` provides also constructors for the distributed-memory versions:

- a constructor with pre-allocated buffer ``H`` for Hermitian matrix and its leading dimension ``ldh``

- a constructor without a pre-allocated buffer for Hermitian matrix: in this case, 
  the required buffer would be internally allocated, a ``memcpy`` operation is always
  required to copy from user-provided matrix to the internally allocated buffer. For
  some historic reasons, this version exists, and we are considering to remove it in
  near future.

- Unlike the constructor for shared-memory version of ChASE, the constructors of distributed-memory versions take an instance of ``ChaseMpiProperties`` as an input. This allows creating different ChASE solver with either **Block Distribution** or **Block-Cyclic Distribution** and user customized MPI configuration.  


``ChaseMpi`` is templated with 2 classes: ``template <typename> class MF`` and ``class T``. The ``T`` is to determine the scalar type that the user
would like to work with. For distributed-memory ChASE ``template <typename> class MF`` should be
either ``ChaseMpiDLABlaslapack<T>`` or ``ChaseMpiDLAMultiGPU<T>``.

With the combination of the templates ``MF`` and ``T``, different instances of ``ChaseMpi`` can be constructed targeting different architectures and scalar types.

For examples, if users want to use ChASE to solve an Hermitian matrix with double precision which is to distribute in a *Block-Cyclic* fashion onto multi-GPUs, an instance of ``ChaseMpi`` should be constructed as follows

.. code-block:: c++

  //N: global size of matrix to be diagonalized
  //nev: number of eigenpairs to be computed
  //nex: external searching space size
  //NB: block factor for block-cyclic distribution
  //dims[0] x dims[1]: 2D MPI grid
  //irsrc, icsrc: over which processor row/column the block-cyclic distribution starts from
  //construct MPI context with block-cyclic distribution
  auto props = new ChaseMpiProperties<std::complex<double>>(N, NB, NB, nev, nex, dims[0], dims[1], (char*)"C", irsrc, icsrc, MPI_COMM_WORLD);
  //buffer for storing eigevectors
  auto V = std::vector<std::complex<double>>(props->get_m() * (nev + nex));
  //buffer for storing computed ritz values
  auto Lambda = std::vector<double>(nev + nex);
  auto ldh =  props->get_m();  
  //buffer for storing Hermitian matrix to be diagonalized  
  std::vector<T> H( ldh *  props->get_n());

  ChaseMpi<ChaseMpiDLAMultiGPU, std::complex<double>> solver(props, H, ldh, V.data(), Lambda.data());


Another example with **Block Distribution** and without pre-allocated buffer for Hermitian matrix is as follows:

.. code-block:: c++

  //N: global size of matrix to be diagonalized
  //nev: number of eigenpairs to be computed
  //nex: external searching space size
  //construct MPI context with block-cyclic distribution
  auto props = new ChaseMpiProperties<std::complex<double>>(N, nev, nex, MPI_COMM_WORLD);
  //buffer for storing eigevectors
  auto V = std::vector<std::complex<double>>(props->get_m() * (nev + nex));
  //buffer for storing computed ritz values
  auto Lambda = std::vector<double>(nev + nex);

  ChaseMpi<ChaseMpiDLAMultiGPU, std::complex<double>> solver(props, V.data(), Lambda.data());


For the details of APIs, please visit :ref:`ChaseMpi`.



Parameter Configuration
-----------------------------

Before the starting of solving step, selected parameters are able to
be customized by users. We give an example to show how to configure 
the parameters for a constructed instance of ChASE solver ``solver``.

.. code-block:: c++

  /*Setup configure for ChASE*/
  auto& config = solver.GetConfig();
  /*Tolerance for Eigenpair convergence*/
  config.SetTol(1e-10);
  /*Initial filtering degree*/
  config.SetDeg(20);
  /*Optimi(S)e degree*/
  config.SetOpt(true);
  /*Set max iteration steps*/
  config.SetMaxIter(25);

.. note::

  For all the versions of ChASE targeting different computing architectures,
  they share a uniform interface for the parameter configuration.

For more details about the APIs of parameter configuration, please visit :ref:`
configuration_object`. For the usage and recommendation of values of these 
parameters, please visit :ref:`parameters_and_config`.

Solve
----------
For both shared-memory and distributed versions of ChASE, they share an uniform interface
for the solving step.

Assume that an instance of ``ChaseMpi`` has been constructed with pre-allocated buffers
for Hermitian matrix and desired eigenpairs.

An isolated problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When an isolated problem is to be solved, there would be three steps for solving:

- update the pre-allocated buffer of Hermitian matrix with the matrix to be diagonalized: e.g., through I/O, generation and redistribution

- set the parameter `approx_` to be ``false``: ``config.setApprox(false)``

- solve the problem as: ``chase::Solve(&solver)``.

A sequence of problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a sequence of eigenproblems are to be solved one by one, the steps for solving are:

- for the 1st problem 

  - update the pre-allocated buffer of Hermitian matrix with the matrix to be diagonalized: e.g., through I/O, generation and redistribution

  - set the parameter `approx_` to be ``false``: ``config.setApprox(false)``

  - solve the problem as: ``chase::Solve(&solver)``.

- for the rest of problems (2nd, 3rd...) 

  - update the pre-allocated buffer of Hermitian matrix with the matrix to be diagonalized: e.g., through I/O, generation and redistribution 

  - set the parameter `approx_` to be ``true``: ``config.setApprox(true)``

  - solve the problems as: ``chase::Solve(&solver)``.


.. note::

  - When the parameter `approx_` is set to be ``false``, it means that the initial guess vectors are filled with random numbers respecting to normal distribution. ChASE generate internally these numbers in parallel.

  - The buffer to the initial guess vectors should be allocated externally by users.

  - For distributed-memory ChASE with GPUs, these random numbers are generated in parallel on GPUs.

Performance Decorator
-----------------------------

A templated class ``PerformanceDecoratorChase<T>`` is also provided, which is
able to record the performance of different numerical kernels in ChASE. 
This class is a derived class of the class ``Chase<T>``. It
is quite simple to use it, and we give an example to show how to decorate 
a constructed instance of ChASE solver ``solver`` as follows:

.. code-block:: c++

  PerformanceDecoratorChase<T> performanceDecorator(&solver);

Then the solving step should go with the instance ``performanceDecorator``, rather than
``solver`` itself:

.. code-block:: c++

  chase::Solve(&performanceDecorator);

After the solving step, the recorded performance can be printed out as follows:

.. code-block:: c++

  performanceDecorator.GetPerfData().print();


The output of this performance decorator is as follows :

.. code-block:: bash

  | Size  | Iterations | Vecs   |  All       | Lanczos    | Filter     | QR         | RR         | Resid      |
  |     1 |          5 |   7556 |      1.116 |   0.135028 |    0.87997 |  0.0164864 |  0.0494752 |  0.0310726 |

which represents respectively:

- the number of MPI processors in the working communicator, 

- the iteration number for convergence, 

- total number of matrix-vector product operations, 

- the total time (s), 

- the time cost of Lanczos, Filter, QR, RR and Residuals, respectively. 


Extract the results
----------------------

If an instance of ``ChaseMpi`` is constructed with user provided buffers `V` and `Lambda`,
they will be overwritten by the desired eigenvectors and eigenvalues, respectively.


To be more precise, the first `nev` columns of `V` and the first `nev` elements of `Lambda`
would be the required eigenpairs.

The residuals of all computed eigenpairs can be obtained as follows:

.. code-block:: c++

  Base<T>* resid = single.GetResid(); 

in which ``Base<T>`` represents a basic type of a scalar type, e.g., ``Base<double>`` is ``double`` and ``Base<std::complex<float>>`` is ``float``.


I/O 
----

ChASE itself doesn't provide any parallel I/O functions to load a large 
matrix from a binary file. The reason is that for the majority of applications of ChASE,
the Hermitian matrix is supposed to be already well distributed by applications, it makes no
sense to provide our own version of parallel I/O. This is also the motivation for us
to provide multiple versions of ChASE with the support of both **Block Distribution**
and **Block-Cyclic Distribution**, to adapt all the possible requirements
of applications.

However, for the users who want to test ChASE as a standalone eigensolver, a parallel I/O
might be necessary. Hence, in this section, we provide a hint to load matrix in parallel from local binary file by using built-in functions of ChASE. This parallel I/O is not tuned for
optimal performance, and we encourage the user to develop their own one based on some mature
parallel I/O libraries, such as `HDF5 <https://www.hdfgroup.org/solutions/hdf5>`_ 
and `sionlib <https://apps.fz-juelich.de/jsc/sionlib/docu/index.html>`_.


Block Distribution
""""""""""""""""""""""""""

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
be obtained by the member function ``get_off`` of :ref:`ChaseMpiProperties` class as follows.


.. code:: c++

  std::size_t xoff;
  std::size_t yoff;
  std::size_t xlen;
  std::size_t ylen;

  props.GetOff(&xoff, &yoff, &xlen, &ylen);


Block-Cyclic Distribution
""""""""""""""""""""""""""

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
``get_m``, ``get_n``, and ``get_offs_lens``  of :ref:`ChaseMpiProperties` class as follows.


.. code:: c++

  /*local block number = mblocks x nblocks*/
  std::size_t mblocks = props.get_mblocks();
  std::size_t nblocks = props.get_nblocks();

  /*local matrix size = m x n*/
  std::size_t m = props.get_m();
  std::size_t n = props.get_n();

  /*global and local offset/length of each block of block-cyclic data*/
  std::size_t *r_offs, *c_offs, *r_lens, *c_lens, *r_offs_l, *c_offs_l;

  props.get_offs_lens(r_offs, r_lens, r_offs_l, c_offs, c_lens, c_offs_l);


Use ChASE from external applications
======================================

In order to embed the ChASE library in an application software, ChASE
can be opportunely linked following the instructions in this section.

In this section, we give the guidelines for the integration
of the ChASE library into a given application software. 


.. _link_by_cmake:

Compiling with CMake
-------------------------------

The ``CMakeLists.txt`` (see code window below) is an example on how to link ChASE installation
using CMake. In this example ChASE is linked to a source file named ``chase_app.cpp``.
The ``CMakeLists.txt`` should then be included in the main directory
of the application software as well as the ``chase_app.cpp`` file.

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.8)

   project(chase-app VERSION 0.0.1 LANGUAGES CXX)

   #find installation of ChASE
   find_package( chase REQUIRED CONFIG)

   add_executable(${PROJECT_NAME})

   # add the source file of application
   target_sources(${PROJECT_NAME} PRIVATE chase_app.cpp)

   # link to ChASE
   target_link_libraries(${PROJECT_NAME} PUBLIC ChASE::chase_mpi)

   # if users want to compile the application with multi-GPU version of ChASE
   # the target should be linked to the both the library ChASE::chase_mpi
   # and the library ChASE::chase_cuda
   target_link_libraries(${PROJECT_NAME} PUBLIC ChASE::chase_cuda)

With CMake, the application software can be compiled by the following commands:

.. code-block:: console

   mkdir build & cd build
   cmake .. -DCMAKE_PREFIX_PATH=${ChASEROOT}
   make

`example: 3_installation <https://github.com/ChASE-library/ChASE/tree/master/examples/3_installation>`_
provides an example which illustrates the way to link ChASE by CMake with or without GPU supports.

.. note::
  We highly recommend to link ChASE with CMake. The installation of ChASE allows to use CMake to find and link it easily.


Compiling with Makefile
-------------------------------

Similar as the direct linking, it is also possible to link ChASE by ``Makefile``. 
Here below is a template of this ``Makefile`` for `example: 3_installation <https://github.com/ChASE-library/ChASE/tree/master/examples/3_installation>`_.

.. code-block:: Makefile

  ChASEROOT = /The/installation/path/of/ChASE/on/your/platform

  CXX = mpicxx #or other mpi CXX compiler

  CXXFLAGS = \
      -Wall -fopenmp -MMD \

  INCLUDE_DIR = ${ChASEROOT}/include #include the headers of ChASE

  LIBS_BLASLAPACK = /your/BLAS/LAPACK/SCALAPACK/LIBRARIES

  ## Optional for multi-GPU version of ChASE ##
  LIBS_CUDA = -lcublas -lcusolver -lcudart -lcurand ## link to the libraries of cuBLAS, cuSOLVER and CUDA runtime

  ## Optional for multi-GPU version of ChASE ##
  LIBS_CHASE_CUDA = ${ChASEROOT}/lib64/libchase_cuda.a

  chase-app: LIBS = ${LIBS_BLASLAPACK} #executable generated by chase-app.cpp

  chase-app-gpu: LIBS = ${LIBS_BLASLAPACK} -L${LIBS_CHASE_CUDA} ${LIBS_CUDA} #executable generated by chase-app-gpu.cpp

  src = ${wildcard *.cpp}
  exe = ${basename ${src}}

  all: $(exe)

  .SUFFIXES:

  %: %.cpp
          ${CXX} ${CXXFLAGS} ${LIBS} -I${INCLUDE_DIR} -o $@ $<

  clean:
          -rm -f $(exe) *.o

  -include *.d


.. _c_fortran_interface:

Interface to C/Fortran
======================================

General Description
---------------------

ChASE provides the interfaces to both C and Fortran. 


The usage of both C and Fortran interfaces are
split into 3 steps:

    - **Initialization**: initialization of the context for ChASE, including the setup of the MPI 2D grid, communicators and allocation of buffers, etc.
    - **Solving**: solving the given problem by ChASE within previously setup ChASE context
    - **Finalization**: Cleanup the ChASE context

.. note::
  When a sequence of eigenprblems are to be solved, multiple **solving** steps can be called in sequence after the **Initialization** step. 
  It is the users' responsibility to form a new eigenproblem by updating the buffer allocated for the Hermitian/Symmetric Matrix.

Both C and Fortran interfaces of ChASE provides 3 versions of utilization:

    - **Sequential ChASE**: using the implementation of ChASE for shared-memory architectures. 
    - **Distributed-memory ChASE with Block-Block distribution**: using the implementation of ChASE for distributed-memory architectures, with Block-Block data layout. 
    - **Distributed-memory ChASE with Block-Cyclic distribution**: using the implementation of ChASE for distributed-memory architectures, with Block-Cyclic data layout.  

.. warning::
  When CUDA is detected, these interfaces would automatically use GPU(s).



.. note::
  The naming logic of the interface functions are as follows:

    - For the names of all the functions for distributed memory ChASE, they starts with a prefix ``p``, which follows a same way of naming in ScaLAPACK.
    - For the **Block** and **Block-Cyclic** data layouts:

      - they share a same interface for **Solving** and **Finalization** steps
    
      - but a different interface for the **Initialization** step. For **Block-Cyclic** data layout, the related **Initialization function** ends with a suffix ``blockcyclic``

    - The Fortran interfaces are implemented based on ``iso_c_binding``. It is standard intrinsic module which defines named constants, types, and procedures for the inter-operation with C functions. C and Fortran functions share the same names. Additionally, unlike the Fortran routines, C functions has a suffix `_`.

Different scalar types are also supported by the interfaces of ChASE. We will use abbreviations ``<x>`` for the corresponding 
short type to make a more concise and clear presentation of the implemented functions. 
``Base<x>`` is defined as the table below. Unless otherwise specified ``<x>`` has the following meanings:

.. list-table:: 
   :widths: 4 16 16 16
   :header-rows: 1

   * - ``<x>``
     - Type in C and Fortran 
     - Meaning
     - ``Base<x>`` in C and Fortran
   * - ``s``
     - ``float`` and ``c_float``
     - real single-precision
     - ``float`` and ``c_float``
   * - ``d``
     - ``double`` and ``c_double``
     - real double-precision
     - ``double`` and ``c_double``
   * - ``c``
     - ``float _Complex`` and ``c_float_complex``
     - complex single-precision
     - ``float`` and ``c_float``
   * - ``z``
     - ``double _Complex`` and ``c_double_complex``
     - complex double precision
     - ``double`` and ``c_double``     


Initialization Functions
------------------------

<x>chase_init
^^^^^^^^^^^^^^

``<x>chase_init`` initialize the context for the shared-memory ChASE.
ChASE is initialized with the buffers ``h``, ``v``, ``ritzv``, which should be allocated 
externally by users. These buffers will be re-used when a sequence of eigenproblems are
to be solved.

The APIs for the C interfaces are as follows:

.. code-block:: C

  void schase_init_(int* n, int* nev, int* nex, float* h, float* v, float* ritzv, int* init)
  void dchase_init_(int* n, int* nev, int* nex, double* h, double* v, double* ritzv, int* init)
  void cchase_init_(int* n, int* nev, int* nex, float _Complex* h, float _Complex* v, float* ritzv, int* init)
  void zchase_init_(int* n, int* nev, int* nex, double _Complex* h, double _Complex* v, double* ritzv, int* init)

The APIs for the Fortran interfaces are as follows:

.. code-block:: Fortran

  SUBROUTINE schase_init(n, nev, nex, h, v, ritzv, init)
  SUBROUTINE dchase_init(n, nev, nex, h, v, ritzv, init)
  SUBROUTINE cchase_init(n, nev, nex, h, v, ritzv, init)
  SUBROUTINE zchase_init(n, nev, nex, h, v, ritzv, init)

The interfaces of C and Fortran share the same parameters as follows:

.. list-table:: 
   :widths: 4 8 36
   :header-rows: 1

   * - Param.
     - In/Out 
     - Meaning
   * - ``n``
     - In 
     - global matrix size of the matrix to be diagonalized
   * - ``nev``
     - In 
     - number of desired eigenpairs
   * - ``nex``
     - In 
     - extra searching space size    
   * - ``h``
     - In  
     - pointer to the matrix to be diagonalized, with size of matrix ``nxn``
   * - ``v``
     - In, Out  
     - ``(nx(nev+nex))`` matrix, input is the initial guess eigenvectors,
       and for output, the first ``nev`` columns 
       are overwritten by the desired eigenvectors        
   * - ``ritzv``
     - Out 
     - an array of size nev which contains the desired eigenvalues, it is of type ``Base<x>``
   * - ``init``
     - Out 
     - a flag to indicate if ChASE has been initialized, if initialized, return ``1``

p<x>chase_init
^^^^^^^^^^^^^^

``p<x>chase_init`` initialize the context for the distributed-memory ChASE with **Block Distribution**. ChASE is initialized with the buffers ``h``, ``v``, ``ritzv``, which should be allocated 
externally by users. These buffers will be re-used when a sequence of eigenproblems are
to be solved.

The APIs for the C interfaces are as follows:

.. code-block:: C

  void pschase_init_(int *nn, int *nev, int *nex, int *m, int *n, float *h, int *ldh, 
                     float *v, float *ritzv, int *dim0, int *dim1, char *grid_major,
                     MPI_Comm *comm, int *init) 
  void pdchase_init_(int *nn, int *nev, int *nex, int *m, int *n, double *h, int *ldh, 
                     double *v, double *ritzv, int *dim0, int *dim1, char *grid_major,
                     MPI_Comm *comm, int *init) 
  void pcchase_init_(int *nn, int *nev, int *nex, int *m, int *n, float _Complex *h, int *ldh, 
                     float _Complex *v, float *ritzv, int *dim0, int *dim1, char *grid_major,
                     MPI_Comm *comm, int *init) 
  void pzchase_init_(int *nn, int *nev, int *nex, int *m, int *n, double _Complex *h, int *ldh, 
                     double _Complex *v, double *ritzv, int *dim0, int *dim1, char *grid_major,
                     MPI_Comm *comm, int *init) 


The APIs for the Fortran interfaces are as follows:

.. code-block:: Fortran

  subroutine  pschase_init (nn, nev, nex, m, n, h, ldh, v, ritzv, dim0, dim1, grid_major, fcomm, init)
  subroutine  pdchase_init (nn, nev, nex, m, n, h, ldh, v, ritzv, dim0, dim1, grid_major, fcomm, init)
  subroutine  pcchase_init (nn, nev, nex, m, n, h, ldh, v, ritzv, dim0, dim1, grid_major, fcomm, init)
  subroutine  pzchase_init (nn, nev, nex, m, n, h, ldh, v, ritzv, dim0, dim1, grid_major, fcomm, init)

The interfaces of C and Fortran share the same parameters as follows:


.. list-table:: 
   :widths: 4 8 36
   :header-rows: 1

   * - Param.
     - In/Out 
     - Meaning
   * - ``nn``
     - In 
     - global matrix size of the matrix to be diagonalized
   * - ``nev``
     - In 
     - number of desired eigenpairs
   * - ``nex``
     - In 
     - extra searching space size  
   * - ``m``
     - In 
     - max row number of local matrix h on each MPI process
   * - ``n``
     - In 
     - max column number of local matrix h on each MPI process              
   * - ``h``
     - In  
     - pointer to the matrix to be diagonalized. ``h`` is a block-block distribution of global matrix. 
       ``h`` is of size ``mxn`` with its leading dimension is ``ldh``
   * - ``ldh``
     - In 
     - leading dimension of ``h`` on each MPI process         
   * - ``v``
     - In, Out  
     - ``(mx(nev+nex))`` matrix, input is the initial guess eigenvectors, and for output, the first ``nev`` 
       columns are overwritten by the desired eigenvectors. ``v`` is only partially distributed within column 
       communicator. It is reduandant among different column communicator.
   * - ``ritzv``
     - Out 
     - an array of size ``nev`` which contains the desired eigenvalues, it is of type ``Base<x>``
   * - ``dim0``
     - In 
     - row number of 2D MPI grid    
   * - ``dim1``
     - In 
     - column number of 2D MPI grid 
   * - ``grid_major``
     - In 
     - major of 2D MPI grid. Row major: grid_major='R', column major: grid_major='C'
   * - ``comm`` or ``fcomm``
     - In 
     - the working MPI communicator. ``comm`` is for MPI-C communicator, and ``fcomm`` is for MPI-Fortran communicator.
   * - ``init``
     - Out 
     - a flag to indicate if ChASE has been initialized, if initialized, return ``1``

p<x>chase_init_blockcyclic
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``p<x>chase_init_blockcyclic`` initialize the context for the distributed-memory version of ChASE with **Block-Cyclic Distribution**. ChASE is initialized with the buffers ``h``, ``v``, ``ritzv``, which should be allocated 
externally by users. These buffers will be re-used when a sequence of eigenproblems are
to be solved.

The APIs for the C interfaces are as follows:

.. code-block:: C

  void pschase_init_blockcyclic_(int *nn, int *nev, int *nex, int *mbsize, int *nbsize, 
                                 float *h, int *ldh, float *v, float *ritzv, 
                                 int *dim0, int *dim1, char *grid_major, int *irsrc, 
                                 int *icsrc, MPI_Comm *comm, int *init) 
  void pdchase_init_blockcyclic_(int *nn, int *nev, int *nex, int *mbsize, int *nbsize, 
                                 double *h, int *ldh, double *v, double *ritzv, 
                                 int *dim0, int *dim1, char *grid_major, int *irsrc, 
                                 int *icsrc, MPI_Comm *comm, int *init) 
  void pcchase_init_blockcyclic_(int *nn, int *nev, int *nex, int *mbsize, int *nbsize, 
                                 float _Complex *h, int *ldh, float _Complex *v, float *ritzv, 
                                 int *dim0, int *dim1, char *grid_major, int *irsrc, 
                                 int *icsrc, MPI_Comm *comm, int *init) 
  void pzchase_init_blockcyclic_(int *nn, int *nev, int *nex, int *mbsize, int *nbsize, 
                                 double _Complex *h, int *ldh, double _Complex *v, double *ritzv, 
                                 int *dim0, int *dim1, char *grid_major, int *irsrc, 
                                 int *icsrc, MPI_Comm *comm, int *init) 

The APIs for the Fortran interfaces are as follows:

.. code-block:: Fortran

  subroutine  pschase_init_blockcyclic (nn, nev, nex, mbsize, nbsize, h, ldh, v, ritzv, dim0, dim1, grid_major, irsrc, icsrc, fcomm, init)
  subroutine  pdchase_init_blockcyclic (nn, nev, nex, mbsize, nbsize, h, ldh, v, ritzv, dim0, dim1, grid_major, irsrc, icsrc, fcomm, init)
  subroutine  pcchase_init_blockcyclic (nn, nev, nex, mbsize, nbsize, h, ldh, v, ritzv, dim0, dim1, grid_major, irsrc, icsrc, fcomm, init)
  subroutine  pzchase_init_blockcyclic (nn, nev, nex, mbsize, nbsize, h, ldh, v, ritzv, dim0, dim1, grid_major, irsrc, icsrc, fcomm, init)


The interfaces of C and Fortran share the same parameters as follows:


.. list-table:: 
   :widths: 4 8 36
   :header-rows: 1

   * - Param.
     - In/Out 
     - Meaning
   * - ``nn``
     - In 
     - global matrix size of the matrix to be diagonalized
   * - ``nev``
     - In 
     - number of desired eigenpairs
   * - ``nex``
     - In 
     - extra searching space size  
   * - ``mbsize``
     - In 
     - block size for the block-cyclic distribution for the rows of global matrix
   * - ``nbsize``
     - In 
     - block size for the block-cyclic distribution for the cloumns of global matrix             
   * - ``h``
     - In  
     - pointer to the matrix to be diagonalized. ``h`` is a block-block distribution of global matrix. 
       ``h`` is of size ``mxn`` with its leading dimension is ``ldh``
   * - ``ldh``
     - In 
     - leading dimension of ``h`` on each MPI process         
   * - ``v``
     - In, Out  
     - ``(mx(nev+nex))`` matrix, input is the initial guess eigenvectors, and for output, the first ``nev`` 
       columns are overwritten by the desired eigenvectors. ``v`` is only partially distributed within column 
       communicator. It is redundant among different column communicator.
   * - ``ritzv``
     - Out 
     - an array of size ``nev`` which contains the desired eigenvalues, it is of type ``Base<x>``
   * - ``dim0``
     - In 
     - row number of 2D MPI grid    
   * - ``dim1``
     - In 
     - column number of 2D MPI grid 
   * - ``irsrc``
     - In 
     -  process row over which the first row of the global matrix ``h`` is distributed  
   * - ``icsrc``
     - In 
     -  process column over which the first column of the global matrix ``h`` is distributed.     
   * - ``grid_major``
     - In 
     - major of 2D MPI grid. Row major: grid_major='R', column major: grid_major='C'
   * - ``comm`` or ``fcomm``
     - In 
     - the working MPI communicator. ``comm`` is for MPI-C communicator, and ``fcomm`` is for MPI-Fortran communicator.
   * - ``init``
     - Out 
     - a flag to indicate if ChASE has been initialized, if initialized, return ``1``


Solving Functions
------------------

<x>chase
^^^^^^^^^^

``<x>chase`` solves an eigenvalue problem with given configuration of parameters on shared-memory architectures. When CUDA is enabled, it will automatically use 1 GPU card. 

.. code-block:: C

  void schase_(int *deg, double *tol, char *mode, char *opt) 
  void dchase_(int *deg, double *tol, char *mode, char *opt) 
  void cchase_(int *deg, double *tol, char *mode, char *opt) 
  void zchase_(int *deg, double *tol, char *mode, char *opt) 

.. code-block:: Fortran

  subroutine  schase (deg, tol, mode, opt)
  subroutine  dchase (deg, tol, mode, opt)
  subroutine  cchase (deg, tol, mode, opt)
  subroutine  zchase (deg, tol, mode, opt)


.. list-table:: 
   :widths: 4 8 36
   :header-rows: 1

   * - Param.
     - In/Out 
     - Meaning
   * - ``deg``
     - In 
     - initial degree of Chebyshev polynomial filter
   * - ``tol``
     - In 
     - desired absolute tolerance of computed eigenpairs
   * - ``mode``
     - In 
     - for sequences of eigenproblems, if reusing the eigenpairs obtained from last system. If mode = 'A', reuse, otherwise, no.
   * - ``opt``
     - In 
     - determining if using internal optimization of Chebyshev polynomial degree. If opt='S', use, otherwise, no.

p<x>chase
^^^^^^^^^^

``p<x>chase`` solves an eigenvalue problem with given configuration of parameters on distributed-memory architectures. When CUDA is enabled, it will automatically use multi-GPUs with the configuration 1GPU per MPI rank. 

The APIs for the C interfaces are as follows:

.. code-block:: C

  void pschase_(int *deg, double *tol, char *mode, char *opt) 
  void pdchase_(int *deg, double *tol, char *mode, char *opt) 
  void pcchase_(int *deg, double *tol, char *mode, char *opt) 
  void pzchase_(int *deg, double *tol, char *mode, char *opt) 


The APIs for the Fortran interfaces are as follows:

.. code-block:: Fortran

  subroutine  pschase (deg, tol, mode, opt)
  subroutine  pdchase (deg, tol, mode, opt)
  subroutine  pcchase (deg, tol, mode, opt)
  subroutine  pzchase (deg, tol, mode, opt)

The interfaces of C and Fortran share the same parameters as follows:

.. list-table:: 
   :widths: 4 8 36
   :header-rows: 1

   * - Param.
     - In/Out 
     - Meaning
   * - ``deg``
     - In 
     - initial degree of Cheyshev polynomial filter
   * - ``tol``
     - In 
     - desired absolute tolerance of computed eigenpairs
   * - ``mode``
     - In 
     - for sequences of eigenproblems, if reusing the eigenpairs obtained from last system. If mode = 'A', reuse, otherwise, no.
   * - ``opt``
     - In 
     - determining if using internal optimization of Chebyshev polynomial degree. If opt='S', use, otherwise, no.


Finalization Functions
-----------------------

<x>chase_finalize
^^^^^^^^^^^^^^^^^^

``<x>chase_finalize`` cleans up the instances of shared-memory ChASE.

The APIs for the C interfaces are as follows:

.. code-block:: C

  void schase_finalize_(int *flag) 
  void dchase_finalize_(int *flag) 
  void cchase_finalize_(int *flag) 
  void zchase_finalize_(int *flag) 

The APIs for the Fortran interfaces are as follows:

.. code-block:: Fortran

  subroutine  schase_finalize (flag)
  subroutine  dchase_finalize (flag)
  subroutine  cchase_finalize (flag)
  subroutine  zchase_finalize (flag)

The interfaces of C and Fortran share the same parameters as follows:

.. list-table:: 
   :widths: 4 8 36
   :header-rows: 1

   * - Param.
     - In/Out 
     - Meaning
   * - ``flag``
     - Out 
     - A flag to indicate if ChASE has been cleared up. If ChASE has been cleaned up, ``flag=0``


p<x>chase_finalize
^^^^^^^^^^^^^^^^^^^

``p<x>chase_finalize`` cleans up the instances of distributed-memory ChASE.

.. note::

  For **Block Distribution** and **Block-Cyclic Distribution** versions of ChASE, they share an uniform interface for the finalization.

The APIs for the C interfaces are as follows:

.. code-block:: C

  void pschase_finalize_(int *flag) 
  void pdchase_finalize_(int *flag) 
  void pcchase_finalize_(int *flag) 
  void pzchase_finalize_(int *flag) 

The APIs for the Fortran interfaces are as follows:

.. code-block:: Fortran

  subroutine  pschase_finalize (flag)
  subroutine  pdchase_finalize (flag)
  subroutine  pcchase_finalize (flag)
  subroutine  pzchase_finalize (flag)

The interfaces of C and Fortran share the same parameters as follows:

.. list-table:: 
   :widths: 4 8 36
   :header-rows: 1

   * - Param.
     - In/Out 
     - Meaning
   * - ``flag``
     - Out 
     - A flag to indicate if ChASE has been cleared up. If ChASE has been cleaned up, ``flag=0``


..
 .. note::
  In order to use C interfaces, it is necessary to link to ``libchase_c.a``. In order to use Fortran interfaces, it is required to link to both ``libchase_c.a`` and ``libchase_f.a``


Examples
-----------

A Snippet of examples for both C and Fortran interfaces are shown as follows.
We provide completed examples for both C and Fortran interfaces with both shared-memory
and distributed-memory architectures in `./examples/4_interface <https://github.com/ChASE-library/ChASE/tree/master/examples/4_interface>`_.

Example of C interface
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: C

  ...
  ...

  void pzchase_init_(int* N, int* nev, int* nex, int* m, int* n,
                     double _Complex* H, int* ldh, double _Complex* V,
                     double* ritzv, int* dim0, int* dim1, char* grid_major,
                     MPI_Comm* comm, int* init);
  void pzchase_finalize_(int* flag);
  void pzchase_(int* deg, double* tol, char* mode, char* opt);

  int main(int argc, char** argv)
  {
      MPI_Init(&argc, &argv);
      int rank = 0, init;

      int N = 1001; //global size of matrix
      int nev = 100; //number of eigenparis to compute
      int nex = 40; //size of external searching space
      int m = 501; //number of rows of local matrix on each MPI rank
      int n = 501; //number of columns of local matrix on each MPI rank
      MPI_Comm comm = MPI_COMM_WORLD; //working MPI communicator
      int dims[2];
      dims[0] = 2; //row number of 2D MPI grid
      dims[1] = 2; //column number of 2D MPI grid
      //allocate buffer to store computed eigenvectors

      double _Complex* V = (double _Complex*)malloc(sizeof(double _Complex) * m * (nev + nex));
      //allocate buffer to store computed eigenvalues    
      double* Lambda = (double*)malloc(sizeof(double) * (nev + nex));
      //allocate buffer to store local block of Hermitian matrix on each MPI rank
      double _Complex* H = (double _Complex*)malloc(sizeof(double _Complex) * m * n);

      // config
      int deg = 20;
      double tol = 1e-10;
      char mode = 'R';
      char opt = 'S';

      //Initialize of ChASE
      pzchase_init_(&N, &nev, &nex, &m, &n, H, &m, V, Lambda, &dims[0], &dims[1],
                    (char*)"C", &comm, &init);

      /*
          Generating or loading matrix into H
      */

      //solve 1st eigenproblem with defined configuration of parameters
      pzchase_(&deg, &tol, &mode, &opt);

      /*
          form a new eigenproblem by updating the buffer H
      */

      //Set the mode to 'A', which can recycle previous eigenvectors
      mode = 'A';

      //solve 2nd eigenproblem with updated parameters
      pzchase_(&deg, &tol, &mode, &opt);


      //finalize and clean up
      pzchase_finalize_(&init);

      MPI_Finalize();
  }


Example of Fortran interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: Fortran

  PROGRAM main
  use mpi
  use chase_diag !use chase fortran interface module

  integer ierr, init, comm
  integer m, n
  integer dims(2)
  integer nn, nev, nex
  real(8) :: tol
  integer :: deg
  character        :: mode, opt, major
  complex(8),  allocatable :: h(:,:), v(:,:)
  real(8), allocatable :: lambda(:)

  call mpi_init(ierr)

  nn = 1001 ! global size of matrix
  nev = 100 ! number of eigenparis to compute
  nex = 40 ! size of external searching space

  comm = MPI_COMM_WORLD ! working MPI communicator
  ! config
  deg = 20
  tol = 1e-10
  mode = 'R'
  opt = 'S'
  major = 'C'

  dims(1) = 2 ! row number of 2D MPI grid
  dims(2) = 2 ! column number of 2D MPI grid

  m = 501 ! number of rows of local matrix on each MPI rank
  n = 501 ! number of columns of local matrix on each MPI rank

  allocate(h(m, n)) ! allocate buffer to store local block of Hermitian matrix on each MPI rank
  allocate(v(m, nev + nex)) ! allocate buffer to store computed eigenvectors
  allocate(lambda(nev + nex)) ! allocate buffer to store computed eigenvalues

  ! Initialize of ChASE
  call pzchase_init(nn, nev, nex, m, n, h, m, v, lambda, dims(1), dims(2), major, comm, init)

  !
  !      Generating or loading matrix into H
  !

  ! solve 1st eigenproblem with defined configuration of parameters
  call pzchase(deg, tol, mode, opt)

  !
  !      form a new eigenproblem by updating the buffer H
  !
  ! Set the mode to 'A', which can recycle previous eigenvectors
  mode = 'A'
  
  ! solve 2nd eigenproblem with updated parameters
  call pzchase(deg, tol, mode, opt)

  ! finalize and clean up
  call pzchase_finalize(init)

  call mpi_finalize(ierr)


  END PROGRAM





