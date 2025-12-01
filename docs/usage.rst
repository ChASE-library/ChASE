************************************
How to use ChASE
************************************

Use ChASE as a standalone solver
=====================================

ChASE provides multiple implementation variants for both sequential (shared-memory)
and parallel (distributed-memory) systems, with or without GPU support. This section
helps users use ChASE to solve their own eigenvalue problems on preferred architectures.

The ChASE library is organized into the following main components:

* **Implementation Classes**: ``chase::Impl::ChASECPU``, ``chase::Impl::ChASEGPU``,
  ``chase::Impl::pChASECPU``, ``chase::Impl::pChASEGPU``
* **Matrix Types**: Sequential matrices (``chase::matrix::Matrix<T>``,
  ``chase::matrix::QuasiHermitianMatrix<T>``) and distributed matrices
  (``chase::distMatrix::BlockBlockMatrix<T, Platform>``,
  ``chase::distMatrix::BlockCyclicMatrix<T, Platform>``, etc.)
* **Configuration**: ``chase::ChaseConfig<T>`` for parameter setup
* **Solve Function**: ``chase::Solve()`` for executing the eigensolver

All implementations share a uniform interface for solving, parameter configuration,
and performance decoration.

Sequential ChASE (Shared-Memory)
----------------------------------

Sequential implementations of ChASE are designed for single-node execution and can
be built with or without GPU support.

Include Headers
^^^^^^^^^^^^^^^^^^^^^^^

For **CPU-only** sequential ChASE, include:

.. code-block:: c++

    #include "Impl/chase_cpu/chase_cpu.hpp"

For **GPU** sequential ChASE, include:

.. code-block:: c++

    #include "Impl/chase_gpu/chase_gpu.hpp"

Creating the Solver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The sequential ChASE solvers can be constructed in two ways:

**Constructor 1: Using raw pointers**

.. code-block:: c++

    // N: global size of matrix to be diagonalized
    // nev: number of eigenpairs to be computed
    // nex: external searching space size
    // H: pointer to the matrix buffer (size N x N)
    // ldh: leading dimension of H
    // V: pointer to eigenvector buffer (size N x (nev + nex))
    // ldv: leading dimension of V
    // Lambda: pointer to eigenvalue buffer (size nev + nex)
    
    auto solver = chase::Impl::ChASECPU(N, nev, nex, H, ldh, V, ldv, Lambda.data());

For GPU version:

.. code-block:: c++

    auto solver = chase::Impl::ChASEGPU(N, nev, nex, H, ldh, V, ldv, Lambda.data());

**Constructor 2: Using matrix objects**

This constructor allows you to specify the matrix type explicitly, which is useful
for pseudo-Hermitian problems:

.. code-block:: c++

    using T = std::complex<double>;
    
    // For Hermitian problems
    auto Hmat = new chase::matrix::Matrix<T>(N, N);
    auto solver = chase::Impl::ChASECPU<T, chase::matrix::Matrix<T>>(
        N, nev, nex, Hmat, V.data(), N, Lambda.data());
    
    // For Pseudo-Hermitian problems (e.g., BSE)
    auto Hmat = new chase::matrix::QuasiHermitianMatrix<T>(N, N);
    auto solver = chase::Impl::ChASECPU<T, chase::matrix::QuasiHermitianMatrix<T>>(
        N, nev, nex, Hmat, V.data(), N, Lambda.data());

Parallel ChASE (Distributed-Memory)
----------------------------------

Parallel implementations of ChASE use MPI for distributed-memory execution and can
be built with or without GPU support. They support multiple matrix distribution
schemes (Block, Block-Cyclic, Redundant).

Include Headers
^^^^^^^^^^^^^^^^^^^^^^^

For **CPU-only** parallel ChASE, include:

.. code-block:: c++

    #include "Impl/pchase_cpu/pchase_cpu.hpp"
    #include "grid/mpiGrid2D.hpp"
    #include "linalg/distMatrix/distMatrix.hpp"
    #include "linalg/distMatrix/distMultiVector.hpp"

For **GPU** parallel ChASE, include:

.. code-block:: c++

    #include "Impl/pchase_gpu/pchase_gpu.hpp"
    #include "grid/mpiGrid2D.hpp"
    #include "linalg/distMatrix/distMatrix.hpp"
    #include "linalg/distMatrix/distMultiVector.hpp"

Setting up MPI Grid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before creating the solver, you need to set up a 2D MPI grid:

.. code-block:: c++

    #include <mpi.h>
    
    MPI_Init(&argc, &argv);
    
    int dims_[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dims_);
    
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
        = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
            dims_[0], dims_[1], MPI_COMM_WORLD);

Creating Distributed Matrices and Vectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Block Distribution**

.. code-block:: c++

    using T = std::complex<double>;
    using ARCH = chase::platform::CPU;  // or chase::platform::GPU
    
    auto Hmat = chase::distMatrix::BlockBlockMatrix<T, ARCH>(N, N, mpi_grid);
    auto Vec = chase::distMultiVector::DistMultiVector1D<T, 
        chase::distMultiVector::CommunicatorType::column, ARCH>(
        N, nev + nex, mpi_grid);

**Block-Cyclic Distribution**

.. code-block:: c++

    std::size_t blocksize = 64;
    auto Hmat = chase::distMatrix::BlockCyclicMatrix<T, ARCH>(
        N, N, blocksize, blocksize, mpi_grid);
    auto Vec = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T,
        chase::distMultiVector::CommunicatorType::column, ARCH>(
        N, nev + nex, blocksize, mpi_grid);

**Pseudo-Hermitian Matrices**

For pseudo-Hermitian problems (e.g., BSE), use the QuasiHermitian variants:

.. code-block:: c++

    // Block distribution
    auto Hmat = chase::distMatrix::QuasiHermitianBlockBlockMatrix<T, ARCH>(
        N, N, mpi_grid);
    
    // Block-Cyclic distribution
    auto Hmat = chase::distMatrix::QuasiHermitianBlockCyclicMatrix<T, ARCH>(
        N, N, blocksize, blocksize, mpi_grid);

Creating the Solver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**CPU Version (pChASECPU)**

.. code-block:: c++

    auto Lambda = std::vector<chase::Base<T>>(nev + nex);
    auto solver = chase::Impl::pChASECPU(nev, nex, &Hmat, &Vec, Lambda.data());

**GPU Version (pChASEGPU)**

For GPU version, you can optionally specify the communication backend:

.. code-block:: c++

    // Using MPI backend (default)
    auto solver = chase::Impl::pChASEGPU(nev, nex, &Hmat, &Vec, Lambda.data());
    
    // Using NCCL backend for optimized GPU communication
    using BackendType = chase::grid::backend::NCCL;
    auto solver = chase::Impl::pChASEGPU<decltype(Hmat), decltype(Vec), BackendType>(
        nev, nex, &Hmat, &Vec, Lambda.data());

**Complete Example: Parallel ChASE with Block-Cyclic Distribution**

.. code-block:: c++

    #include "Impl/pchase_cpu/pchase_cpu.hpp"
    #include "grid/mpiGrid2D.hpp"
    #include "linalg/distMatrix/distMatrix.hpp"
    #include "linalg/distMatrix/distMultiVector.hpp"
    #include <mpi.h>
    
    using T = std::complex<double>;
    using namespace chase;
    
    int main(int argc, char** argv)
    {
        MPI_Init(&argc, &argv);
        
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        
        std::size_t N = 1200;
        std::size_t nev = 80;
        std::size_t nex = 60;
        
        // Setup MPI grid
        int dims_[2] = {0, 0};
        MPI_Dims_create(world_size, 2, dims_);
        std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
                dims_[0], dims_[1], MPI_COMM_WORLD);
        
        // Create distributed matrices and vectors
        std::size_t blocksize = 64;
        auto Hmat = chase::distMatrix::BlockCyclicMatrix<T, chase::platform::CPU>(
            N, N, blocksize, blocksize, mpi_grid);
        auto Vec = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T,
            chase::distMultiVector::CommunicatorType::column, chase::platform::CPU>(
            N, nev + nex, blocksize, mpi_grid);
        
        // Create solver
        auto Lambda = std::vector<chase::Base<T>>(nev + nex);
        auto solver = chase::Impl::pChASECPU(nev, nex, &Hmat, &Vec, Lambda.data());
        
        // Configure and solve (see next sections)
        
        MPI_Finalize();
        return 0;
    }

Parameter Configuration
-----------------------------

Before solving, you can configure ChASE parameters through the ``GetConfig()`` method,
which returns a reference to a ``chase::ChaseConfig<T>`` object.

.. code-block:: c++

    auto& config = solver.GetConfig();
    
    // Tolerance for eigenpair convergence
    config.SetTol(1e-10);
    
    // Initial filtering degree
    config.SetDeg(20);
    
    // Enable/disable degree optimization
    config.SetOpt(true);
    
    // Maximum number of iterations
    config.SetMaxIter(25);
    
    // For sequences: use approximate solution (reuse previous eigenvectors)
    config.SetApprox(false);  // false for first problem, true for subsequent problems
    
    // Additional parameters (optional)
    config.SetMaxDeg(36);              // Maximum degree of Chebyshev filter
    config.SetLanczosIter(26);         // Number of Lanczos iterations
    config.SetNumLanczos(4);            // Number of stochastic vectors for spectral estimates

All implementations (ChASECPU, ChASEGPU, pChASECPU, pChASEGPU) share the same
configuration interface.

For more details about the configuration API, please visit :ref:`configuration_object`.
For recommendations on parameter values, please visit :ref:`parameters_and_config`.

Solve
----------

All ChASE implementations share a uniform interface for solving eigenvalue problems.

An Isolated Problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For solving a single isolated problem:

.. code-block:: c++

    // 1. Update the matrix buffer with your matrix data
    //    (e.g., through I/O, generation, or redistribution)
    
    // 2. Set approx to false for random initial guess
    config.SetApprox(false);
    
    // 3. Solve the problem
    chase::Solve(&solver);

A Sequence of Problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When solving a sequence of related eigenproblems:

.. code-block:: c++

    for (int i = 0; i < num_problems; ++i)
    {
        // 1. Update the matrix buffer with the new matrix
        //    (e.g., through I/O, generation, or redistribution)
        
        // 2. For the first problem, use random initial guess
        //    For subsequent problems, reuse previous eigenvectors
        if (i == 0)
        {
            config.SetApprox(false);  // Random initial guess
        }
        else
        {
            config.SetApprox(true);   // Reuse previous eigenvectors
        }
        
        // 3. Solve the problem
        chase::Solve(&solver);
    }

.. note::

  * When ``SetApprox(false)``, ChASE generates random initial guess vectors
    internally in parallel. The buffer for initial guess vectors should be
    allocated externally by users.
  
  * For distributed-memory ChASE with GPUs, random numbers are generated
    in parallel on GPUs.

Performance Decorator
-----------------------------

A performance decorator class is provided to record the performance of different
numerical kernels in ChASE:

.. code-block:: c++

    #include "algorithm/performance.hpp"
    
    // Decorate the solver
    PerformanceDecoratorChase<T> performanceDecorator(&solver);
    
    // Solve using the decorator instead of the solver directly
    chase::Solve(&performanceDecorator);
    
    // After solving, print performance data
    performanceDecorator.GetPerfData().print();

The output format is:

.. code-block:: bash

  | Size  | Iterations | Vecs   |  All       | Lanczos    | Filter     | QR         | RR         | Resid      |
  |     1 |          5 |   7556 |      1.116 |   0.135028 |    0.87997 |  0.0164864 |  0.0494752 |  0.0310726 |

The columns represent:
* **Size**: Number of MPI processes in the working communicator
* **Iterations**: Number of iterations for convergence
* **Vecs**: Total number of matrix-vector product operations
* **All**: Total time (seconds)
* **Lanczos**: Time for Lanczos algorithm
* **Filter**: Time for Chebyshev filtering
* **QR**: Time for QR factorization
* **RR**: Time for Rayleigh-Ritz procedure
* **Resid**: Time for residual computation

Extract the Results
----------------------

After solving, the results are stored in the buffers provided during construction.

**Eigenvalues**: The first ``nev`` elements of the ``Lambda`` array contain the
computed eigenvalues.

**Eigenvectors**: The first ``nev`` columns of the ``V`` matrix (or ``Vec`` multi-vector)
contain the computed eigenvectors.

**Residuals**: You can obtain the residuals of all computed eigenpairs:

.. code-block:: c++

    chase::Base<T>* resid = solver.GetResid();

where ``chase::Base<T>`` represents the base type of the scalar type ``T``:
* ``chase::Base<double>`` is ``double``
* ``chase::Base<std::complex<float>>`` is ``float``
* ``chase::Base<std::complex<double>>`` is ``double``

**Example: Printing Results**

.. code-block:: c++

    chase::Base<T>* resid = solver.GetResid();
    std::cout << "Eigenvalues and Residuals:\n";
    std::cout << "| Index |       Eigenvalue      |         Residual      |\n";
    std::cout << "|-------|-----------------------|-----------------------|\n";
    
    for (std::size_t i = 0; i < nev; ++i)
    {
        std::cout << "|  " << std::setw(4) << i + 1 << " | "
                  << std::setw(20) << Lambda[i] << "  | "
                  << std::setw(20) << resid[i] << "  |\n";
    }

I/O for Distributed Matrices
----

ChASE itself doesn't provide parallel I/O functions to load large matrices from
binary files. For most applications, the matrix is already well-distributed by the
application, making ChASE's own I/O unnecessary. This is why ChASE supports multiple
distribution schemes (Block, Block-Cyclic) to adapt to different application requirements.

However, for users who want to test ChASE as a standalone eigensolver, you may need
to implement your own parallel I/O. We recommend using mature parallel I/O libraries
such as `HDF5 <https://www.hdfgroup.org/solutions/hdf5>`_ and
`sionlib <https://apps.fz-juelich.de/jsc/sionlib/docu/index.html>`_.

For distributed matrices, you can access local data using:

.. code-block:: c++

    // For CPU matrices
    T* local_data = Hmat.l_data();
    std::size_t local_rows = Hmat.l_rows();
    std::size_t local_cols = Hmat.l_cols();
    
    // For GPU matrices
    Hmat.allocate_cpu_data();  // Allocate CPU buffer if needed
    T* cpu_data = Hmat.cpu_data();
    T* gpu_data = Hmat.gpu_data();

Matrix Redistribution
""""""""""""""""""""""""""

ChASE provides a ``redistributeImpl()`` method to redistribute matrices between
different distribution schemes:

.. code-block:: c++

    // Create a redundant matrix (full copy on each rank)
    auto Redundant = chase::distMatrix::RedundantMatrix<T, ARCH>(N, N, mpi_grid);
    
    // Fill the redundant matrix with data
    // ...
    
    // Redistribute to block-cyclic distribution
    auto Hmat = chase::distMatrix::BlockCyclicMatrix<T, ARCH>(
        N, N, blocksize, blocksize, mpi_grid);
    Redundant.redistributeImpl(&Hmat);

Use ChASE from external applications
======================================

In order to embed the ChASE library in an application software, ChASE
can be linked following the instructions in this section.

.. _link_by_cmake:

Compiling with CMake
-------------------------------

The following ``CMakeLists.txt`` is an example on how to link ChASE installation
using CMake. In this example ChASE is linked to a source file named ``chase_app.cpp``.

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.8)

   project(chase-app VERSION 0.0.1 LANGUAGES CXX)

   # Find installation of ChASE
   find_package(ChASE REQUIRED CONFIG)

   add_executable(${PROJECT_NAME} chase_app.cpp)

   # Link to ChASE
   # For sequential CPU version
   target_link_libraries(${PROJECT_NAME} PUBLIC ChASE::chase_cpu)
   
   # For sequential GPU version (if available)
   # target_link_libraries(${PROJECT_NAME} PUBLIC ChASE::chase_gpu)
   
   # For parallel CPU version
   # target_link_libraries(${PROJECT_NAME} PUBLIC ChASE::pchase_cpu)
   
   # For parallel GPU version (if available)
   # target_link_libraries(${PROJECT_NAME} PUBLIC ChASE::pchase_gpu)

With CMake, the application software can be compiled by the following commands:

.. code-block:: console

   mkdir build && cd build
   cmake .. -DCMAKE_PREFIX_PATH=${ChASEROOT}
   make

The `example: 3_installation <https://github.com/ChASE-library/ChASE/tree/master/examples/3_installation>`_
provides an example which illustrates the way to link ChASE by CMake with or without GPU supports.

.. note::
  We highly recommend linking ChASE with CMake. The installation of ChASE allows
  CMake to find and link it easily using the ``find_package(ChASE)`` command.

Compiling with Makefile
-------------------------------

Similar to CMake, it is also possible to link ChASE using a ``Makefile``.
Here is a template ``Makefile``:

.. code-block:: Makefile

  ChASEROOT = /The/installation/path/of/ChASE/on/your/platform

  CXX = mpicxx  # or other MPI C++ compiler

  CXXFLAGS = -Wall -fopenmp -MMD -std=c++17

  INCLUDE_DIR = ${ChASEROOT}/include  # include the headers of ChASE

  LIBS_BLASLAPACK = /your/BLAS/LAPACK/SCALAPACK/LIBRARIES

  ## Optional for GPU version of ChASE ##
  LIBS_CUDA = -lcublas -lcusolver -lcudart -lcurand

  ## Libraries to link ##
  LIBS_CHASE_CPU = ${ChASEROOT}/lib64/libchase_cpu.a
  LIBS_CHASE_GPU = ${ChASEROOT}/lib64/libchase_gpu.a
  LIBS_PCHASE_CPU = ${ChASEROOT}/lib64/libpchase_cpu.a
  LIBS_PCHASE_GPU = ${ChASEROOT}/lib64/libpchase_gpu.a

  chase-app: LIBS = -L${LIBS_CHASE_CPU} -lchase_cpu ${LIBS_BLASLAPACK}
  
  chase-app-gpu: LIBS = -L${LIBS_CHASE_GPU} -lchase_gpu ${LIBS_CUDA} ${LIBS_BLASLAPACK}
  
  chase-app-parallel: LIBS = -L${LIBS_PCHASE_CPU} -lpchase_cpu ${LIBS_BLASLAPACK}
  
  chase-app-parallel-gpu: LIBS = -L${LIBS_PCHASE_GPU} -lpchase_gpu ${LIBS_CUDA} ${LIBS_BLASLAPACK}

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

ChASE provides interfaces to both C and Fortran for users who prefer not to use
the C++ API directly. The usage of both C and Fortran interfaces is split into
3 steps:

    - **Initialization**: Initialize the context for ChASE, including the setup
      of the MPI 2D grid, communicators, and allocation of buffers.
    - **Solving**: Solve the given problem by ChASE within the previously setup
      ChASE context.
    - **Finalization**: Cleanup the ChASE context.

.. note::
  When a sequence of eigenproblems are to be solved, multiple **solving** steps
  can be called in sequence after the **Initialization** step. It is the users'
  responsibility to form a new eigenproblem by updating the buffer allocated for
  the Hermitian/Symmetric Matrix.

Both C and Fortran interfaces of ChASE provide 3 versions of utilization:

    - **Sequential ChASE**: Using the implementation of ChASE for shared-memory
      architectures.
    - **Distributed-memory ChASE with Block distribution**: Using the implementation
      of ChASE for distributed-memory architectures, with Block data layout.
    - **Distributed-memory ChASE with Block-Cyclic distribution**: Using the
      implementation of ChASE for distributed-memory architectures, with
      Block-Cyclic data layout.

.. warning::
  When CUDA is detected, these interfaces automatically use GPU(s).

.. note::
  The naming logic of the interface functions:

    - For the names of all functions for distributed memory ChASE, they start with
      a prefix ``p``, following the same naming convention as ScaLAPACK.
    - For **Block** and **Block-Cyclic** data layouts:
      - They share the same interface for **Solving** and **Finalization** steps
      - But use different interfaces for the **Initialization** step. For
        **Block-Cyclic** data layout, the related **Initialization function** ends
        with the suffix ``blockcyclic``
    - The Fortran interfaces are implemented based on ``iso_c_binding``, a standard
      intrinsic module which defines named constants, types, and procedures for
      inter-operation with C functions. C and Fortran functions share the same names.
      Additionally, unlike the Fortran routines, C functions have a suffix ``_``.

Different scalar types are also supported by the interfaces of ChASE. We use
abbreviations ``<x>`` for the corresponding short type to make a more concise and
clear presentation of the implemented functions. ``Base<x>`` is defined in the
table below. Unless otherwise specified, ``<x>`` has the following meanings:

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

``<x>chase_init`` initializes the context for the shared-memory ChASE.
ChASE is initialized with the buffers ``h``, ``v``, ``ritzv``, which should be
allocated externally by users. These buffers will be re-used when a sequence of
eigenproblems are to be solved.

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

``p<x>chase_init`` initializes the context for the distributed-memory ChASE with
**Block Distribution**. ChASE is initialized with the buffers ``h``, ``v``, ``ritzv``,
which should be allocated externally by users. These buffers will be re-used when
a sequence of eigenproblems are to be solved.

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
     - pointer to the matrix to be diagonalized. ``h`` is a block-block distribution
       of global matrix. ``h`` is of size ``mxn`` with its leading dimension is ``ldh``
   * - ``ldh``
     - In 
     - leading dimension of ``h`` on each MPI process         
   * - ``v``
     - In, Out  
     - ``(mx(nev+nex))`` matrix, input is the initial guess eigenvectors, and for
       output, the first ``nev`` columns are overwritten by the desired eigenvectors.
       ``v`` is only partially distributed within column communicator. It is redundant
       among different column communicators.
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
     - the working MPI communicator. ``comm`` is for MPI-C communicator, and ``fcomm``
       is for MPI-Fortran communicator.
   * - ``init``
     - Out 
     - a flag to indicate if ChASE has been initialized, if initialized, return ``1``

p<x>chase_init_blockcyclic
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``p<x>chase_init_blockcyclic`` initializes the context for the distributed-memory
version of ChASE with **Block-Cyclic Distribution**. ChASE is initialized with the
buffers ``h``, ``v``, ``ritzv``, which should be allocated externally by users.
These buffers will be re-used when a sequence of eigenproblems are to be solved.

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
     - block size for the block-cyclic distribution for the columns of global matrix             
   * - ``h``
     - In  
     - pointer to the matrix to be diagonalized. ``h`` is a block-cyclic distribution
       of global matrix. ``h`` is of size ``mxn`` with its leading dimension is ``ldh``
   * - ``ldh``
     - In 
     - leading dimension of ``h`` on each MPI process         
   * - ``v``
     - In, Out  
     - ``(mx(nev+nex))`` matrix, input is the initial guess eigenvectors, and for
       output, the first ``nev`` columns are overwritten by the desired eigenvectors.
       ``v`` is only partially distributed within column communicator. It is redundant
       among different column communicators.
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
     - process row over which the first row of the global matrix ``h`` is distributed  
   * - ``icsrc``
     - In 
     - process column over which the first column of the global matrix ``h`` is distributed.     
   * - ``grid_major``
     - In 
     - major of 2D MPI grid. Row major: grid_major='R', column major: grid_major='C'
   * - ``comm`` or ``fcomm``
     - In 
     - the working MPI communicator. ``comm`` is for MPI-C communicator, and ``fcomm``
       is for MPI-Fortran communicator.
   * - ``init``
     - Out 
     - a flag to indicate if ChASE has been initialized, if initialized, return ``1``

Solving Functions
------------------

<x>chase
^^^^^^^^^^

``<x>chase`` solves an eigenvalue problem with given configuration of parameters
on shared-memory architectures. When CUDA is enabled, it will automatically use
1 GPU card.

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
     - for sequences of eigenproblems, if reusing the eigenpairs obtained from
       last system. If mode = 'A', reuse, otherwise, no.
   * - ``opt``
     - In 
     - determining if using internal optimization of Chebyshev polynomial degree.
       If opt='S', use, otherwise, no.

p<x>chase
^^^^^^^^^^

``p<x>chase`` solves an eigenvalue problem with given configuration of parameters
on distributed-memory architectures. When CUDA is enabled, it will automatically use
multi-GPUs with the configuration 1 GPU per MPI rank.

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

The interfaces of C and Fortran share the same parameters as described for ``<x>chase`` above.

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

.. list-table:: 
   :widths: 4 8 36
   :header-rows: 1

   * - Param.
     - In/Out 
     - Meaning
   * - ``flag``
     - Out 
     - A flag to indicate if ChASE has been cleared up. If ChASE has been cleaned
       up, ``flag=0``

p<x>chase_finalize
^^^^^^^^^^^^^^^^^^^

``p<x>chase_finalize`` cleans up the instances of distributed-memory ChASE.

.. note::

  For **Block Distribution** and **Block-Cyclic Distribution** versions of ChASE,
  they share a uniform interface for the finalization.

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

The interfaces of C and Fortran share the same parameters as described for ``<x>chase_finalize`` above.

.. note::
  In order to use C interfaces, it is necessary to link to ``libchase_c.a``.
  In order to use Fortran interfaces, it is required to link to both ``libchase_c.a``
  and ``libchase_f.a``.

Examples
-----------

Complete examples for both C and Fortran interfaces with both shared-memory
and distributed-memory architectures are provided in
`./examples/4_interface <https://github.com/ChASE-library/ChASE/tree/master/examples/4_interface>`_.

Example of C interface
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: C

  #include <mpi.h>
  #include "chase_c.h"  // Include ChASE C interface header
  
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
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      int N = 1001; // global size of matrix
      int nev = 100; // number of eigenpairs to compute
      int nex = 40; // size of external searching space
      int m = 501; // number of rows of local matrix on each MPI rank
      int n = 501; // number of columns of local matrix on each MPI rank
      MPI_Comm comm = MPI_COMM_WORLD; // working MPI communicator
      int dims[2];
      dims[0] = 2; // row number of 2D MPI grid
      dims[1] = 2; // column number of 2D MPI grid
      
      // allocate buffer to store computed eigenvectors
      double _Complex* V = (double _Complex*)malloc(sizeof(double _Complex) * m * (nev + nex));
      // allocate buffer to store computed eigenvalues    
      double* Lambda = (double*)malloc(sizeof(double) * (nev + nex));
      // allocate buffer to store local block of Hermitian matrix on each MPI rank
      double _Complex* H = (double _Complex*)malloc(sizeof(double _Complex) * m * n);

      // config
      int deg = 20;
      double tol = 1e-10;
      char mode = 'R';
      char opt = 'S';

      // Initialize ChASE
      pzchase_init_(&N, &nev, &nex, &m, &n, H, &m, V, Lambda, &dims[0], &dims[1],
                    (char*)"C", &comm, &init);

      /*
          Generating or loading matrix into H
      */

      // solve 1st eigenproblem with defined configuration of parameters
      pzchase_(&deg, &tol, &mode, &opt);

      /*
          form a new eigenproblem by updating the buffer H
      */

      // Set the mode to 'A', which can recycle previous eigenvectors
      mode = 'A';

      // solve 2nd eigenproblem with updated parameters
      pzchase_(&deg, &tol, &mode, &opt);

      // finalize and clean up
      pzchase_finalize_(&init);

      MPI_Finalize();
      return 0;
  }

Example of Fortran interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: Fortran

  PROGRAM main
  use mpi
  use chase_diag ! use chase fortran interface module

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
  nev = 100 ! number of eigenpairs to compute
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

  ! Initialize ChASE
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
