************************************
Modules
************************************

Overview
=============

The implementation of ChASE provides a stand-alone high-performance 
parallel library based on our original design of the Chebyshev
accelerated subspace iteration algorithm. The ChASE library promises the portability 
to heterogeneous architectures and the easy integration into existing codes. This goal is
achieved by separating the implementation of the ChASE algorithm from the required numerical
kernels via an interface based on pure C++ abstract classes. Classes derived from this interface 
handle data distribution and (parallel) execution of each kernel. The required numerical kernels 
are based on Basic Linear Algebra Subprograms (BLAS)-3 compatible kernels, such as a (parallel) 
matrix-matrix multiplication and QR factorization. This modern "stand-alone" strategy grants 
ChASE an unprecedented degree of flexibility that makes the integration of this library in 
most application codes quite simple. ChASE efficiently uses available machine resources.

All the implementation of ChASE take place within a C++ namespace ``chase``. The library is 
organized into several key modules:

* **Core Algorithm Module**: Abstract interfaces and algorithm implementation
* **Implementation Classes**: Concrete implementations for different architectures
* **Matrix Classes**: Sequential and distributed matrix types
* **Grid and Communication**: MPI grid management and communication backends
* **Linear Algebra Kernels**: Low-level numerical kernels for different platforms
* **Interface Modules**: C and Fortran bindings

Core Algorithm Module
======================

chase::ChaseBase
----------------

The numerical kernels required by ChASE algorithm are defined in the abstract base class 
``chase::ChaseBase<T>``. All the functions are defined as virtual functions, and further 
implementations are required targeting different computing architectures. It includes the
following functionalities:

  * ``HEMM()``: Hermitian Matrix-Matrix Multiplication

  * ``QR()``: QR factorization (with S-orthonormalization for pseudo-Hermitian matrices)

  * ``RR()``: Rayleigh-Ritz projection and small problem solver
    - For Hermitian problems: Standard Rayleigh-Ritz projection
    - For Pseudo-Hermitian problems: Oblique Rayleigh-Ritz projection with S-metric

  * ``Resd()``: Compute the eigenpair residuals

  * ``Lanczos()``: Estimate the bounds of user-interested spectrum by Lanczos eigensolver

  * ``LanczosDos()``: Estimate the spectral distribution of eigenvalues

  * ``Swap()``: Swap the two matrices of vectors used in the Chebyshev filter

  * ``Lock()``: Lock the converged eigenpairs

  * ``Shift()``: Shift the diagonal of matrix ``A`` used in the 3-term
    recurrence relation implemented in the Chebyshev filter

  * ``isSym()``: Check if the matrix is symmetric/Hermitian

  * ``isPseudoHerm()``: Check if the matrix is pseudo-Hermitian

  * ``GetConfig()``: Get the configuration object

  * ``GetResid()``: Get the residual vector

**API Reference**: :ref:`api_chasebase`


chase::Algorithm
------------------

The class ``chase::Algorithm<T>`` has awareness of the class ``chase::ChaseBase<T>``, and
it defines the algorithmic implementation of ChASE using the
defined virtual kernels in ``chase::ChaseBase<T>``. It includes the
functionalities:

  * Chebyshev filter

  * Calculation of degree of the filter

  * Lanczos solver to estimate the bound of spectra

  * Locking the converged Ritz pairs

  * Convergence checking

The function ``chase::Solve(ChaseBase<T>*)`` provides the main entry point for the ChASE 
algorithm by assembling the algorithms and numerical kernels implemented in ``chase::ChaseBase<T>`` 
and ``chase::Algorithm<T>``.

.. note::
   
   This class implements the ChASE algorithm using the virtual functions. It cannot run in practice
   until concrete implementations of these virtual functions are provided through the 
   implementation classes (see :ref:`implementation_classes`).

**API Reference**: :ref:`api_algorithm`


chase::ChaseConfig
--------------------

The class ``chase::Algorithm<T>`` is aware of the class ``chase::ChaseConfig<T>``, which 
defines the functions to set different parameters of ChASE.

Besides setting up the standard parameters such as size of the
matrix defining the eigenproblem, number of wanted
eigenvalues, the public functions of this class
initialize all internal parameters and allow the experienced
user to set up the values of parameters of core functionalities
(e.g. lanczos DoS). The aim is to influence the behavior of the
library in special cases when the default values of the
parameters return a sub-optimal efficiency in terms of
performance and/or accuracy.


.. note::

  For more details of all available functions, please refer to
  :ref:`configuration_object`.

**API Reference**: :ref:`api_chaseconfig`


chase::ChasePerfData
---------------------------

This class defines the performance data for different algorithm and numerical
kernels of ChASE, e.g., the floating operations of ChASE for given size of matrix
and a required number of eigenpairs to be computed.

The ``chase::ChasePerfData`` class collects and handles information relative to the
execution of the eigensolver. It collects information about

   - Number of subspace iterations

   - Number of filtered vectors

   - Timings of each main algorithmic procedure (Lanczos, Filter, etc.)

   - Number of FLOPs executed

The number of iterations and filtered vectors can be used to
monitor the behavior of the algorithm as it attempts to converge
all the desired eigenpairs. The timings and number of FLOPs are
used to measure performance, especially parallel performance. The
timings are stored in a vector of objects derived by the class
template `std::chrono::duration`.


.. note::

  For more details of all available functions, including usage examples,
  please refer to the :doc:`usage` documentation, specifically the 
  "Performance Decorator" section.

**API Reference**: :ref:`api_performance`


chase::PerformanceDecoratorChase
-------------------------------------

This is a class derived from the ``chase::ChaseBase<T>`` which plays the
role of decorator for performance measurement. All
members of the ``chase::ChaseBase<T>`` class are virtual functions. These
functions are re-implemented in the ``chase::PerformanceDecoratorChase<T>``
class. All derived members that provide an interface to
computational kernels are re-implemented by *decorating* the
original function with time pointers which are members of the
``chase::ChasePerfData`` class. All derived members that provide an
interface to input or output data are called without any
specific decoration. In addition to the virtual members of the
``chase::ChaseBase<T>`` class, the ``chase::PerformanceDecoratorChase<T>`` class has also among
its public members a reference to an object of type
``chase::ChasePerfData``. When using ChASE to solve an eigenvalue problem,
the members of the PerformanceDecoratorChase are called instead
of the virtual function members of the ``chase::ChaseBase<T>`` class. In this
way, all parameters and counters are automatically invoked and
returned in the correct order.

.. note::

  For more details of all available functions, including usage examples,
  please refer to the :doc:`usage` documentation, specifically the 
  "Performance Decorator" section.

**API Reference**: :ref:`api_performance`


.. _implementation_classes:

Implementation Classes
========================

The ChASE library provides four main implementation classes that derive from 
``chase::ChaseBase<T>`` and provide concrete implementations of all virtual 
numerical kernels. These classes are located in the ``chase::Impl`` namespace.

Sequential Implementations
---------------------------

chase::Impl::ChASECPU
^^^^^^^^^^^^^^^^^^^^^^

The class ``chase::Impl::ChASECPU<T, MatrixType>`` provides a sequential CPU 
implementation of ChASE. It supports:

  * **Matrix Types**: 
    - ``chase::matrix::Matrix<T>`` for Hermitian (symmetric) eigenvalue problems
    - ``chase::matrix::PseudoHermitianMatrix<T>`` for pseudo-Hermitian eigenvalue problems

  * **Backend**: BLAS and LAPACK libraries for numerical computations

  * **Use Case**: Single-node, CPU-only eigenvalue problems

  * **Platform**: CPU only

This implementation is suitable for problems that fit in the memory of a single 
node and do not require parallel computation.

**API Reference**: :ref:`api_implementations`


chase::Impl::ChASEGPU
^^^^^^^^^^^^^^^^^^^^^^

The class ``chase::Impl::ChASEGPU<T, MatrixType>`` provides a sequential GPU 
implementation of ChASE. It supports:

  * **Matrix Types**: 
    - ``chase::matrix::Matrix<T, chase::platform::GPU>`` for Hermitian problems
    - ``chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU>`` for pseudo-Hermitian problems

  * **Backend**: cuBLAS and cuSOLVER libraries for GPU computations

  * **Use Case**: Single-node, GPU-accelerated eigenvalue problems

  * **Platform**: GPU only

This implementation is suitable for problems that fit in GPU memory and can 
benefit from GPU acceleration on a single node.

**API Reference**: :ref:`api_implementations`


Parallel Implementations
--------------------------

chase::Impl::pChASECPU
^^^^^^^^^^^^^^^^^^^^^^^

The class ``chase::Impl::pChASECPU<MatrixType, InputMultiVectorType, BackendType>`` 
provides a parallel CPU implementation of ChASE using MPI. It supports:

  * **Matrix Types**: Distributed matrix classes
    - ``chase::distMatrix::BlockBlockMatrix<T, chase::platform::CPU>``
    - ``chase::distMatrix::BlockCyclicMatrix<T, chase::platform::CPU>``
    - ``chase::distMatrix::RedundantMatrix<T, chase::platform::CPU>``
    - ``chase::distMatrix::PseudoHermitianBlockBlockMatrix<T, chase::platform::CPU>``
    - ``chase::distMatrix::PseudoHermitianBlockCyclicMatrix<T, chase::platform::CPU>``

  * **Backend**: 
    - ``chase::grid::backend::MPI`` for communication
    - ScaLAPACK for distributed linear algebra operations

  * **Use Case**: Multi-node, CPU-only eigenvalue problems

  * **Platform**: CPU with MPI

This implementation is suitable for large-scale problems that require distributed 
memory computation across multiple nodes.

**API Reference**: :ref:`api_implementations`


chase::Impl::pChASEGPU
^^^^^^^^^^^^^^^^^^^^^^^

The class ``chase::Impl::pChASEGPU<MatrixType, InputMultiVectorType, BackendType>`` 
provides a parallel GPU implementation of ChASE. It supports:

  * **Matrix Types**: Distributed GPU matrix classes
    - ``chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>``
    - ``chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>``
    - ``chase::distMatrix::PseudoHermitianBlockBlockMatrix<T, chase::platform::GPU>``
    - ``chase::distMatrix::PseudoHermitianBlockCyclicMatrix<T, chase::platform::GPU>``

  * **Backends**: 
    - ``chase::grid::backend::MPI`` for CPU-based MPI communication
    - ``chase::grid::backend::NCCL`` for GPU-to-GPU communication via NCCL

  * **Use Case**: Multi-node, multi-GPU eigenvalue problems

  * **Platform**: GPU with MPI/NCCL

This implementation is suitable for large-scale problems that require distributed 
memory computation across multiple nodes with GPU acceleration. The NCCL backend 
provides optimized GPU-to-GPU communication for better performance.

**API Reference**: :ref:`api_implementations`


Matrix Classes
===============

The ChASE library provides matrix classes for both sequential and distributed 
computations, supporting both Hermitian and pseudo-Hermitian eigenvalue problems.

Sequential Matrix Classes
--------------------------

chase::matrix::Matrix
^^^^^^^^^^^^^^^^^^^^^^

The class ``chase::matrix::Matrix<T, Platform, Allocator>`` is the base matrix 
class for Hermitian (symmetric) eigenvalue problems. It provides:

  * **Template Parameters**:
    - ``T``: Scalar type (``float``, ``double``, ``std::complex<float>``, ``std::complex<double>``)
    - ``Platform``: ``chase::platform::CPU`` or ``chase::platform::GPU``
    - ``Allocator``: Memory allocator (optional)

  * **Storage**: Column-major storage compatible with BLAS/LAPACK

  * **Use Case**: Standard eigenvalue problems of the form :math:`A \hat{x} = \lambda \hat{x}` 
    where :math:`A = A^\dagger` (or :math:`A = A^T` for real matrices)

**API Reference**: :ref:`api_matrices`


chase::matrix::PseudoHermitianMatrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The class ``chase::matrix::PseudoHermitianMatrix<T, Platform, Allocator>`` is 
derived from ``chase::matrix::Matrix<T, Platform, Allocator>`` and is designed 
for pseudo-Hermitian eigenvalue problems, such as those arising from the 
Bethe-Salpeter Equation (BSE). It provides:

  * **Template Parameters**: Same as ``chase::matrix::Matrix``

  * **Storage**: Same column-major storage, but with additional support for 
    dual basis vectors required for pseudo-Hermitian problems

  * **Use Case**: Pseudo-Hermitian eigenvalue problems where the matrix satisfies 
    :math:`SH = H^*S` with a signature matrix :math:`S`

**API Reference**: :ref:`api_matrices`


Type Tags
^^^^^^^^^^

The library also provides type tags for matrix classification:

  * ``chase::matrix::Hermitian``: Type tag for Hermitian matrices
  * ``chase::matrix::PseudoHermitian``: Type tag for pseudo-Hermitian matrices


Distributed Matrix Classes
---------------------------

The distributed matrix classes are located in the ``chase::distMatrix`` namespace 
and support various distribution schemes for parallel computation.

Hermitian Distributed Matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

chase::distMatrix::BlockBlockMatrix
""""""""""""""""""""""""""""""""""""

The class ``chase::distMatrix::BlockBlockMatrix<T, Platform>`` provides 
block-wise distribution of matrices across MPI processes. This distribution 
scheme is most efficient for matrix-matrix operations and is the default 
choice for many applications.

  * **Distribution**: Block-wise (rectangular blocks)
  * **Use Case**: General-purpose distributed computation
  * **Performance**: Optimal for matrix-matrix multiplications

**API Reference**: :ref:`api_matrices`


chase::distMatrix::BlockCyclicMatrix
""""""""""""""""""""""""""""""""""""

The class ``chase::distMatrix::BlockCyclicMatrix<T, Platform>`` provides 
block-cyclic distribution of matrices across MPI processes. This distribution 
scheme provides better load balance for some operations.

  * **Distribution**: Block-cyclic (round-robin block assignment)
  * **Use Case**: Applications requiring better load balance
  * **Performance**: Better for operations with irregular access patterns

**API Reference**: :ref:`api_matrices`


chase::distMatrix::RedundantMatrix
"""""""""""""""""""""""""""""""""""

The class ``chase::distMatrix::RedundantMatrix<T, Platform>`` stores a full 
copy of the matrix on each MPI rank. This is useful for small matrices or 
when redistribution is needed.

  * **Distribution**: Full copy on each rank
  * **Use Case**: Small matrices, redistribution operations
  * **Memory**: Higher memory requirement (full matrix per rank)

**API Reference**: :ref:`api_matrices`


Pseudo-Hermitian Distributed Matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

chase::distMatrix::PseudoHermitianBlockBlockMatrix
"""""""""""""""""""""""""""""""""""""""""""""""""

The class ``chase::distMatrix::PseudoHermitianBlockBlockMatrix<T, Platform>`` 
provides block-wise distribution for pseudo-Hermitian matrices.

  * **Distribution**: Block-wise (same as BlockBlockMatrix)
  * **Use Case**: Distributed pseudo-Hermitian problems with block distribution

**API Reference**: :ref:`api_matrices`


chase::distMatrix::PseudoHermitianBlockCyclicMatrix
""""""""""""""""""""""""""""""""""""""""""""""""""

The class ``chase::distMatrix::PseudoHermitianBlockCyclicMatrix<T, Platform>`` 
provides block-cyclic distribution for pseudo-Hermitian matrices.

  * **Distribution**: Block-cyclic (same as BlockCyclicMatrix)
  * **Use Case**: Distributed pseudo-Hermitian problems with block-cyclic distribution

**API Reference**: :ref:`api_matrices`


Distributed Multi-Vectors
^^^^^^^^^^^^^^^^^^^^^^^^^^

The library also provides distributed multi-vector classes for managing 
eigenvectors and workspace vectors in distributed memory:

  * ``chase::distMultiVector::DistMultiVector1D<T, CommunicatorType, Platform>``: 
    1D distributed multi-vector
    
  * ``chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, CommunicatorType, Platform>``: 
    1D block-cyclic distributed multi-vector

  * ``chase::distMultiVector::AbstractDistMultiVector<T, CommunicatorType, Derived, Platform>``: 
    Abstract base class for distributed multi-vectors

The ``CommunicatorType`` can be ``row``, ``column``, or ``all``, determining 
which MPI communicator is used for the distribution.

**API Reference**: :ref:`api_matrices`


Grid and Communication
=======================

The ``chase::grid`` namespace provides classes and utilities for managing MPI 
process grids and communication backends.

chase::grid::MpiGrid2D
-----------------------

The class ``chase::grid::MpiGrid2D<GridMajor>`` manages a 2D MPI process grid 
for distributed computation. It provides:

  * **Template Parameter**: ``GridMajor`` - Either ``chase::grid::GridMajor::RowMajor`` 
    or ``chase::grid::GridMajor::ColMajor``

  * **Functionality**:
    - Grid dimension and coordinate management
    - MPI communicator creation (row, column, and full grid communicators)
    - ScaLAPACK context integration (if ScaLAPACK is available)
    - NCCL communicator support (if NCCL is available)

  * **Use Case**: Required for all parallel implementations (pChASECPU, pChASEGPU)

The grid is typically created with dimensions that factor the total number of MPI 
processes, e.g., for 16 processes, a 4x4 or 2x8 grid can be used.

**API Reference**: :ref:`api_grid`


chase::grid::MpiGrid2DBase
----------------------------

The class ``chase::grid::MpiGrid2DBase`` is the abstract base class for 
``chase::grid::MpiGrid2D``, providing the interface for grid management.

**API Reference**: :ref:`api_grid`


Backend Types
--------------

chase::grid::backend::MPI
^^^^^^^^^^^^^^^^^^^^^^^^^^

The struct ``chase::grid::backend::MPI`` is a type tag indicating that MPI 
should be used for communication. This is the standard backend for CPU-based 
parallel computation and can also be used with CUDA-aware MPI for GPU computation.

**API Reference**: :ref:`api_grid`


chase::grid::backend::NCCL
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The struct ``chase::grid::backend::NCCL`` is a type tag indicating that NCCL 
(NVIDIA Collective Communications Library) should be used for GPU-to-GPU 
communication. This backend provides optimized communication for multi-GPU 
setups and is only available when NCCL is enabled.

**API Reference**: :ref:`api_grid`


Grid Major Ordering
--------------------

The ``chase::grid::GridMajor`` enumeration specifies the major ordering of the 
MPI grid:

  * ``chase::grid::GridMajor::RowMajor``: Row-major grid layout
  * ``chase::grid::GridMajor::ColMajor``: Column-major grid layout (typically used)


Linear Algebra Kernels
======================

The ChASE library implements low-level numerical kernels in the 
``chase::linalg::internal`` namespace. These kernels are organized by 
computational platform and provide the building blocks for the higher-level 
algorithm implementations.

Kernel Organization
--------------------

The kernels are organized into several sub-namespaces:

  * **``chase::linalg::internal::cpu``**: CPU-based kernels using BLAS/LAPACK
    (:ref:`API Reference <api_kernels>`)
  * **``chase::linalg::internal::cuda``**: GPU-based kernels using cuBLAS/cuSOLVER
    (:ref:`API Reference <api_kernels>`)
  * **``chase::linalg::internal::mpi``**: Distributed CPU kernels using MPI and ScaLAPACK
    (:ref:`API Reference <api_kernels>`)
  * **``chase::linalg::internal::nccl``**: Distributed GPU kernels using NCCL
    (:ref:`API Reference <api_kernels>`)
  * **``chase::linalg::internal::cuda_aware_mpi``**: GPU kernels with CUDA-aware MPI
    (:ref:`API Reference <api_kernels>`)

Core Kernel Functions
---------------------

Each kernel namespace provides implementations of the following operations:

  * **Rayleigh-Ritz Projection**:
    - ``rayleighRitz()``: Standard Rayleigh-Ritz for Hermitian problems
    - ``pseudo_hermitian_rayleighRitz()``: Oblique Rayleigh-Ritz for pseudo-Hermitian problems

  * **Lanczos Algorithm**:
    - ``lanczos()``: Spectrum estimation for Hermitian problems
    - ``pseudo_hermitian_lanczos()``: Spectrum estimation for pseudo-Hermitian problems

  * **Matrix Operations**:
    - ``hemm()``: Hermitian matrix-matrix multiplication
    - ``pseudo_hermitian_hemm()``: Pseudo-Hermitian matrix-matrix multiplication

  * **Factorization**:
    - ``cholqr()``: Cholesky-QR factorization (with S-orthonormalization for pseudo-Hermitian)

  * **Residual Computation**:
    - ``residuals()``: Compute eigenpair residuals

  * **Utility Functions**:
    - ``shiftDiagonal()``: Diagonal shifting for Chebyshev filter
    - ``flipSign()``: Sign flipping operations
    - ``symOrHerm()``: Symmetry/Hermiticity checks

Type Traits
-----------

The ``chase::linalg::internal`` namespace also provides type traits for 
determining multi-vector types:

  * ``ResultMultiVectorType<MatrixType, InputMultiVectorType>``: Determines result 
    multi-vector type for operations
    
  * ``ColumnMultiVectorType<MatrixType>``: Column multi-vector type for a matrix
    
  * ``RowMultiVectorType<MatrixType>``: Row multi-vector type for a matrix

.. note::

   Detailed documentation of these kernels is available in the developer 
   documentation. For user-facing documentation, the implementation classes 
   (see :ref:`implementation_classes`) provide the main interface.


Platform Types
===============

The ``chase::platform`` namespace provides type tags for identifying computational 
platforms:

  * **``chase::platform::CPU``**: CPU platform identifier
  * **``chase::platform::GPU``**: GPU platform identifier

These types are used as template parameters in matrix classes and other components 
to specify the target platform for computation.

**API Reference**: :ref:`api_platform`


Type Utilities
===============

The ``chase`` namespace provides type utilities for working with complex numbers 
and precision conversion:

  * **``chase::Base<T>``**: Type trait to extract the base type from 
    ``std::complex<T>``. For example, ``chase::Base<std::complex<double>>`` 
    is ``double``.

  * **Precision Conversion Traits**: Type traits for converting between single 
    and double precision, supporting mixed-precision computations.

**API Reference**: See individual API pages above


C and Fortran Interfaces
==========================

The ChASE library provides C and Fortran interfaces for users who prefer not to 
use the C++ API directly. These interfaces are located in the ``interface/`` 
directory.

C Interface
------------

The C interface provides functions for initializing and solving eigenvalue 
problems. The function naming convention follows:

  * **Initialization functions**: ``{s|d|c|z}chase_init_`` for sequential, 
    ``p{s|d|c|z}chase_init_`` for parallel
    - ``s``: single precision real
    - ``d``: double precision real
    - ``c``: single precision complex
    - ``z``: double precision complex

  * **Solver functions**: ``{s|d|c|z}chase_`` for sequential, 
    ``p{s|d|c|z}chase_`` for parallel

  * **Finalization functions**: ``{s|d|c|z}chase_finalize_``

  * **Parallel variants**: Additional functions for block-cyclic distribution, 
    e.g., ``p{s|d|c|z}chase_init_blockcyclic_``

For example:
  - ``zchase_init_()``: Initialize double-precision complex sequential solver
  - ``pzchase_init_()``: Initialize double-precision complex parallel solver
  - ``pzchase_init_blockcyclic_()``: Initialize with block-cyclic distribution


Fortran Interface
-----------------

The Fortran interface provides the same functionality as the C interface but 
with Fortran naming conventions (without the trailing underscore). The interface 
uses ``iso_c_binding`` for interoperability with the C implementation.

For example:
  - ``zchase_init()``: Fortran subroutine corresponding to ``zchase_init_()``
  - ``pzchase_init()``: Fortran subroutine corresponding to ``pzchase_init_()``

.. note::

   For detailed documentation of the C and Fortran interfaces, including 
   function signatures and usage examples, please refer to the 
   :doc:`usage` documentation and the example programs in the ``examples/`` directory.
