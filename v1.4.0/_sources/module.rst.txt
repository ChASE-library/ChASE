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
matrix-matrix multiplication and QR factorization. This modern “stand-alone” strategy grants 
ChASE an unprecedented degree of flexibility that makes the integration of this library in 
most application codes quite simple. ChASE efficiently uses available machine resources.

We give an UML class diagram as follows. This diagram uses the implementation of 
Chebyshev `Filter`, whose kernel is a series of Hermitian Matrix-Matrix Products (*HEMM*), as an example, 
to show the scheme of implementing of ChASE and porting to different architectures. This section will
give the user an insight on how to set up their eigenproblems and
solve them by ChASE on different computing architectures.

.. image:: /images/ChASE_UML.jpg
   :scale: 99 %
   :align: center


As shown in the diagram above, all the implementation of ChASE take place within a C++ namespace ``chase``.  

Basic Classes
================

chase::Chase
--------------

The numerical kernels required by ChASE algorithm are defined in class ``chase::Chase``. All the
functions are defined as virtual functions, and further implementations are required targeting different
computing architectures. It includes the
following functionalities:

  * ``HEMM``: Hermitian Matrix-Matrix Multiplication

  * ``QR``: QR factorization

  * ``RR``: Rayleigh-Ritz projection and small problem solver

  * ``Resd``: compute the eigenpair residuals

  * ``Lanczos``: estimate the bounds of user-interested spectrum by Lanczos eigensolver

  * ``LanczosDos``: estimate the spectral distribution of eigenvalues

  * ``Swap``: swap the two matrices of vectors used in the Chebyschev filter

  * ``Locking``: locking the converged eigenpairs

  * ``Shift``: shift the diagonal of matrix ``A`` used in the 3-terms
    recurrence relation implemented in the Chebyschev filter

  * etc ..

.. note::

   For more details on the virtual kernels, 
   please refer to :ref:`virtual_abstrac_numerical_kernels`. 
   Different parallel implementations of these virtual kernels
   can also be found :ref:`parallel_implementations`.


chase::Algorithm
------------------

The class ``chase::Algorithm`` has the awareness of the class ``chase::Chase``, and
it defines algorithmic implementation of ChASE using the
defined virtual kernels in ``chase::Chase``. It includes the
functionalities:

  * Chebyshev filter

  * calculation of degree of the filter

  * Lanczos solver to estimate the bound of spectra

  * locking the converged Ritz pairs

  * etc ..

The function ``chase::Solve`` provides the implementation of ChASE algorithm by assembling the algorithms and numerical kernels implemented in ``chase::Chase`` and ``chase::Algorithm``.

.. note::
   
   This class implements the ChASE algorithm by the virtual functions, it cannot run in practice
   until the further implementations of these virtual functions are provided.

.. note::

  The details of this class are only provided in the developer documentation,
  please refer to :ref:`algorithmic-structure`.

chase::ChaseConfig
--------------------

The class ``chase::Algorithm`` is aware of the class ``chase::ChaseConfig``, which defines the functions to set different parameters of ChASE.

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
use to measure performance, especially parallel performance. The
timings are stored in a vector of objects derived by the class
template `std::chrono::duration`.


.. note::

  For more details of all available functions, please refer to
  :ref:`performance`.

chase::PerformanceDecoratorChase
-------------------------------------

This is a class derived from the ``chase::Chase`` which plays the
role of interface for the kernels used by the library. All
members of the ``chase::Chase`` class are virtual functions. These
functions are re-implemented in the ``chase::PerformanceDecoratorChase``
class. All derived members that provide an interface to
computational kernels are re-implemented by *decorating* the
original function with time pointers which are members of the
``chase::ChasePerfData`` class. All derived members that provide an
interface to input or output data are called without any
specific decoration. In addition to the virtual member of the
``chase::Chase`` class, the ``chase::PerformanceDecoratorChase`` class has also among
its public members a reference to an object of type
``chase::ChasePerfData``. When using Chase to solve an eigenvalue problem,
the members of the PerformanceDecoratorChase are called instead
of the virtual functions members of the ``chase::Chase`` class. In this
way, all parameters and counters are automatically invoked and
returned in the correct order.

.. note::

  For more details of all available functions, please refer to
  :ref:`performance`.


Override of Virtual Functions
================================

The exact implementation of numerical kernels used by ChASE are within the namespace ``chase::mpi``.
This namespace is defined inside the namespace ``chase``, which provides the parallel implementation of ChASE based on ``MPI`` (and ``CUDA``) by re-implementing
the numerical kernels as virtual functions within the abstraction targeting homogeneous and heterogeneous architectures (multi-GPUs).


chase::mpi::ChaseMpiMatrices
------------------------------

The class ``chase::mpi::ChaseMpiMatrices`` defines the allocation of buffers for matrices and vectors in ChASE library for both non-MPI mode and MPI mode.

.. note::

  For more details of all available functions, please refer to
  :ref:`ChaseMpiMatrices`.

chase::mpi::ChaseMpiProperties
--------------------------------

The class ``chase::mpi::ChaseMpiProperties`` defines the construction of MPI environment and data distribution scheme (both **Block Distribution** and **Block-Cyclic Distribution**) for ChASE. 
This class has the awareness of the class ``chase::mpi::ChaseMpiMatrices``. It will allocate the 
required buffer based on the configuration MPI environment and data distribution by using different
constructors of ``chase::mpi::ChaseMpiMatrices``.

.. note::

  For more details of all available functions, please refer to
  :ref:`ChaseMpiProperties`.


chase::mpi::ChaseMpi
----------------------

``chase::mpi::ChaseMpi`` is a derived class of ``chase::Chase``. This class gives an implementation of the virtual functions of ``chase::Chase`` class
which defines the essential numerical kernels of ChASE algorithm. It is a templated class with two types required: 
an implementation of ``chase::mpi::ChaseMpiDLAInterface`` and the scalar type to be used in the applications. The numerical kernels defined in ``chase::Chase`` has further decoupled into Dense Linear
Algebra operations (DLAs). Different objects of ``chase::mpi::ChaseMpi``
can be created targeting different computing platforms by selecting various derived classes of ``chase::mpi::ChaseMpiDLAInterface``. 

To be more precise, it is derived from the ``chase::Chase`` class 
which plays the role of interface for the kernels used by the library:
    
- All members of the ``chase::Chase`` class are virtual functions. These functions are re-implemented in the ``chase::mpi::ChaseMpi`` class.

- All the members functions of ``chase::mpi::ChaseMpi``, which are the implementation of the virtual functions in class ``chase::Chase``, are implemented using the *DLA* routines provided by the class ``chase::mpi::ChaseMpiDLAInterface``.
    
-  The DLA functions in ``chase::mpi::ChaseMpiDLAInterface`` are also virtual functions, which are differently implemented targeting different computing architectures (sequential/parallel, CPU/GPU, shared-memory/distributed-memory, etc). In the class ``chase::mpi::ChaseMpi``, the calling of DLA functions are indeed calling their implementations from different derived classes. Thus this ChaseMpi class is able to have customized implementation for various architectures.
    
- The class ``chase::mpi::ChaseMpi`` has the awareness of the class ``chase::mpi::ChaseMpiMatrices`` and ``chase::mpi::ChaseMpiProperties``. 

   - For the shared-memory implementation, the constructor of ``chase::mpi::ChaseMpi`` takes an instance of ``chase::mpi::ChaseMpiMatrices`` as input

   - For the distributed-memory implementation of the class ``chase::mpi::ChaseMpi``, the setup of MPI environment and communication scheme, and the distribution of data (matrix, vectors) across MPI nodes are following the ``chase::mpi::ChaseMpiProperties`` class, the distribution of matrix can be either **Block** or **Block-Cyclic** scheme. The required buffers are allocated during the construction of an object of ``chase::mpi::ChaseMpiProperties``.

.. note::

  For more details of all available functions, please refer to
  :ref:`ChaseMpi`.


DLAs for shared-memory architectures 
--------------------------------------

The DLAs for shared-memory architectures with or without GPU are implemented in the classes ``chase::mpi::ChaseMpiDLACudaSeq`` and ``chase::mpi::ChaseMpiDLABlaslapackSeq``, respectively.


.. toctree::
   :maxdepth: 3

   module/chasempidlaseq
   module/chasempidlacudaseq 


DLAs for distributed-memory architectures 
-------------------------------------------

For the implementation of DLAs for distributed-memory architectures, they have been further decoupled
into two layers:

   - the first layer is for the collective communication between different computing nodes

   - the second layer is for the implementation of local computation within each node

      - for homogeneous systems with CPUs-only, the local computation takes place on each individual MPI processor, with potential parallelization of multi-threading, e.g., with OpenMP. 

      - for the heterogeneous systems with GPUs, some local computation takes place on each individual MPI processor, and more intensive computation are offloaded to each GPU bound to relevant MPI processor.


The local computations with or without GPUs are implemented in the classes ``chase::mpi::ChaseMpiDLABlaslapack`` and ``chase::mpi::ChaseMpiDLAMultiGPU``, respectively. 

The collective communication layer is shared between the distributed memory ChASE with or without GPU support, which is implemented in the class ``chase::mpi::chaseMpiDLA``. This class takes an instance of ``chase::mpi::ChaseMpiDLAInterface``, either ``chase::mpi::ChaseMpiDLABlaslapack`` or ``chase::mpi::ChaseMpiDLAMultiGPU`` as input. In this way, it is able to access to different implementations of local computation kernels.

.. note::

   When an instance of ``chase::mpi::ChaseMpi`` is constructed for distributed-memory systems, one of its template parameter should be provided either ``chase::mpi::ChaseMpiDLABlaslapack`` and ``chase::mpi::ChaseMpiDLAMultiGPU``. Then a instance of the class ``chase::mpi::ChaseMpiDLA`` will the
   created with the selected implementations of local computations kernels. In this way, ChASE is able to
   be ported to different computation architectures.


.. toctree::
   :maxdepth: 3

   module/chasempidlaImpl
   module/chasempidlablaslapack
   module/chasempidlamultigpu

