************************************
Modules
************************************

Overview
=============

The implementation of ChASE provides a stand-alone high-performance 
parallel library based on our original design of the Chebyshev
accelerated subspace iteration algorithm. the ChASE library promises the portability 
to heterogeneous architectures and the easy intergration ito existing codes. This goal is
achieved by seperating the implementation of the ChASE algorithm from the required numerical
kernels via an interface based on pure C++ abstract classes. Classes derived from this interface 
handle data distribution and (parallel) execution of each kernel. The required numerical kernels 
are based on Basic Linear Algebra Subprograms (BLAS)-3 compatible kernels, such as a (parallel) 
matrix-matrix multiplication and QR factorization. This modern “stand-alone” strategy grants 
ChASE an unprecedented degree of flexibility that makes the integration of this library in 
most application codes quite simple. ChASE efficiently uses available machine resources.

In this section, we introduce some important modules with the aim of
giving the user an insight on how to set up their eigenproblems and
solve them by ChASE on different computing architectures.


Namespace: chase
==================

An overview of the namespaces and related important classes defined in
ChASE is given in this section.
The basic namespace of ChASE is ``chase``, which includes (1) the abstract 
classes of numerical kernels with virtual functions; (2) the numerical algorithm of ChASE implemented by these virtual 
functions; (3) the module that sets the configuration of the parameter of ChASE; (4) the module
that sets the profiling and timing of the numerical kernels of ChASE.


.. toctree::
   :maxdepth: 3

   module/namespace

Selected Classes
-----------------

Several important classes, implemented in the namespace ``chase``, are
illustrated in this section.
which faciliate the user for the parameter configuration and performance profiling.

.. toctree::
   :maxdepth: 2

   module/config
   module/perf


ChASE-MPI
==========

The namespace ``chase::mpi``, which is defined inside the namespace ``chase`` and in parallel with namespace ``chase::elemental``,
provides the parallel implementation of ChASE based on ``MPI`` (and ``CUDA``) by re-implementing
the numerical kernels as virtual functions within the abstraction targeting homogenous and hetergenous architectures (multi-GPUs).
For the user documentation, we provides only the constructors of the related classes without
the details of implementation.


1. The class ``ChaseMpiMatrices`` defines the construction of matrices and vectors
for both non-MPI mode and MPI mode which are used in ChASE.

.. toctree::
   :maxdepth: 2

   module/chasempimatrices

2. The class ``ChaseMpiProperties`` defines the construction of MPI environment for ChASE, both for the block
distribution and block-cyclic distribution.   

.. toctree::
   :maxdepth: 2

   module/chasempiproperties


3. ``ChaseMpi`` defines the ChASE-MPI class. It is an templated classes with two types required: 
an implementation of ``ChaseMpiDLAInterface`` and the scalar type to be used in the applications.


.. toctree::
   :maxdepth: 2

   module/chasempi

4. ``ChaseMpiDLA``, in which ``DLA`` is short for ``Dense Linear Algebra``, provides multi-implementation
of dense matrix operations on different computing architectures that are required by ``ChASE``, e.g., Hermitian Matrix-Matrix Multiplication (HEMM), QR factorization, etc. These 's no single class ``ChaseMpiDLA`` in ``ChASE``, we provide an interface class for these `DLA` operations, and multiple classes with different implentations derived from this interface class.

.. toctree::
   :maxdepth: 3

   module/chasempidla


ChASE-Elemental
================

The namespace ``chase::elemental``, which is defined inside the namespace ``chase``,
provides the parallel implementation of ChASE based on the Elemental library by re-implementing
the numerical kernels as virtual functions calls within the abstract class. For the benefit of the user, only the constructor of this ChaseElemental
class is provided here.

.. toctree::
   :maxdepth: 2

   module/elemental

