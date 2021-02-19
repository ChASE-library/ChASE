..
  todo:: Depending on the parallel implementation (SEBASTIAN + JAN)

   * Installing Elemental for ChASE-Elemental (just a brief intro)
   * LAPACK+BLAS (e.g. MKL library)
   * CuBLAS
   * MAGMA
   * ...

Dependencies
------------

In order to install the ChASE library on a general purpose computing cluster,
one has to install or load the necessary dependencies. For the standard MPI
version of the library these dependencies are the following:

* a ``C++`` Compiler;
* a Message Passing Interface (MPI) implementation;
* CMake (version 3.8 or higher);
* a Basic Linear Algebra Subprograms (BLAS) and Linear Algebra PACKage (LAPACK) library;
* a CUDA compiler (only for the GPU build of ChASE);
* the Elemental library (ChASE + Elemental only).

Loading Modules on Cluster
---------------------------

CMake builds ChASE by automatically detecting the location of the
installed dependencies. On most supercomputers it is sufficient to
just load the corresponding modules, e.g. ``module load
<modulename>``. If you have loaded/installed multiple versions for the
necessary compilers and libraries, then you have to provide CMake with
specific paths so that it may choose the correct package. For more
details, see :ref:`build-label`.
