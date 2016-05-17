Parallel implementations
************************

Reference implementation
========================
ChASE reference version can be use as a sequential code. ChASE makes extensive
use of Level 3 BLAS routines, and so will benefit from multi-threaded BLAS.
Some BLAS versions require an environmental variable to be set in order to
employ multiple threads. I.e., for MKL set ``MKL_NUM_THREADS=np`` and for OpenBLAS
set ``OPENBLAS_NUM_THREADS=np``.
As part of the reference version a special build can be invoked that
offloads only the filter to a GPU device.

.. toctree::
   :maxdepth: 2

   reference/chase
   reference/lanczos
   reference/filter

CUDA implementation
===================
The CUDA ChASE version is a highly optimized version for one or more
GPU devices as long as they reside on the same compute nodecard.

MPI (Elemental) implementation
==============================
The pure MPI ChASE version is based on the Elemental library which
needs to be installed as a prerequisite
