Parallel implementations
************************

Reference implementation
========================
ChASE reference version can be use as a sequential code or by setting
the variable NUM_THREADS = np
As part of the reference version a special build can be invoked that
offload only the filter to a GPU device.

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
