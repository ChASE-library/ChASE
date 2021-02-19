************************
Versions of the library
************************

The library comes in two main versions which are labelled ChASE-MPI
and ChASE-Elemental. ChASE-MPI is the default version of the library
and can be installed with the minimum amount of dependencies (BLAS,
LAPACK, and MPI). ChASE-Elemental requires the additional installation
of the `Elemental <https://github.com/elemental/Elemental>`__ library.

ChASE-MPI
==========

Multi-Configurations
---------------------

ChASE-MPI supports different configurations depending on the available
hardware resources.

   * **Shared memory build:** This is the simplest configuration and
     should be exclusively selected when ChASE is used on only one
     computing node or on a single CPU. The simplicity of this
     configuration resides in the way the Matrix-Matrix kernel is
     implemented with respect to the full MPI build.

   * **MPI+Threads build:** On multi-core homogeneous CPU clusters ChASE
     is best used in its pure MPI build. In this configuration, ChASE
     is typically used with one MPI rank per computing node and as
     many threads as number of available cores per node.

   * **GPU build:** ChASE-MPI can be configured to take advantage of
     graphics card on heterogeneous computing clusters. Currently only
     one GPU card per MPI rank is supported. Support for multi-GPUs is
     currently under testing phase and it it is expected to be part of the
     next release in 2021.


Matrix Distribution
--------------------

In ChASE-MPI, the MPI nodes are constructed as 2D grid, in which each
MPI rank is assigned a block of dense matrix **A**. The most important
kernel of ChASE is the **Hermitian Matrix-Matrix Multiplication**. This
block data distribution results in a matrix-matrix multiplications on each
node that is large and contiguous, often resulting in a performance close to
the hardware theoretical peak. In addition, this data distribution allows an easy
offloading of the multiplication to accelerators such as GPUs.


.. math::

   A=\left(\begin{array}{c|c|c}
     A_{0,0} & A_{0,1} & A_{0,2}  \\
     \hline
     A_{1,0} & A_{1,1} & A_{1,2}  \\
     \hline
     A_{2,0} & A_{2,1} & A_{2,2}
   \end{array}\right)

The figure above gives an example of ChASE which distributes a :math:`n \times n`
dense matrix :math:`A` into a :math:`3 \times 3` grid of MPI nodes. In this example,
the matrix `A` is split into 2D, with a series of submatrices :math:`A_{i,j}` in which
:math:`i \in [0,2]` and :math:`j \in [0,2]`. Therefore, :math:`A_{0,0}` is distributed
to MPI rank 1, :math:`A_{1,0}` is distributed to rank 2, :math:`A_{2,0}` is distributed to rank 3, :math:`A_{0,1}` is distributed to rank 4, :math:`A_{1,1}` is distributed to rank 5, and so on.


ChASE-Elemental
================

ChASE-Elemental takes advantage of the `Elemental
<http://libelemental.org/>`__ library routines to execute its tasks in
parallel. The data is distributed using Elemental distribution
classes. ChASE-Elemental is available in a **pure MPI build** and,
while it is slightly less performant than ChASE-MPI, it ensure a
better scalability since, contrary to ChASE-MPI, the QR factorization
is also executed in parallel. For stability reason, ChASE uses
Elemental version 0.84.
