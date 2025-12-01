************************************
Installation and Setup on a Cluster
************************************

This section provides guidelines to install and setup the ChASE
library on computing clusters.
Each guideline is given as a step by step set of instructions to install ChASE on a
cluster either with or w/o support for GPUs. After the setup of ChASE,
users are provided with a description of how 
to link and integrate ChASE into their own codes.
ChASE provides the interfaces to both ``C`` and ``Fortran`` , 
thus such integration can be achieved either by using one of the 
interfaces provided by the library, or by following our instructions
to implement the user's own interface.


Library Dependencies
====================

This section gives a brief introduction to the ChASE library dependencies and
on how to load the required libraries on a given supercomputer.


Dependencies
------------

In order to install the ChASE library on a general purpose computing cluster,
one has to install or load the necessary dependencies.

**Required Dependencies:**
   * A ``C++`` compiler with C++17 support (e.g., GCC 7+, Clang 5+, or Intel C++ 17+)
   * `CMake <http://www.cmake.org/>`__ version 3.8 or higher
   * `BLAS <http://netlib.org/blas>`__ (Basic Linear Algebra Subprograms)
   * `LAPACK <http://netlib.org/lapack>`__ (Linear Algebra PACKage)

**Optional Dependencies:**
   * `MPI <http://en.wikipedia.org/wiki/Message_Passing_Interface>`__ - Required only for parallel implementations (pChASECPU, pChASEGPU). Not needed for sequential builds (ChASECPU, ChASEGPU).
   * `CUDA <https://developer.nvidia.com/cuda-toolkit>`__ - Required only for GPU implementations (ChASEGPU, pChASEGPU)
   * `ScaLAPACK <http://www.netlib.org/scalapack/>`__ - Optional, for distributed Householder QR factorization
   * `NCCL <https://developer.nvidia.com/nccl>`__ - Optional, for optimized multi-GPU communication in pChASEGPU

Note: For building examples with command-line parsing, the `popl <https://github.com/badaix/popl>`__ library is automatically downloaded by CMake using FetchContent. No manual installation is required.

Loading Modules on Cluster
---------------------------

CMake builds ChASE by automatically detecting the location of the
installed dependencies. On most supercomputers it is sufficient to
just load the corresponding modules, e.g. ``module load
<modulename>``. If you have loaded/installed multiple versions for the
necessary compilers and libraries, then you have to provide CMake with
specific paths so that it may choose the correct package. For more
details, see :ref:`build-label`.


Installation
=========================

This section has two main goals: First, it provides the instructions
for the installation of ChASE on a given supercomputer
with or w/o multi-GPUs supports. Second, it describes how the user can
take advantage of a number of ready-to-use examples to build a 
simple driver and have a first try running ChASE on a cluster.

Installation on a CPU-only Cluster
------------------------------------


The following snippet shows how to install ChASE on the JUWELS cluster
(the main general purpose cluster at the Jülich Supercomputing Centre):

.. code-block:: console

  git clone https://github.com/ChASE-library/ChASE.git
  cd ChASE/
  mkdir build
  cd build/
  ###       GCC      ###
  ml GCC/8.3.0  ParaStationMPI/5.4.4-1 imkl CMake
  cmake .. -DCMAKE_INSTALL_PREFIX=${ChASEROOT}
  make install
  ### Intel Compiler ###
  ml intel-para CMake
  cmake .. -DCMAKE_INSTALL_PREFIX=${ChASEROOT} -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc
  make install

.. note::

  For the installation with the ``Intel Compiler``, two additional flags ``-DCMAKE_C_FLAGS=-no-multibyte-chars`` and ``-DCMAKE_CXX_FLAGS=-no-multibyte-chars`` might be required if
  the following error ``Catastrophic error: could not set locale "" to
  allow processing of multibyte characters`` are encountered, which are produced by an internal
  bug appearing in some versions of the Intel Compiler.

Installation with GPU Support
------------------------------------

In order to compile ChASE with GPU support one needs to have installed
and loaded a CUDA compiler, which will also enable the use of
``cuBLAS``, ``cuSOLVER`` and ``cuRAND``. On the JUWELS cluster this can
be achieved by loading the module CUDA in addition to the
modules necessary to compile ChASE on a CPU-only cluster. Make sure, that
ChASE is executed on a computer/node with at least one GPU device
(e.g. check with ``nvidia-smi``) and that the correct  CUDA
compiler is loaded (e.g. check ``which nvcc`` or if you are using a module system
look at ``module list``). The following instruction snippet builds ChASE with CUDA
support on JUWELS:

.. code-block:: console

  git clone https://github.com/ChASE-library/ChASE.git
  cd ChASE/
  mkdir build
  cd build/
  ml GCC/8.3.0  ParaStationMPI/5.4.4-1 imkl CUDA CMake
  cmake .. -DCMAKE_INSTALL_PREFIX=${ChASEROOT}
  make install


.. note::
  It is also recommended to build ChASE with the configuration of CUDA compute compatibility through ``CMAKE_CUDA_ARCHITECTURES``. If CUDA compute compatibility of your GPU is 8.6 (e.g. RTX 3090) you should build with ``-DCMAKE_CUDA_ARCHITECTURES=86``. In the case you want to build code for more than one CUDA compute capability (e.g. 70, 75, 80 and 86) then build with ``-DCMAKE_CUDA_ARCHITECTURES="70;75;80;86"``.

  If ``CMAKE_VERSION < 3.18`` then CMake is not compliant with *CMAKE policy CMP0104* (introduced 
  in CMake 3.18) which defines that the variable ``CMAKE_CUDA_ARCHITECTURES`` has to be initialized. In that case, the code generation flag has to be set manually. 
  For simplicity and compatibility with newer (3.18+) CMake version, the ``CMAKE_CUDA_ARCHITECTURES`` variable has to be always set, not matter the cmake version.


Building ChASE with Examples
---------------------------------

To build and install ChASE with examples, the 
additional option to the cmake build process
``-DCHASE_BUILD_WITH_EXAMPLES=ON`` has to be turned on. The following
instruction snippet builds ChASE with
examples on the JUWELS cluster:

.. code-block:: console

  git clone https://github.com/ChASE-library/ChASE.git
  cd ChASE/
  mkdir build
  cd build/
  ml intel-para CMake
  ##### If you want to install ChASE with GPU supporting, make sure CUDA is loaded #####
  ml load CUDA
  cmake .. -DCMAKE_INSTALL_PREFIX=${ChASEROOT} -DCHASE_BUILD_WITH_EXAMPLES=ON
  make install
  ### Run example #0 ###
  ./examples/0_hello_world/0_hello_world

An MPI launcher has to be used to run an example in parallel. For
instance on the JUWELS cluster (or any other ``SLURM`` based Cluster)
the following command line runs the "`hello world`" example in parallel.

.. code-block:: console

  srun -n 2 ./examples/0_hello_world/0_hello_world


.. note::
  The output of intermediate convergence information and a simple performance report of 
  different numerical kernels can be enabled when compiling ChASE with the flag ``-DCHASE_OUTPUT=ON``.

Recommendation on the usage of Computing Resources
====================================================

Attaining the best performance with the available computing resources
requires to understand the inner working of the ChASE library. Since
the standard user is not expected to have such an understanding, this section
supplies a number of simple recommendations for the submission and
execution of jobs involving ChASE on a given computing cluster.

ChASE with MPI+OpenMP
---------------------

Modern homogeneous supercomputers are often equipped with hundreds of thousands of nodes which
are connected with fast networks. Each node is of NUMA (Non-uniform memory access) types, which
composes several NUMA domains. Each NUMA domain has its local memory, and is able to access the
local memory of another NUMA domain within the same node. Within a
NUMA domain, a processor can access
its own local memory faster than any other non-local memory.

When running ChASE on modern homogeneous clusters in the ``MPI/OpenMP`` hybrid mode, this `NUMA effect`
should be considered. In order to attain good performance, we recommend:

    1. Ensure each NUMA domain having at least 1 MPI task.
    
    2. Bind the CPUs to the relevant MPI tasks.


Allocating Ressources and Running jobs (SLURM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The optimal use of resources is usually achieved by carefully
designing the script code which is used for the job submission. An
example of a job script for a  ``SLURM`` scheduler is given below:

.. code-block:: bash

    # This is an example on JUWELS, in which each node is composed of 2 NUMA sockets.
    # This example allocates 4 nodes, 8 MPI tasks, each socket has 1 task,
    # and 24 CPUs are bound to each MPI tasks.
    #!/bin/bash -x
    #SBATCH --nodes=4
    #SBATCH --ntasks=8
    #SBATCH --ntasks-per-socket=1
    #SBATCH --cpus-per-task=24

Memory Requirement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An important aspect of executing ChASE on a parallel cluster is the
memory footprint of the library. It is important to avoid that such
memory footprint per MPI task exceeds the amount of main memory
available to the compiled code. The memory requirements differ between
**Hermitian** and **Pseudo-Hermitian** (Quasi-Hermitian) eigenvalue problems
due to the additional storage needed for the dual basis and oblique Rayleigh-Ritz
procedure in the pseudo-Hermitian case.

**Hermitian Eigenvalue Problems**

For **Block distribution** of matrix, the memory requirement per MPI rank is::

  sizeof(float_type) *[n * m + 2 * (n + m) * block + 1 + 5*block + 2*pow(block,2)]/(1024^3) GigaByte

For **Block-Cyclic distribution** of matrix, additional memory of
size ``sizeof(float_type) * N`` is required for managing the internal reshuffling
for block-cyclic data layout. Thus the total memory required is::

  sizeof(float_type) *[n * m + 2 * (n + m) * block + N + 1 + 5*block + 2*pow(block,2)]/(1024^3) GigaByte

**Pseudo-Hermitian Eigenvalue Problems**

Pseudo-Hermitian problems (e.g., from Bethe-Salpeter Equation) require additional
memory for the dual basis vectors and larger workspace matrices used in the
oblique Rayleigh-Ritz procedure. For **Block distribution** of matrix, the
memory requirement per MPI rank is::

  sizeof(float_type) *[n * m + 4 * (n + m) * block + 3 * block * block + 1 + 5*block + 2*pow(block,2)]/(1024^3) GigaByte

For **Block-Cyclic distribution** of matrix, additional memory of
size ``sizeof(float_type) * N`` is required for managing the internal reshuffling::

  sizeof(float_type) *[n * m + 4 * (n + m) * block + 3 * block * block + N + 1 + 5*block + 2*pow(block,2)]/(1024^3) GigaByte

**Common Parameters**

In the formulas above:
   * ``n`` and ``m`` are fractions of ``N`` which depend on the size
     of the MPI grid of processors. For instance in the job script above
     ``n = N/nrows`` and ``m = N/ncols``, with the size of MPI grid ``nrows*ncols``.
   * ``N`` is the size of the eigenproblem.
   * ``block`` is at most ``nev + nex``, where ``nev`` is the number of wanted
     eigenpairs and ``nex`` is the extra search dimensions.
   * ``sizeof(float_type)`` is valid for single precision real,
     double precision real, single precision complex and double precision complex floating numbers.
     The value of this factor for these four types of floating numbers are respectively:
     ``4``, ``8``, ``8``, ``16``.

**Example**

Using such formulas one can verify if the allocation of
resources is enough to solve for the problem at hand. For instance, if we use **Block distribution**
for a Hermitian problem with ``N = 360,000`` and a ``block = nev + nex = 3,000`` with ``1152`` MPI ranks in 2D MPI grid of size ``32x36``, 
the required memory per MPI rank is ``1.989 GB``. For ChASE with **Block-Cyclic Distribution**: the memory requirement per MPI-rank
is ``1.992 GB``, a little larger than the former case.

For the same problem size but with a **Pseudo-Hermitian** matrix using **Block distribution**,
the memory requirement per MPI rank is approximately ``2.45 GB``, reflecting the additional
storage needed for the dual basis and oblique Rayleigh-Ritz workspace.


ChASE with multi-GPUs
---------------------

Currently, ChASE is able to offload almost all the intensive computations, e.g., Hermitian Matrix-Matrix 
Multiplications), QR factorization and Rayleigh-Ritz computation to GPUs. 
The multi-GPUs version of ChASE is able to use all available cards for
each node. This multi-GPUs version supports 1 MPI task
to manage only 1 bound GPU card. Some less intensive computation is also assigned to this MPI task and executed
in multi-threading mode.

Allocating Ressources and Running jobs (SLURM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below is an example of a job script for a ``SLURM`` scheduler which allocates
multi-GPUs per node and each GPU card bound to 1 MPI task:

.. code-block:: bash

    # This is an example on the JUWELS GPU partition, in which each node has 4 V100 NVIDIA GPUs.
    # This example allocates 4 nodes, 16 MPI tasks, each node has 4 task,
    # and 4 GPUs per node, each GPU card is bound to 1 MPI task.
    #!/bin/bash -x
    #SBATCH --nodes=4
    #SBATCH --ntasks=16
    #SBATCH --ntasks-per-node=4
    #SBATCH --cpus-per-task=24
    #SBATCH --gres=gpu:4


Estimating Memory Requirement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ChASE with multi-GPUs, the memory requirements differ between Hermitian and
Pseudo-Hermitian problems, similar to the CPU case.

**Hermitian Eigenvalue Problems**

For both **Block distribution** and **Block-Cyclic distribution** of matrix,
the memory requirement per GPU is::

  sizeof(float_type) *[n * m + 2 * (n + m) * block + 1 + 5*block + 2*pow(block,2)]/(1024^3) GigaByte

For **Block-Cyclic distribution**, add ``sizeof(float_type) * N/(1024^3)`` to account
for the internal reshuffling buffer.

**Pseudo-Hermitian Eigenvalue Problems**

For **Block distribution** of matrix, the memory requirement per GPU is::

  sizeof(float_type) *[n * m + 4 * (n + m) * block + 3 * block * block + 1 + 5*block + 2*pow(block,2)]/(1024^3) GigaByte

For **Block-Cyclic distribution**, add ``sizeof(float_type) * N/(1024^3)`` to account
for the internal reshuffling buffer.

The parameters ``n``, ``m``, ``N``, ``block``, and ``sizeof(float_type)`` have the same
meaning as described in the CPU memory requirement section above.

.. warning::

    The estimation of memory requirement is only based on the algorithmic aspects of ChASE. The buffer and memory requirement of libraries such as ``MPI`` has not been considered. So despite the provided formulas to calculate the memory consumption, some combination of MPI libraries (e.g., ParastationMPI) could lead to the crash of ChASE with ``out of memory`` even if the memory available is within the estimated bounds. 


CMake Configuration Options
===========================

This section provides a comprehensive list of all CMake configuration options
available when building ChASE. These options can be set using the ``-D`` flag
during the CMake configuration step, e.g., ``cmake .. -DOPTION_NAME=value``.

ChASE-Specific Options
-----------------------

.. list-table:: ChASE CMake Configuration Options
   :widths: 35 10 55
   :header-rows: 1

   * - Option Name
     - Default
     - Description
   * - ``CHASE_OUTPUT``
     - ``OFF``
     - Enable output of intermediate convergence information and
       performance reports of different numerical kernels at each
       iteration. When enabled, ChASE will print detailed information
       during the solution process.
   * - ``CHASE_ENABLE_OPENMP``
     - ``ON``
     - Enable OpenMP support for multi-threading. This option enables
       parallel execution within a single MPI rank using OpenMP threads.
       Set to ``OFF`` to disable OpenMP support.
   * - ``CHASE_ENABLE_MIXED_PRECISION``
     - ``OFF``
     - Enable mixed precision support. When enabled, ChASE can use
       different floating-point precisions for different operations
       to optimize performance while maintaining accuracy.
   * - ``CHASE_ENABLE_MPI_IO``
     - ``OFF``
     - Enable MPI I/O functionality to read Hamiltonian matrices
       from local files in parallel. This is useful for loading
       large matrices distributed across multiple MPI processes.
   * - ``CHASE_USE_NVTX``
     - ``OFF``
     - Enable NVIDIA Tools Extension (NVTX) for profiling GPU
       operations. This option is useful for performance analysis
       and debugging on NVIDIA GPUs using tools like Nsight Systems.
   * - ``CHASE_BUILD_WITH_EXAMPLES``
     - ``OFF``
     - Build the example programs provided with ChASE. When enabled,
       example executables will be built in the ``examples/``
       directory. The ``popl`` library for command-line parsing
       will be automatically downloaded if needed.
   * - ``CHASE_BUILD_WITH_DOCS``
     - ``OFF``
     - Build the documentation using Sphinx. When enabled, HTML
       documentation will be generated in the build directory.
   * - ``ChASE_DISPLAY_COND_V_SVD``
     - ``OFF``
     - Compute and display the condition number of the matrix V
       from the Singular Value Decomposition (SVD). This is useful
       for debugging and understanding numerical stability.
   * - ``ENABLE_TESTS``
     - ``OFF``
     - Enable building of unit tests. When enabled, GoogleTest
       will be automatically downloaded and test executables will
       be built. Requires MPI to be available for parallel tests.

Standard CMake Options
----------------------

The following standard CMake variables can also be used to configure the build:

**Installation Path:**
   * ``CMAKE_INSTALL_PREFIX`` - Installation directory for ChASE (default: ``/usr/local`` on Unix systems)

**Compiler Selection:**
   * ``CMAKE_CXX_COMPILER`` - Path to the C++ compiler (e.g., ``/usr/bin/g++``, ``/usr/bin/clang++``)
   * ``CMAKE_C_COMPILER`` - Path to the C compiler (e.g., ``/usr/bin/gcc``, ``/usr/bin/clang``)
   * ``CMAKE_Fortran_COMPILER`` - Path to the Fortran compiler (e.g., ``/usr/bin/gfortran``)

**MPI Configuration:**
   * ``MPI_CXX_COMPILER`` - Path to the MPI C++ compiler wrapper (e.g., ``/usr/bin/mpicxx``)
   * ``MPI_C_COMPILER`` - Path to the MPI C compiler wrapper (e.g., ``/usr/bin/mpicc``)
   * ``MPI_Fortran_COMPILER`` - Path to the MPI Fortran compiler wrapper (e.g., ``/usr/bin/mpif90``)

**CUDA Configuration:**
   * ``CMAKE_CUDA_ARCHITECTURES`` - CUDA compute capability architectures to target. Can be a single
     value (e.g., ``86`` for RTX 3090) or a semicolon-separated list (e.g., ``"70;75;80;86"``).
     This option should always be set when building with CUDA support, regardless of CMake version.

**Build Type:**
   * ``CMAKE_BUILD_TYPE`` - Build type: ``Release`` (optimized, default), ``Debug`` (with debug symbols),
     ``RelWithDebInfo`` (optimized with debug info), or ``MinSizeRel`` (minimum size)

Example Usage
-------------

Here are some example CMake configuration commands demonstrating the use of various options:

**Basic build with examples:**
   .. code-block:: console

      cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install \
               -DCHASE_BUILD_WITH_EXAMPLES=ON

**Build with GPU support and specific CUDA architecture:**
   .. code-block:: console

      cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install \
               -DCMAKE_CUDA_ARCHITECTURES=86 \
               -DCHASE_BUILD_WITH_EXAMPLES=ON

**Build with debugging output and profiling:**
   .. code-block:: console

      cmake .. -DCMAKE_BUILD_TYPE=Debug \
               -DCHASE_OUTPUT=ON \
               -DCHASE_USE_NVTX=ON

**Build with custom compilers and MPI:**
   .. code-block:: console

      cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
               -DCMAKE_C_COMPILER=/usr/bin/gcc-11 \
               -DMPI_CXX_COMPILER=/usr/bin/mpicxx \
               -DCHASE_ENABLE_OPENMP=ON

**Build with tests enabled:**
   .. code-block:: console

      cmake .. -DENABLE_TESTS=ON \
               -DMPI_RUN=srun \
               -DMPI_RUN_ARGS="--ntasks=4"

Note: When using ``ENABLE_TESTS=ON``, you may also need to set ``MPI_RUN`` (the MPI launcher command,
e.g., ``mpirun`` or ``srun``) and optionally ``MPI_RUN_ARGS`` (additional arguments for the MPI launcher).



