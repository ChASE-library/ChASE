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
one has to install or load the necessary dependencies. For the standard MPI
version of the library these dependencies are the following:

* a ``C++`` Compiler;
* a Message Passing Interface (MPI) implementation;
* CMake (version 3.8 or higher);
* a Basic Linear Algebra Subprograms (BLAS) and Linear Algebra PACKage (LAPACK) library;
* a CUDA compiler (only for the GPU build of ChASE);
* **Optional**: ScaLAPACK for using distributed Househoulder QR factorization

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
with or w/o multi-GPUs supports. Second, it describe how the user can
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
  allow processing of multibyte characters`` are encountered, which is produced by an internal
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
  It is also recommended to build ChASE with the configuration of CUDA compute compatibility through ``CMAKE_CUDA_ARCHITECTURES``. If cuda compute compatibility of your GPU is 8.6 (e.g. RTX 3090) you should build with ``-DCMAKE_CUDA_ARCHITECTURES=86``. In the case you want to build code for more than one CUDA compute capability (e.g. 70, 75, 80 and 86) then build with ``-DCMAKE_CUDA_ARCHITECTURES="70;75;80;86"``.

  If ``CMAKE_VERSION < 3.18`` then CMake is not compliant with *CMAKE policy CMP0104* (introduced 
  in CMake 3.18) which defines that the variable ``CMAKE_CUDA_ARCHITECTURES`` has to be initialized. In that case, the code generation flag has to be set manually. 
  For simplicity and compatibility with newer (3.18+) CMake version, the ``CMAKE_CUDA_ARCHITECTURES`` variable has to be always set, not matter the cmake version.


Building ChASE with Examples
---------------------------------

To build and install ChASE with examples, the 
additional option to the cmake build process
``-DBUILD_WITH_EXAMPLES=ON`` has to be turned on. The following
instruction snippet builds ChASE with
examples on the JUWELS cluster:

.. code-block:: console

  git clone https://github.com/ChASE-library/ChASE.git
  cd ChASE/
  mkdir build
  cd build/
  ml intel-para CMake Boost
  ##### If you want to install ChASE with GPU supporting, make sure CUDA is loaded #####
  ml load CUDA
  cmake .. -DCMAKE_INSTALL_PREFIX=${ChASEROOT} -DBUILD_WITH_EXAMPLES=ON
  make install
  ### Run example #0 ###
  ./examples/0_hello_world/0_hello_world

An MPI launcher has to be used to run an example in parallel. For
instance on the JUWELS cluster (or any other ``SLRUM`` based Cluster)
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
available to the compiled code. To help the user to make the correct
decision in terms of resources a simple formula for **Block distribution** of matrix can be used ::

  sizeof(float_type) *[n * m + 2 * (n + m) * block + 1 + 5*block + 2*pow(block,2)]/(1024^3) GigaByte

where ``n`` and ``m`` are fractions of ``N`` which depend on the size
of the MPI grid of processors. For instance in the job script above
``n = N/nrows`` and ``m = N/ncols``, with the size of MPI grid ``nrows*ncols``. 
Correspondingly ``N`` is
the size of the eigenproblem and ``block`` is at most ``nev + nex``.
Note that the factor ``sizeof(float_type)`` is valid for single precision real,
double precision real, single precision complex and double precision complex floating numbers.
The value of this factor for these four types of floating numbers are respectively:
``4``, ``8``, ``8``, ``16``.

For ChASE with **Block-Cyclic distribution** of matrix, additional memory of
size ``sizeof(float_type) * N`` is required for managing the internal reshuffling
for block-cyclic data layout. Thus the total memory required is::

  sizeof(float_type) *[n * m + 2 * (n + m) * block + N + 1 + 5*block + 2*pow(block,2)]/(1024^3) GigaByte


Using such a formula one can verify if the allocation of
resources is enough to solve for the problem at hand. For instance, if we use **Block distribution**
for a ``N = 360,000`` and a ``block = nev + nex = 3,000`` with ``1152`` MPI ranks in 2D MPI grid of size ``32x36``, 
the requirement memory per MPI rank is ``1.989 GB``. For ChASE with **Block-Cyclic Distribution**: the memory requirement per MPI-rank
is 1.992 GB, a littler larger than the former case.


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

For ChASE with multi-GPUs using both **Block distribution** and **Block-Cyclic distribution**
of matrix, the  memory requirement per GPU is always ::

  sizeof(float_type) *[n * m + 2 * (n + m) * block + 1 + 5*block + 2*pow(block,2)]/(1024^3) GigaByte

.. warning::

    The estimation of memory requirement is only based on the algorithmic aspects of ChASE. The buffer and memory requirement of libraries such as ``MPI`` has not been considered. So despite the provided formulas to calculate the memory consumption, some combination of MPI libraries (e.g., ParastationMPI) could lead to the crash of ChASE with ``out of memory`` even if the memory available is within the estimated bounds. 



