************************************
Installation and Setup on a Cluster
************************************

This section provides guidelines to install and setup the ChASE
library on computing clusters.
Each guideline is given as a step by step set of instructions to install ChASE on a
cluster either with or w/o support for GPUs. After the setup of ChASE,
users are provided with a description of how 
to link and integrate ChASE into their own codes. Such integration can
be achieved either by using one of the 
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


Installation on Cluster
=========================

This section has two main goals: First, it provides the instructions
for the installation of ChASE on a given supercomputer
with or w/o multi-GPUs supports. Second, it describe how the user can
take advantage of a number of ready-to-use examples to build a 
simple driver and have a first try running ChASE on a cluster.

Installation on a CPU-only Cluster
------------------------------------


The following snippet shows how to install ChASE on the JUWELS cluster
(the main general purpose cluster at the Juelich Supercomputing Centre):

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
  allow processing of multibyte characters`` are encoutered, which is produced by an internal
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



Recommendation on the usage of Computing Resources
====================================================

Attaining the best performance with the available computing resources
requires to understand the inner working of the ChASE library. Since
the standard user is not expected to have such an understanding, this section
supplies a number of simple recommendations for the submission and
execution of jobs involving ChASE on a given computing cluster.

ChASE with MPI+OpenMP
---------------------

Modern homogenous supercomputers are often equipped with hunderds of thousands of nodes which
are connected with fast networks. Each node is of NUMA (Non-uniform memory access) types, which
composes several NUMA domains. Each NUMA domain has its local memory, and is able to access the
local memory of another NUMA domain within the same node. Within a
NUMA domain, a processor can access
its own local memory faster than any other non-local memory.

When running ChASE on modern homogenous clusters in the ``MPI/OpenMP`` hybrid mode, this `NUMA effect`
should be considered. In order to attain good performance, we recommand:

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

Estimating Memory Requirement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An important aspect of executing ChASE on a parallel cluster is the
memory footprint of the library. It is important to avoid that such
memory footprint per MPI task exceeds the amount of main memory
available to the compiled code. To help the user to make the correct
decision in terms of resources a simple formula for **Block distribution** of matrix can be used ::

  sizeof(float_type) *[n * m + 2 * (n + m) * block + 1 + 5*block + 2*pow(block,2)]/(1024^3) GigaByte

where ``n`` and ``m`` are fractions of ``N`` which depend on the size
of the MPI grid of precessors. For instance in the job script above
``n = N/nrows`` and ``m = N/ncols``, with the size of MPI grid ``nrows*ncols``. 
Correspondingly ``N`` is
the size of the eigenproblem and ``block`` is at most ``nev + nex``.
Note that the factor ``sizeof(float_type)`` is valid for single precision real,
double precision real, single precision complex and double precision complex floating numbers.
The value of this factor for these four types of floating numbers are respectively:
``4``, ``8``, ``8``, ``16``.

For ChASE with **Block-Cyclic distribution** of matrix, addtional memory of
size ``sizeof(float_type) * N`` is required for managing the internal reshuffing
for block-cyclic data layout. Thus the total memory required is::

  sizeof(float_type) *[n * m + 2 * (n + m) * block + N + 1 + 5*block + 2*pow(block,2)]/(1024^3) GigaByte


We provide a simple python script to estimate the memory requirement of ChASE depending
on the matrix size and available computation ressources: `analyze-mem-requirements.py <https://github.com/ChASE-library/ChASE/blob/master/scripts/analyze-mem-requirements.py>`_

The usage of this script is quite simple. For ChASE with **Block Distribution**:

.. code-block:: console 

    python analyze-mem-requirements.py --n ${n} --nev ${nev} --nex ${nex} --mpi ${nodes}


in which ``${n}`` is the rank of matrix, ``${nev}`` is the number of eigenpairs to be computed,  ``${nex}`` is the external size of searching space, and ``${nodes}`` are the number of MPI ranks to be used. Below is an example of output:

.. code-block:: bash

   Problem size
   -------------------------------
   Matrix size:   360000
   Eigenpairs:    2500
   Extra vectors: 500
   Precision:     double (8 bytes)

   MPI configuration
   -------------------------------
   #MPI ranks:    1152
   MPI grid size: 32 x 36
   Block size:    11250.0 x 10000.0

   Matrix Distribution
   -------------------------------
   Data Layout:   block


   Main memory usage per MPI-rank: 1.989 GB
   Total main memory usage (1152 ranks): 2291.808 GB


Using such a formula one can verify if the allocation of
resources is enough to solve for the problem at hand. For instance,
for a ``N = 360,000`` and a ``nev + nex = 3,000`` with ``1152`` MPI ranks, the total memory per MPI rank is ``1.989 GB``.


For ChASE with **Block-Cylic Distribution**:

.. code-block:: console 

    python analyze-mem-requirements.py --n ${n} --nev ${nev} --nex ${nex} --mpi ${nodes} --nrows ${nrows} --ncols ${ncols} --layout block-cyclic


For the estimation of the memory requirement of ChASE with **Block-Cyclic Distribution**, at least three more arguments by the flags ``--nrows``, ``--ncols`` and ``--layout``. The implementation of ChASE with **Block-Cyclic Distribution** requires users provides explicitly
the required MPI grid size. Moreover, the flag ``--layout`` should also be explicitly set as ``block-cyclic`` to active the mode of **Block-Cyclic Distribution**. Below is an example of output:

.. code-block:: bash

   Problem size
   -------------------------------
   Matrix size:   360000
   Eigenpairs:    2500
   Extra vectors: 500
   Precision:     double (8 bytes)

   MPI configuration
   -------------------------------
   #MPI ranks:    1152
   MPI grid size: 32 x 36
   Block size:    11250.0 x 10000.0

   Matrix Distribution
   -------------------------------
   Data Layout:   block-cyclic


   Main memory usage per MPI-rank: 1.992 GB
   Total main memory usage (1152 ranks): 2294.898 GB


ChASE with multi-GPUs
---------------------

Currently, ChASE is able to offload the most intensive computation (Hermitian Matrix-Matrix 
Multiplications), QR factorization and Rayleigh-Ritz computation to GPUs. 
The multi-GPUs version of ChASE is able to use all available cards for
each node. This multi-GPUs version supports 1 MPI task
to manage only 1 binded GPU card. Some less intensive computation is also assigned to this MPI task and executed
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

As mentiond in the previous section, for ChASE with multi-GPUs, it is important to make sure that
the memory footprint of the library does not exceed the memory
available on the GPU card. For ChASE with multi-GPUs using **Block distribution** of matrix, the 
memory requirement of CPU is::

  sizeof(float_type) *[n * m + 2 * (n + m) * block + 1 + 5*block + 2*pow(block,2)]/(1024^3) GigaByte


And the memory requirement of each GPU is also ::

  sizeof(float_type) *[n * m + 2 * (n + m) * block + 1 + 5*block + 2*pow(block,2)]/(1024^3) GigaByte


It is possible to estimate the memory costs of both CPUs and GPUs for ChASE multi-GPUs by this python script: `analyze-mem-requirements.py <https://github.com/ChASE-library/ChASE/blob/master/scripts/analyze-mem-requirements.py>`_

.. code-block:: console 

    python analyze-mem-requirements.py --n ${n} --nev ${nev} --nex ${nex} --mpi ${nodes} --gpus ${nb_gpus}

It is quite similar to the one for ChASE with pure-CPUs, the only additional required information is ``${nb_gpus}``, which enables the estimate of GPU memory requirement.
Here is an example of output:


.. code-block:: bash

   Problem size
   -------------------------------
   Matrix size:   360000
   Eigenpairs:    2500
   Extra vectors: 500
   Precision:     double (8 bytes)

   MPI configuration
   -------------------------------
   #MPI ranks:    1152
   MPI grid size: 32 x 36
   Block size:    11250.0 x 10000.0

   Matrix Distribution
   -------------------------------
   Data Layout:   block

   GPU configuration per MPI-rank
   -------------------------------
   #GPUs:      1
   GPU grid:   1 x 1
   Block size: 11250.0 x 10000.0


   Main memory usage per MPI-rank: 1.989 GB
   Total main memory usage (1152 ranks): 2291.808 GB

   Memory requirement per GPU: 1.989 GB
   Total GPU memory per MPI-rank (1 GPUs): 1.989 GB


For ChASE with multi-GPUS using **Block-Cyclic Distribution**, the memory requirement of GPU is the same as the one with **Block Distribution**, and the CPUs require addtional memory of
size ``sizeof(float_type) * N * block``. Thus the formule is::

  sizeof(float_type) *[n * m + 2 * (n + m) * block + N + 1 + 5*block + 2*pow(block,2)]/(1024^3) GigaByte


The usage of provided python script is:

.. code-block:: console 

    python analyze-mem-requirements.py --n ${n} --nev ${nev} --nex ${nex} --mpi ${nodes} --nrows ${nrows} --ncols ${ncols} --layout block-cyclic --gpus ${nb_gpus}

Here is an example of output:

.. code-block:: bash

   Problem size
   -------------------------------
   Matrix size:   360000
   Eigenpairs:    2500
   Extra vectors: 500
   Precision:     double (8 bytes)

   MPI configuration
   -------------------------------
   #MPI ranks:    1152
   MPI grid size: 32 x 36
   Block size:    11250.0 x 10000.0

   Matrix Distribution
   -------------------------------
   Data Layout:   block-cyclic

   GPU configuration per MPI-rank
   -------------------------------
   #GPUs:      1
   GPU grid:   1 x 1
   Block size: 11250.0 x 10000.0


   Main memory usage per MPI-rank: 1.992 GB
   Total main memory usage (1152 ranks): 2294.898 GB

   Memory requirement per GPU: 1.989 GB
   Total GPU memory per MPI-rank (1 GPUs): 1.989 GB


  
.. warning::

    The estimation of memory requirement by `analyze-mem-requirements.py <https://github.com/ChASE-library/ChASE/blob/master/scripts/analyze-mem-requirements.py>`_ is only based on the algorithmic aspects of ChASE. The buffer and memory requirement of libraries such as ``MPI`` has not been considered. So despite the python script calculation of memory consumption, some combination of MPI libraries (e.g., ParastationMPI) could lead to the crash of ChASE with ``out of memory`` even if the memory available is within the estimated bounds. 



