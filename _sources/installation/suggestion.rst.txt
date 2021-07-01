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

  sizeof(float_type) *[n * m + (n + m + max(m,n)) * block + 2 * N *
  block + 1 + 5*n + 2*pow(n,2) + 2 * nb]/(1024^3) GigaByte

where ``n`` and ``m`` are fractions of ``N`` which depend on the size
of the MPI grid of precessors. For instance in the job script above
``n = N/nrows`` and ``m = N/ncols``, with the size of MPI grid ``nrows*ncols``. 
Correspondingly ``N`` is
the size of the eigenproblem and ``block`` is at most ``nev + nex``.
Note that the factor ``sizeof(float_type)`` is valid for single precision real,
double precision real, single precision complex and double precision complex floating numbers.
The value of this factor for these four types of floating numbers are respectively:
``4``, ``8``, ``8``, ``16``.
``nb`` is the algorithmic block size used by the implementation of QR.

For ChASE with **Block-Cyclic distribution** of matrix, addtional memory of
size ``sizeof(float_type) * N * block`` is required for managing the internal reshuffing
for block-cyclic data layout. Thus the total memory required is::

  sizeof(float_type) *[n * m + (n + m + max(m,n)) * block + 3 * N *
  block + 1 + 5*n + 2*pow(n,2) + 2 * nb]/(1024^3) GigaByte


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


    Main memory usage per MPI-rank: 17.798 GB
    Total main memory usage (1824 ranks): 20503.089 GB


Using such a formula one can verify if the allocation of
resources is enough to solve for the problem at hand. For instance,
for a ``N = 360,000`` and a ``nev + nex = 3,000`` with ``1152`` MPI ranks, the total memory per MPI rank is ``17.354 GB``.


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


    Main memory usage per MPI-rank: 25.844 GB
    Total main memory usage (1152 ranks): 29772.803 GB


ChASE with multi-GPUs
---------------------

Currently, ChASE is able to offload the most intensive computation (Hermitian Matrix-Matrix 
Multiplications), QR factorization and Rayleigh-Ritz computation to GPUs. 
The multi-GPUs version of ChASE is able to use all available cards for
each node. This multi-GPUs version supports either 1 MPI task to manage all cards or 1 MPI task
to manage only 1 binded GPU card. Some less intensive computation is also assigned to this MPI task and executed
in multi-threading mode.

Allocating Ressources and Running jobs (SLURM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below is an example of a job script for a ``SLURM`` scheduler which allocates 1 MPI task with 
multi-GPUs per node:

.. code-block:: bash

    # This is an example on the JUWELS GPU partition, in which each node has 4 V100 NVIDIA GPUs.
    # This example allocates 4 nodes, 4 MPI tasks, each node has 1 task,
    # and 4 GPUs per node.
    #!/bin/bash -x
    #SBATCH --nodes=4
    #SBATCH --ntasks=4
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=24
    #SBATCH --gres=gpu:4

    export CUDA_VISIBLE_DEVICES=0,1,2,3

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

    export CUDA_VISIBLE_DEVICES=0,1,2,3


.. note::
    For the usage of ChASE with multi-GPUs, the environment variable ``CUDA_VISIBLE_DEVICES`` should also be set before the execution of ChASE, which indicates explicity the available
    GPU devices per computing node. More the number of available GPU/node should be always equal to or
    larger than the allocated MPI ranks per node. 


Estimating Memory Requirement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As mentiond in the previous section, for ChASE with multi-GPUs, it is important to make sure that
the memory footprint of the library does not exceed the memory
available on the GPU card. For ChASE with multi-GPUs using **Block distribution** of matrix, the 
memory requirement of CPU is::

  sizeof(float_type) *[n * m + (n + m + max(m,n)) * block + 2 * N *
  block + 1 + 5*n + 2*pow(n,2)]/(1024^3) GigaByte

And the memory requirement of each GPU is::

  sizeof(float_type) *[3 * block * max(gpu_m, gpu_n) + gpu_m * gpu_n + (2 * N + block) * block]/(1024^3) GigaByte

In the formule related to GPU, new introduced parameters ``gpu_m`` and ``gpu_n`` are fractions of ``m`` and ``n`` which depend on the size of grid of GPUs per MPI rank. More precisely, ``gpu_m=m/gpu_col`` and ``gpu_n=n/gpu_row``, in which the grid of GPUs per MPI rank is ``gpu_row * gpu_col``. In ChASE, ``gpu_row`` and ``gpu_col`` are automatically computed by considering the available number of GPUs per MPI rank. 

It is possible to estimate the memory costs of both CPUs and GPUs for ChASE multi-GPUs by this python script: `analyze-mem-requirements.py <https://github.com/ChASE-library/ChASE/blob/master/scripts/analyze-mem-requirements.py>`_

.. code-block:: console 

    python analyze-mem-requirements.py --n ${n} --nev ${nev} --nex ${nex} --mpi ${nodes} --gpus ${nb_gpus}

It is quite similar to the one for ChASE with pure-CPUs, the only additional required information is ``${nb_gpus}``, which indicates the number of GPUs used per MPI rank.
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
    #GPUs:      4
    GPU grid:   2 x 2
    Block size: 5625.0 x 5000.0


    Main memory usage per MPI-rank: 17.792 GB
    Total main memory usage (1152 ranks): 20496.497 GB

    Memory requirement per GPU: 16.747 GB
    Total GPU memory per MPI-rank (4 GPUs): 66.988 GB 


For ChASE with multi-GPUS using **Block-Cyclic Distribution**, the memory requirement of GPU is the same as the one with **Block Distribution**, and the CPUs require addtional memory of
size ``sizeof(float_type) * N * block``. Thus the formule is::

  sizeof(float_type) *[n * m + (n + m + max(m,n)) * block + 3 * N *
  block + 1 + 5*n + 2*pow(n,2)]/(1024^3) GigaByte

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
    #GPUs:      4
    GPU grid:   2 x 2
    Block size: 5625.0 x 5000.0


    Main memory usage per MPI-rank: 25.839 GB
    Total main memory usage (1152 ranks): 29766.212 GB

    Memory requirement per GPU: 16.747 GB
    Total GPU memory per MPI-rank (4 GPUs): 66.988 GB

  
.. warning::

    The estimation of memory requirement by `analyze-mem-requirements.py <https://github.com/ChASE-library/ChASE/blob/master/scripts/analyze-mem-requirements.py>`_ is only based on the algorithmic aspects of ChASE. The buffer and memory requirement of libraries such as ``MPI`` has not been considered. So despite the python script calculation of memory consumption, some combination of MPI libraries (e.g., ParastationMPI) could lead to the crash of ChASE with ``out of memory`` even if the memory available is within the estimated bounds. 