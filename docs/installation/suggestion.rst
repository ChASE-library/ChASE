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

ChASE with multi-GPUs
---------------------

Currently, ChASE is able to offload the most intensive computation (Hermitian Matrix-Matrix 
Multiplications), QR factorization and Rayleigh-Ritz computation to GPUs. 
The multi-GPUs version of ChASE is able to use all available cards for
each node. This multi-GPUs version supports either 1 MPI task to manage all cards or 1 MPI task
to manage only 1 binded GPU card. Some less intensive computation is also assigned to this MPI task and executed
in multi-threading mode.

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
