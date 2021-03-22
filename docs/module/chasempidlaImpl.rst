
ChaseMpiDLA
^^^^^^^^^^^^^

``ChaseMPIDLA`` class implements the MPI collective communications part of ``ChaseMpiDLAInterface`` when the MPI is supported.

The computation intra-node, either with CPUs or GPUs, are implemented within classes ``ChaseMpiDLABlaslapack`` and ``ChaseMpiDLAMultiGPU``, respectively.


.. note::
    For more details of the implementation of ChASE-MPI, please refer to :ref:`ara-chase-inter-node`
    in the Developer Documentation.

