.. _para-chase-mpi-gpu:

multi-GPUs in node
----------------------

Implementation of ChaseMpiDLAInterface for multi-GPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: chase::mpi::ChaseMpiDLAMultiGPU
   :project: ChASE
   :members: ChaseMpiDLABlaslapack,shiftMatrix, preApplication,apply,postApplication,applyVec,lange,gegqr,axpy,scal,nrm2,dot,gemm_small,gemm_large,stemr,RR_kernel


Implementation of Selected DLA with multi-GPUs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: chase::mpi::mgpu_cudaDLA
   :project: ChASE
   :members: mgpu_cudaDLA, distribute_H,distribute_V,computeHemm,return_W,synchronizeAll,gegqr,RR_kernel
