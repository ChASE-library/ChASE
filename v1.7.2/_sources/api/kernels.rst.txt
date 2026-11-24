.. _api_kernels:

Linear Algebra Kernels
======================

chase::linalg::internal::cpu
-----------------------------

.. doxygennamespace:: chase::linalg::internal::cpu
   :project: ChASE
   :members:
   :undoc-members:

chase::linalg::internal::cuda
------------------------------

.. doxygennamespace:: chase::linalg::internal::cuda
   :project: ChASE
   :members:
   :undoc-members:

chase::linalg::internal::mpi
----------------------------

.. note::

   The MPI kernel namespace contains functions for distributed CPU operations.
   These are internal implementation details. For user-facing documentation,
   refer to the implementation classes (:ref:`api_implementations`).

chase::linalg::internal::nccl
------------------------------

.. note::

   The NCCL kernel namespace contains functions for distributed GPU operations
   using NCCL. These are internal implementation details. For user-facing
   documentation, refer to the implementation classes (:ref:`api_implementations`).

chase::linalg::internal::cuda_aware_mpi
----------------------------------------

.. note::

   The CUDA-aware MPI kernel namespace contains functions for GPU operations
   with CUDA-aware MPI. These are internal implementation details. For
   user-facing documentation, refer to the implementation classes (:ref:`api_implementations`).

