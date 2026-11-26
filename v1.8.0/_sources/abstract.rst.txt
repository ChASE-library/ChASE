.. _virtual_abstrac_numerical_kernels:

**************************************
Virtual Abstract of Numerical Kernels
**************************************

ChASE relies on a modest number of numerical kernels whose calls execute almost all FLOPs.
Because of its simple structure, there would be clear advantages in the separation of 
the numerical kernels from the algorithm implementation. In ChASE, the linear algebra 
kernels are separated from the main algorithm through the use of an object-oriented software interface.

Benefits
---------

With the help of the seperation of numerical kernels,

1. First, ChASE can be easily integrated into existing codes thanks to the relative simplicity of the software interface. For instance, the low-level kernels can be implemented according to an existing distribution of the matrix elements of A so as to avoid the need to re-distribute data;

2. Second, ChASE can easily exploit existing linear algebra libraries such as BLAS and LAPACK all the way up to GPU-based kernels, and even complex distributed-memory dense linear algebra frameworks such as Elemental.

Definition of Interface
------------------------

For the complete API documentation of ``chase::ChaseBase``, please refer to :ref:`api_chasebase`.

