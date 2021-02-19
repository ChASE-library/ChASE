Classes
-----------

chase::Chase
^^^^^^^^^^^^^

This class defines the `virtual kernels` as the virtual functions
defined in the abstract class that correspond to calls to numerical
kernels required by ChASE. It includes the
following functionalities:

  * HEMM: Hermitian Matrix-Matrix Multiplication

  * QR: QR factorization

  * RR: Rayleigh-Ritz projection and small problem solver

  * Resd: compute the eigenpair residuals

  * Lanczos: estimate the bounds of user-interested spectrum by Lanczos eigensolver

  * LanczosDos: estimate the spectral distribution of eigenvalues

  * Swap: swap the two matrices of vectors used in the Chebyschev filter

  * Locking: locking the converged eigenpairs

  * Shift: shift the diagonal of the A matrix used in the 3-terms
    recurrence relation implemented in the Chebyschev filter

  * etc ..

.. note::

   For more details on the virtual kernels, 
   please refer to :ref:`virtual_abstrac_numerical_kernels`. 
   Different parallel implementations of these virtual kernels
   can also be found :ref:`parallel_implementations`.

chase::Algorithm
^^^^^^^^^^^^^^^^^

This class defines algoritmic implementation of ChASE using the
defined virtual kernels. It includes the
functionalties:

  * Chebyshev filter

  * calculation of degree of the filter

  * Lanczos solver to estimate the bound of spectra

  * locking the converged Ritz pairs

  * etc ..

.. note::

  The details of this class are only provided in the developer documentation,
  please refer to :ref:`algorithmic-structure`.

chase::Base
^^^^^^^^^^^^

This class defines ...

chase::ChaseConfig
^^^^^^^^^^^^^^^^^^^

This class defines the functions to set different parameters of ChASE.

.. note::

  For more details of all available functions, please refer to
  :ref:`configuration_object`.


chase::ChasePerfData
^^^^^^^^^^^^^^^^^^^^^

This class defines the performance data for different algorithm and numerical
kernels of ChASE, e.g., the floating operations of ChASE for given size of matrix
and a required number of eigenpairs to be computed.

.. note::

  For more details of all available functions, please refer to
  :ref:`performance`.

chase::PerformanceDecoratorChase
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This class is implemented based on ``chase::ChasePerfData`` which decorate different
numerical operations in ChASE, to mesure their performance (cost time, flop/s, etc).

.. note::

  For more details of all available functions, please refer to
  :ref:`performance`.


Functions
-----------

chase::Solve
^^^^^^^^^^^^^

This function provides the implementation of ChASE algorithm by assembling the algorithms and 
numerical kernels implemented in ``chase::Chase`` and ``chase::Algorithm``.

 
