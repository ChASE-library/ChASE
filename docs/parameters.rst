.. _parameters_and_config:

Parameters and Configurations
************************************


General Parameters
===================

.. list-table:: General Parameters
   :widths: 8 18 18 10
   :header-rows: 1

   * - Parameter
     - Description
     - Suggested value
     - Default value
   * - :cpp:var:`N_`
     - Rank of the matrix :math:`A`\ .
     - 
     - N/A
   * - :cpp:var:`nev_`
     - Number of desired eigenpairs
     - No more than 20% of the total number of eigenvalues (or rank of :math:`A`)
     - N/A
   * - :cpp:var:`nex_`
     - Search space increment such that the overall size of the search space is :cpp:var:`nev_` + :cpp:var:`nex_`\ .
     - In most cases 20% of the value of :cpp:var:`nev_`\ . Best case scenario for optimal convergence is when there is a large spectrum gap between :cpp:var:`nev_` and :cpp:var:`nex_`\ .
     - N/A
   * - :cpp:var:`approx_`
     - An optional flag indicating whether the user provides ChASE with approximate eigenvectors or uses ChASE in isolation as a traditional black-box solver
     - When :cpp:var:`approx_` is set to ``true``, ChASE expects to receive in input two arrays holding approximate vectors and values, respectively
     - ``false`` (black-box solver)
   * - :cpp:var:`tol_`
     - An optional parameter indicating the minimal value for the residual such that the corresponding eigenpair is declared converged.
     - As a rule of thumb a minimum value of ``1e-08`` and ``1e-04`` are suggested for DP and SP, respectively. The tolerance should hardly be set below ``1e-14`` in double precision due to accuracy limits of the dense eigensolver.
     - ``1e-10`` in DP. ``1e-5`` in SP.
   * - :cpp:var:`max_iter_`
     - An optional parameter set as to avoid that ChASE internal while loop runs unchecked in the rare cases where convergence of the desired eigenpairs cannot be secured. Once reached the :cpp:var:`max_iter_`, ChASE stops execution and returns.
     - Typically ChASE does not need more than 10 iterations even in the most complex cases and averages 4-5 iterations to convergence.
     - ``25``



Chebyshev Filter
================

.. list-table:: Chebyshev Parameters
   :widths: 8 18 18 10
   :header-rows: 1

   * - Parameter
     - Description
     - Suggested value
     - Default value
   * - :cpp:var:`deg_`
     - Polynomial degree for the Chebyshev filter. When the value of :cpp:var:`optimization_` = ``true``, this is the `initial` polynomial degree used in only the first subspace iteration. Otherwise the same value :cpp:var:`deg_` is used for every vector for each filter call.
     - When :cpp:var:`optimization_` = ``true``, it is advisable to set to a value not larger than ``10``. If :cpp:var:`optimization_` = ``false``, then it is advised to select a value between ``10`` and ``25``. It is strongly suggested to avoid values above ``40`` or the value returned by :cpp:var:`max_deg_`. If an odd value is provided, it will be made even.
     - ``20`` in DP, ``10`` in SP
   * - :cpp:var:`optimization_`
     - An optional flag indicating that the filter uses a vector of degrees optimized for each single filtered vector.
     - Despite the fact that the default value is set to ``true``, it is advisable to keep this flag enabled in order to avoid wasting extra FLOPs. Set to ``false`` only if you want to use a fixed degree for all vectors.
     - ``true``
   * - :cpp:var:`deg_extra_`
     - A small value used only in combination with :cpp:var:`optimization_` = ``true``.
     - Usually a small number never larger than ``5``. Apart for rare cases, avoid changing the default value.
     - ``2``
   * - :cpp:var:`max_deg_`
     - A parameter which avoids that vectors with a rather small convergence ratio get overfiltered entering in a regime of numerical instability.
     - This value is a result of practice and experience. It can be lowered in case of early instabilities but should not be lower than ``20-25`` to avoid the filter becomes ineffective. It can be increased when there is a spectral gap between :cpp:var:`nev_` and :cpp:var:`nev_` + :cpp:var:`nex_`. It is strongly suggested to never exceed ``70``. If an odd value is provided, it will be made even.
     - ``36`` in DP, ``18`` in SP

   

Lanczos DoS (Spectral Estimator)
================================

.. list-table:: Lanczos DoS Parameters
   :widths: 8 18 18 10
   :header-rows: 1

   * - Parameter
     - Description
     - Suggested value
     - Default value
   * - :cpp:var:`lanczos_iter_`
     - In order to estimate the spectral bounds, ChASE executes a limited number of Lanczos steps. These steps are then used to compute an estimate of :math:`\lambda_1`, :math:`\lambda_{nev+nex}`, and :math:`\lambda_N`  based on the Density of State (DoS) algorithm.
     - ChASE does not need very precise spectral estimates because at each iteration such estimates are automatically improved by the approximate spectrum computed. For the DoS algorithm to work effectively without overburdening the eigensolver, the number of Lanczos iteration should be not less than ``10`` but also no more than ``100``.
     - ``25``
   * - :cpp:var:`num_lanczos_`
     - After having executed a number of Lanczos steps, ChASE uses a cheap and efficient estimator to calculate the value of the upper extremum of the search space. Such an estimator uses a small number of stochastic vectors indicated by the variable :cpp:var:`num_lanczos_`.
     - Because ChASE does not need precise estimates of the upper extremum of the search space, the number of vectors used is quite small. The expert user can change the value to a larger number than the default value (it is not suggested to use a smaller value) and pay a higher computing cost. It is suggested to not set a value for :cpp:var:`num_lanczos_` higher than ``20``.
     - ``4``

Additional Parameters
=====================

.. list-table:: Additional Parameters
   :widths: 8 18 18 10
   :header-rows: 1

   * - Parameter
     - Description
     - Suggested value
     - Default value
   * - :cpp:var:`cholqr_`
     - An optional flag indicating whether to use flexible CholeskyQR or Householder QR for orthonormalization.
     - Use flexible CholQR (``true``) for better performance in most cases. Use Householder QR (``false``) if numerical stability is a concern.
     - ``true``
   * - :cpp:var:`sym_check_`
     - An optional flag indicating whether to enable symmetry checking to verify matrix properties during computation.
     - Enable symmetry checking (``true``) to verify matrix properties during computation.
     - ``true``
   * - :cpp:var:`decaying_rate_`
     - An optional parameter controlling the decaying rate for the polynomial lower bound. The lower bound of the Chebyshev polynomial is set based on an approximation of eigenvalues by few iterations of Lanczos.
     - Usually keep the default value of ``1.0`` (no decaying). Use a value less than ``1.0`` to underestimate the lower bound if target eigenvalues are packed.
     - ``1.0``
   * - :cpp:var:`upperb_scale_rate_`
     - An optional parameter controlling how the upper bound is scaled based on its sign. For positive upper bound, it's multiplied by this rate. For negative upper bound, it's multiplied by (2 - rate).
     - Default value ``1.0`` works well for most cases. For positive upper bound, it's multiplied by this rate. For negative upper bound, it's multiplied by (2 - rate).
     - ``1.0``
   * - :cpp:var:`cluster_aware_degrees_`
     - An optional flag indicating whether to use cluster-aware degree optimization. When enabled, the algorithm detects clusters of eigenvalues and adjusts polynomial degrees accordingly.
     - Enable cluster-aware degree optimization (``true``) to improve convergence for clustered eigenvalues. Disable (``false``) only if you experience performance issues.
     - ``true``

.. _configuration_object:

Configuration Object
====================

The ``chase::ChaseConfig<T>`` class provides the API for configuring all ChASE
parameters programmatically. This class is accessed through the ``GetConfig()``
method of any ChASE solver instance.

.. note::

   For detailed API documentation of the ``chase::ChaseConfig<T>`` class, including
   all available methods and their descriptions, please refer to the Doxygen-generated
   documentation in the build directory or visit the online documentation.

..
  Parallel configuration
  ======================

..
  todo:: MPI and num_threads optimal values (EDO)
..
   * ``MPI_ranks`` Default: num_proc_nodes
   * ``num_threads`` Default: num_cores_node
