.. _parameters_and_config:

Parameters and Configurations
************************************


General Parameters
===================

.. table::
   :widths: 8,18,18,10
   :name: General Parameters

   ===================== ========================= ======================== ==================
   Parameter             Description               Suggested value          Default value
   ===================== ========================= ======================== ==================
   :cpp:var:`N_`         Rank of the matrix                                 N/A         
                         :math:`A`\ .                                                         
   :cpp:var:`nev_`       Number of desired         No more than 20%          N/A         
                         eigenpairs                of the total number of                      
                                                   eigenvalues                                
               	                                   (or rank of :math:`A`)                      
   :cpp:var:`nex_`       Search space increment    In most cases 20% of      N/A
                         such that the overall     the value of                               
		         size of the search space  :cpp:var:`nev_`\ . Best                    
		         is :cpp:var:`nev_` +      case scenario for                            
		         :cpp:var:`nex_`\ .        optimal convergence                           
			   		           is when there is a                           
					  	   large                                        
					           spectrum gap between                        
					           :cpp:var:`nev_` and                
					           :cpp:var:`nex_`\ .                           
   :cpp:var:`approx_`    An optional flag          When :cpp:var:`approx_`   ``false`` 
	                 indicating whether the    is set to ``true``,      (black-box solver)
		         user provides ChASE with  ChASE expects to receive                       
		         approximate eigenvectors  in input two arrays                           
		         or uses ChASE in          holding   
		         isolation as a            approximate vectors and 
		         traditional black-box     values, respectively
		         solver                                              
   :cpp:var:`tol_`       An optional parameter     As a rule of thumb a      ``1e-10`` in          
                         indicating the minimal    minimum value of          DP. ``1e-05``       
		         value for the residual    ``1e-08`` and ``1e-04``   in SP.
		         such that the             are suggested for DP
		         corresponding eigenpair   and SP,
		         is declared converged.    respectively.
   :cpp:var:`max_iter_`  An optional parameter set Typically ChASE does not  ``25``
                         as to avoid that ChASE    need more than 10 
		         internal while loop runs  iterations even in the 
		         unchecked in the rare     most complex cases and
		         cases where convergence   averages 4-5 iterations
		         of the desired eigenpairs to convergence. 
		         cannot be secured. Once
		         reached the
		         :cpp:var:`max_iter_`,
	                 ChASE stops execution and
		         returns.
   ===================== ========================= ======================== ==================

   



Chebyshev Filter
================

.. table::
   :widths: 8,18,18,10
   :name: Chebyshev Parameters

   ======================== =================================== ======================== ==================
   Parameter                Description                         Suggested value          Default value
   ======================== =================================== ======================== ==================
   :cpp:var:`deg_`          Polynomial degree for the           When                     ``20`` in DP, 
   	                    Chebyshev filter. When              :cpp:var:`optimization_` ``10`` in SP
		            the value of                        = ``true``, it is 
		            :cpp:var:`optimization_`            advisable to set                
		            = ``true``,                         to a value not larger 
		            this is the `initial`               than ``10``. If 
		            polynomial degree used              :cpp:var:`optimization_` 
		            in only the first                   = ``false``, then it is 
		            subspace iteration.                 advised to select a 
		            Otherwise the same value            value not smaller than 
		            :cpp:var:`deg_` is used 	        ``15`` but not larger 
		            for every vector	                than ``30``.
		            for each filter call. 
   :cpp:var:`optimization_` An optional flag                    Despite the fact that     ``false``     
                            indicating that the                 the default value is
                            filter uses a vector of             set to ``false``, it is
                            degrees optimized for               advisable to set this 
		            each single filtered                flag to ``true`` often
		            vector.                             as possible in order to
		                                                avoid wasting extra
						                FLOPs.
   :cpp:var:`deg_extra_`    A small value used only             Usually a small number    ``2``
                            in combination with                 never larger than ``5``.
		            :cpp:var:`optimization_`            Apart for rare cases,
                            = ``true``.                         avoid changing the
						                default value.
   :cpp:var:`max_deg_`      A parameter which avoids            This value is a result    ``36`` in DP,
	                    that vectors with a                 of practice and           ``18`` in SP. 
		            rather small convergence            experience. We suggest
		            ratio get overfiltered              to avoid setting it
		            entering in a regime of             below ``30`` and be
		            numerical instability.              quite careful to set it                        
		                                                too high (> ``50``). 
   ======================== =================================== ======================== ==================

   

Lanczos DoS (Spectral Estimator)
================================

.. table::
   :widths: 8,18,18,10
   :name: Lanczos DoS Parameters

   ======================== =================================== ======================== ==================
   Parameter                Description                         Suggested value          Default value
   ======================== =================================== ======================== ==================
   :cpp:var:`lanczos_iter_` In order to estimate the spectral   ChASE does not need very ``25``
	                    bounds, ChASE executes a limited    precise spectral 
			    number of Lanczos steps. These      estimates because at
			    steps are then used to compute an   each iteration such
			    estimate of :math:`\lambda_1`,      estimates are
			    :math:`\lambda_{nev+nex}`, and      automatically
			    :math:`\lambda_N`  based on the     improved by the
			    Density of State (DoS) algorithm.   approximate spectrum
			                                        computed. For the DoS
				                                algorithm to work
								effectively without
								overburdening the
                                                                eigensolver, the number
                                                                of Lanczos iteration
                                                                should be not less than
                                                                ``10`` but also no
								more than ``100``.
    :cpp:var:`num_lanczos_` After having executed a number      Because ChASE does not     ``4`` 
	                    of Lanczos steps, ChASE uses a      need precise estimates
			    cheap and efficient estimator       of the upper extremum
			    to calculate the value of the       of the search space,
			    upper extremum of the search space. the number of vectors
			    Such an estimator uses a small      used is quite small.
			    number of stochastic vectors        The expert user can
			    indicated by the variable           can change the value 
			    :cpp:var:`num_lanczos_`.            to a larger number than
                                                                the default value (it
                                                                is not suggested to 
                                                                use a smaller value) 
                                                                and pay a higher
                                                                computing cost. It is
                                                                suggested to not
								set a value for
								:cpp:var:`num_lanczos_`
								higher than ``20``.     
   ======================== =================================== ======================== ==================

..
  Parallel configuration
  ======================

..
  todo:: MPI and num_threads optimal values (EDO)
..
   * ``MPI_ranks`` Default: num_proc_nodes
   * ``num_threads`` Default: num_cores_node
