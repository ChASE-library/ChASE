Parameters and Configurations
************************************


General Parameters
===================

.. table::
   :widths: 8,18,18,10
   :name: General Parameters

   ================== ========================= ======================== ==============
   Parameter          Description               Suggested value          Default value
   ================== ========================= ======================== ==============
   :cpp:var:`N`       Rank of the matrix                                 N/A         
                      :math:`A`\ .                                                         
   :cpp:var:`nev`     Number of desired         No more than 20%          N/A         
                      eigenpairs                of the total number of                      
                                                eigenvalues                                
               	                                (or rank of :math:`A`)                      
   :cpp:var:`nex`     Search space increment    In most cases 20% of      N/A
                      such that the overall     the value of                               
		      size of the search space  :cpp:var:`nev`\ . Best                    
		      is :cpp:var:`nev` +       case scenario for                            
		      :cpp:var:`nex`\ .         optimal convergence                           
					        is when there is a                           
						large                                        
					        spectrum gap between                        
					        :cpp:var:`nev` and                
					        :cpp:var:`nex`\ .                           
   :cpp:var:`approx`  An optional flag          When :cpp:var:`approx`   ``false`` 
	              indicating whether the    is set to true, ChASE    (black-box solver)
		      user provides ChASE with  expects the arrays                          
		      approximate eigenvectors  :cpp:var:`V` and                           
		      or uses ChASE in          :cpp:var:`ritzv` to hold   
		      isolation as a            approximate vectors and 
		      traditional black-box     values, respectively
		      solver                                              
   :cpp:var:`tol`     An optional parameter     As a rule of thumb a      ``1e-10`` in            
                      indicating the minimal    minimum value of          DP. ``1e-05``              
		      value for the residual    ``1e-08`` and ``1e-04``   in SP.
		      such that the             are suggested for DP
		      corresponding eigenpair   and SP,
		      is declared converged.    respectively.
   :cpp:var:`maxIter` An optional parameter set Typically ChASE does not  ``25``
                      as to avoid that ChASE    need more than 10 
		      internal while loop runs  iterations even in the 
		      unchecked in the rare     most complex cases and
		      cases where convergence   averages 4-5 iterations
		      of the desired eigenpairs to convergence. 
		      cannot be secured. Once
		      reached the
		      :cpp:var:`maxIter`, ChASE
	              stops execution and
		      returns.
   ================== ========================= ======================== ==============

   



Chebyshev Filter
================

.. table::
   :widths: 8,18,18,10
   :name: Chebyshev Parameters

   =================== ========================= ======================== ==================
   Parameter           Description               Suggested value          Default value
   =================== ========================= ======================== ==================
   :cpp:var:`deg`      Polynomial degree for the When                     ``20`` in DP, 
   	               Chebyshev filter. When    :cpp:var:`optim` = true  ``10`` in SP.
		       the value of              it is advisable to set
		       :cpp:var:`optim` = true   to a value not larger 
		       , this is the `initial`   than ``10``. If 
		       polynomial degree used    :cpp:var:`optim` = false
		       in only the first         then it is advised to
		       subspace iteration.       select a value not
		       Otherwise the same value  smaller than ``15``
		       :cpp:var:`deg` is used 	 but not larger than
		       for every vector	         ``30``.
		       for each filter call. 
   :cpp:var:`optim`    An optional flag          Despite the fact that     ``false``     
                       indicating that the       the default value is
                       filter uses a vector of   set to ``false``, it is
                       degrees optimized for     advisable to set this 
		       each single filtered      flag to ``true`` often
		       vector.                   as possible in order to
		                                 avoid wasting extra
						 FLOPs.
   :cpp:var:`degExtra` A small value used only   Usually a small number    ``2``
                       in combination with       never larger than ``5``.
		       :cpp:var:`optim` = true.  Apart for rare cases,
                                                 avoind changing the
						 default value.
   :cpp:var:`degMax`   A parameter which avoids  This value is a result    ``36`` in DP,
	               that vectors with a       of practice and           ``18`` in SP. 
		       rather small convergence  experience. We suggest
		       ratio get overfiltered    to avoid setting it
		       entering in a regime of   below ``30`` and be
		       numerical instability.    quite careful to set it                        
		                                 too high (> ``50``). 
   =================== ========================= ======================== ==================

   
..
   Spectral Estimator
   ==================
   
   ..
     todo:: Lanczos DoS Parameters explanation and default values
     (EDO)
..     
   * ``k`` Default: 25
   * ``n_vec`` Default: 4

..
  Parallel configuration
  ======================

..
  todo:: MPI and num_threads optimal values (EDO)
..
   * ``MPI_ranks`` Default: num_proc_nodes
   * ``num_threads`` Default: num_cores_node
