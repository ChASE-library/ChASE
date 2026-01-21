The :cpp:func:`chase` routine
*****************************

The chase.cpp routine is the main routine of the ChASE
library and includes several functions whose scope reflects the
algorithm description provided in the :ref:`algorithmic-structure`
section. The signature of the :cpp:func:`chase` function comprise variables
that appear in other functions inside chase.cpp as well as in other
routines.

Function definitions
~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: void chase( MKL_Complex16* H, int N, MKL_Complex16* V, MKL_Complex16* W, double* ritzv, int nev, const int nex, const int deg, int* const degrees, const double tol, const CHASE_MODE_TYPE mode, const CHASE_OPT_TYPE opt )

.. cpp:function:: void ColSwap( MKL_Complex16 *V, int N, int i, int j )

.. cpp:function:: int calc_degrees( int N, int unconverged, int core, double upperb, double lowerb, double tol, double *ritzv, double *resid, int *degrees, MKL_Complex16 *V, MKL_Complex16 *W )

.. cpp:function:: int locking( int N, int unconverged, double tol, double *ritzv, double *resid, int *degrees, MKL_Complex16 *V )

.. cpp:function:: void calc_residuals( int N, int unconverged, double tol, double *ritzv, double *resid, MKL_Complex16 *H, MKL_Complex16 *V, MKL_Complex16 *W )

.. cpp:function:: void QR( int N, int nevex, int converged, MKL_Complex16 *W, MKL_Complex16 *tau, MKL_Complex16 *saveW )

.. cpp:function:: void RR( int N, int block, double *Lambda, MKL_Complex16 *H, MKL_Complex16 *V, MKL_Complex16 *W )

		  

Type definitions
~~~~~~~~~~~~~~~~

.. cpp:type:: MKL_Complex16

	      Defined by a preprocessing directive as
	      ``std::complex<double>`` type in case ChASE does not
	      make use of the MKL library, but can use MKL's BLAS and LAPACK.

.. cpp:type:: CHASE_MODE_TYPE

	      Defined as type ``char`` via a preprocessing directive and
	      used to define the type of the values of the
	      :cpp:var:`mode` variable.

.. cpp:type:: CHASE_OPT_TYPE

	      Defined as type ``char`` via a preprocessing directive and
	      used to define the type of the values of the
	      :cpp:var:`opt` variable.


Variable definitions
~~~~~~~~~~~~~~~~~~~~

.. cpp:var:: MKL_Complex16 * H
	     
	     Array of size :cpp:var:`N` :math:`\times`
	     :cpp:var:`N`. It stores the input hermitian matrix.
	     **TODO** If uplo = UPPER/LOWER only the leading
	     upper/lower triangular part of H stores the entries of
	     the upper/lower entries of the matrix H while the
	     strictly lower/upper triangular part of H is not
	     referenced.

.. cpp:var:: int N

	     Number of rows and columns of Matrix :cpp:var:`H`.
	     
.. cpp:var:: MKL_Complex16 * V

	     Array containing a collection of vectors, of size (:cpp:var:`N`,
	     :cpp:var:`nev` + :cpp:var:`nex`).  On entry, if :cpp:var:`mode` =
	     ``A``, it is used to provide approximate eigenvectors which work as
	     a preconditioner, speeding up the convergence of required
	     eigenpairs.  On exit, the leading (:cpp:var:`N`, :cpp:var:`nev`)
	     block contains eigenvectors at least as accurate as the required
	     :cpp:var:`tol`.

.. cpp:var:: MKL_Complex16 * W
	     
	     Array containing a collection of auxiliary vectors used as working
	     space throughout the code, of size (:cpp:var:`N`, :cpp:var:`nev` +
	     :cpp:var:`nex`).
     
.. cpp:var:: double * ritz

	     Array of size :cpp:var:`nev` + :cpp:var:`nex`.  It
	     contains and ordered collection of real numbers. On
	     entry, if mode = ``A``, it contains approximations to the
	     lowest nev+nex eigenvalues of H.  On exit the leading
	     :cpp:var:`nev` block contains the smallest nev
	     eigenvalues of :cpp:var:`H`, in descending order. The
	     remaining :cpp:var:`nex` entries are an approximation to
	     the next :cpp:var:`nex` eigenvalues.

.. cpp:var:: int nev

	     Specifies the number of required eigenvalues. This is usually
	     a fraction of the total eigenspectrum. The maximum value of such
	     fraction dependes on the size of the eigenproblem but as a rule
	     of thumb should not exceed 10-20% in order to use ChASE
	     efficiently.

.. cpp:var:: int nex

	     It specifies the initial size of the search subspace *S*
	     such that size(*S*) = (:cpp:var:`nev` +
	     :cpp:var:`nex`). Its optimal choice represents a
	     trade-off between extra computations and an enhanced
	     eigenpairs convergence ratio. Ideally it should be a
	     fraction of :cpp:var:`nev` guaranteeing to include a
	     spectral gap between the first :cpp:var:`nev` and the
	     next :cpp:var:`nev` + :cpp:var:`nex`
	     eigenvalues. Example: for :cpp:var:`nev` = 250, a value
	     of :cpp:var:`nex` = 50 should suffice unless eigenvalues
	     between index 250 and 300 are quite clustered.

.. cpp:var:: const int deg

	     If :cpp:var:`opt` = ``N``, it specifies the constant
	     Chebyshev polynomial degree used in the filter to enhance
	     the convergence ration of sought after
	     eigenvectors. Suggested value: :cpp:var:`deg` = 25.  If
	     :cpp:var:`opt` = ``S``, it specifies the Chebyshev
	     polynomial degree used in the first call to the
	     filter. Following the first call the optimal polynomial
	     degree is computed on the fly. Suggested value:
	     :cpp:var:`deg` = 10.
     
.. cpp:var:: int* const degrees

	     Array of size :cpp:var:`nev` + :cpp:var:`nex`. While a required
	     input array, the contents are ignored.  This variable is planned
	     for cases where ChASE is utilized to solve a sequence or correlated
	     eigenproblems.

.. cpp:var:: const double tol

	     It specifies the accuracy of the
	     solutions. :cpp:var:`tol` is the minimum tolerance the
	     residuals of the required eigenpairs should have to be
	     declared converged.

.. cpp:var:: const CHASE_MODE_TYPE mode
	      
	     It specifies whether the user provides ChASE with
	     information about the approximate solution of the
	     eigenproblem (e.g. when dealing with a sequence of
	     correlated eigenproblems) or uses it in isolation from
	     application knowledge as a traditional black-box solver.
	     When equal to ``A``, the arrays :cpp:var:`V` and
	     :cpp:var:`ritz` contain on entry the approximate vectors
	     and values respectively. The first and last value in
	     :cpp:var:`ritz` are used as estimates for the lower and
	     upper end of the sought after eigenspectrum. The vectors
	     in :cpp:var:`V` are used in the Chebyshev filter to
	     accelerate convergence. When equal to ``R``
	     the initial vectors used by the Chebyshev filter are
	     computed by the Lanczos routine starting from a set of
	     random vectors. Likewise, the Lanczos routine computes
	     estimates for the upper and lower end of the sought after
	     spectrum.
          
.. cpp:var:: const CHASE_OPT_TYPE opt

	     It specifies whether ChASE uses the same polynomial
	     degree for all the vectors to be filtered or run only one
	     loop iteration using the same polynomial degree for all
	     vectors and computes an array of optimal
	     :cpp:var:`degrees` for each vector at all successive
	     iteration loops.  When equal to ``N``, the same
	     polynomial degree :cpp:var:`deg` is used to filter all
	     the vectors in the search space *S*.  When equal to
	     ``S``, the polynomial degree :cpp:var:`deg` is used only
	     for the first call to the filter. Additional calls to the
	     filter use an optimal polynomial degree tailored to each
	     filtered vector.
		  
