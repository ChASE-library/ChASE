.. image:: /images/ChASE_Logo_RGB.png

***************************************************
ChASE: an Iterative Solver for Dense Eigenproblems
***************************************************

Overview
=========

The **Ch**\ ebyshev **A**\ ccelerated **S**\ ubspace **E**\ igensolver
(ChASE) is a modern and scalable library to solve two types of dense
algebraic eigenvalue problems:

1. **Hermitian (Symmetric) eigenvalue problems** of the form

   .. math::

      A \hat{x} = \lambda \hat{x} \quad \textrm{with} \quad A^\dagger=A \in
      \mathbb{C}^{n\times n}\ \ (A^T=A \in \mathbb{R}^{n\times n}),

   where :math:`\hat{x} \in \mathbb{C}^{n}\backslash \{0\}` and
   :math:`\lambda \in \mathbb{R}` are the eigenvector and the eigenvalue
   of :math:`A`, respectively.

2. **Pseudo-Hermitian eigenvalue problems**, for example from Bethe-Salpeter
   Equation (BSE). The Hamiltonian :math:`H` derived from the BSE equation is
   of the form

   .. math::

      H := \begin{bmatrix}
          A & B\\
          -\bar{B} & -\bar{A}
      \end{bmatrix} \quad \textrm{with} \quad A = A^* \quad \textrm{and} \quad B = B^T.

   The :math:`m \times m` blocks :math:`A` and :math:`B` are respectively
   referred to as resonant and coupling terms. The two blocks :math:`\bar{A}`
   and :math:`\bar{B}` stand for the conjugate of :math:`A` and :math:`B`.
   Because of the properties of :math:`A` and :math:`B` stated above,
   :math:`\bar{A} = A^T` and :math:`\bar{B} = B^*`, and the size of :math:`H`
   is :math:`n := 2m`. The Hamiltonian :math:`H` is termed a
   pseudo-Hermitian matrix, as it satisfies the relation

   .. math::

      SH = H^*S \quad \textrm{with} \quad S := \begin{bmatrix}
          I & 0 \\
          0 & -I
      \end{bmatrix}.

Algorithm
==========

.. image:: /images/flow-chart-chase_standalone.png
   :scale: 90 %
   :align: center

.. table::
   :widths: 10,55
   :name: Flowchart legenda

   =============================== ===========================================================================================================
   :math:`A, N`                    The Hermitian matrix and corresponding rank.                    
   nev,nex                         Number of desired eigenpairs, Extra size of search space        
   tol,deg                         Threshold of residual's tolerance, initial degree of Chebyshev polynomials
   :math:`\hat{V},\hat{Q},\hat{X}` Matrix of vectors: filtered, orthonormalized, deflated and locked 
   :math:`\tilde{\Lambda},\Lambda` Matrix of eigenvalues: computed, deflated and locked
   :math:`m[],\textsf{res}[]`      Vectors: optimized degrees, eigenpairs residuals
   FILTER                          Chebyshev polynomial filter aligning :math:`\hat{V}` to the desired eigenspace 	 
   ORTHONORMALIZE                  QR factorization orthogonalizing filtered vectors together with deflated ones
   NORMAL-RAYLEIGH-RITZ            Projection of :math:`A` into search space defined by :math:`\hat{Q}` and diagonalization of reduced problem
   OBLIQUE-RAYLEIGH-RITZ           Projection of :math:`H` into search space defined by :math:`\hat{Q}` and diagonalization of reduced problem
   DEFL&LOCK                       Deflation and locking of eigenpairs whose residuals are below the tolerance threshold
   DEGREES                         Computation of optimal polynomial degree for each vector in :math:`\hat{V}` 
   =============================== ===========================================================================================================
   

Use Case and Features
======================

   * **Real and Complex:** ChASE is templated for real and complex numbers. So it can
     be used to solve *real symmetric* eigenproblems as well as *complex
     Hermitian* ones.
   * **Eigespectrum:** ChASE algorithm is designed to solve for the
     *extremal portion* of the eigenspectrum (i.e., :math:`\{\lambda_1,
     \dots ,\lambda_\textsf{nev}\}`). By default it computes the
     lowest portion of the spectrum but it can compute as well the
     largest portion by solving for :math:`-A`\ . The library is
     particularly efficient when no more than 20% of the extremal
     portion of the eigenspectrum is sought after. For larger
     fractions the subspace iteration algorithm may struggle to be
     competitive. Converge could become an issue for fractions close
     to or larger than 50%.
   * **Type of Problem:** ChASE can currently handle only standard
     eigenvalue problems. Generalized eigenvalue problems of the form
     :math:`A\hat{x} = \lambda B \hat{x}`, with :math:`B` s.p.d., can
     be solved after factorizing :math:`B = L L^T` and transforming
     the problem into standard form :math:`A = L^{-1} A L^{-T}`.
   * **Sequences:** ChASE is particularly efficient when dealing with
     *sequences of eigenvalue problems*\ , where the eigenvectors
     solving for one problem can be use as input to accelerate the
     solution of the next one.

   * **Vectors input:** Since it is based on subspace iteration, ChASE
     can receive as input a matrix of vector :math:`\hat{V}` equal to
     the number of desired eigenvalues. ChASE can experience
     substantial speed-ups when :math:`\hat{V}` contains some
     information about the sought after eigenvectors.

   * **Degree optimization:** For a fixed accuracy level, ChASE can
     optimize the degree of the Chebyshev polynomial filter so as to
     minimize the number of FLOPs necessary to reach convergence.

   * **Precision:** ChASE is also templated to work in *Single
     Precision* (SP) or *Double Precision* (DP).
