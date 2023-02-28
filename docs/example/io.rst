ChASE Object and Configuration 
------------------------------

ChASE parameters can be configured as follows:

.. code:: c++
  
  typedef ChaseMpi<ChaseMpiDLABlaslapack, T> CHASE;
  /* Construction of ChASE object */
  CHASE single(new ChaseMpiProperties<T>(N, nev, nex, MPI_COMM_WORLD), V.data(),
               Lambda.data());

  /* Get the class of configuration from ChASE object */
  auto& config = single.GetConfig();

  /* Parameter configurations */
  config.SetTol(conf.tol);
  config.SetDeg(conf.deg);
  config.SetOpt(true);
  ...

.. note::

    For all the provided public funtions of ChASE configuration, please refer to :ref:`configuration_object`.


Execution
----------

This examples provides multiple implementation of ChASE targeting different computing architectures with different data distribution scheme:

1. ``2_input_output_seq``: sequential ChASE without MPI support, and **Block Distribution**.

2. ``2_input_output``: ChASE for distributed memory systems with MPI support (pure CPUs),  and **Block Distribution**.

3. ``2_input_output_block_cyclic``: ChASE for distributed memory systems with MPI support (pure CPUs),  and **Block-Cyclic Distribution**.

4. ``2_input_output_mgpu``: ChASE for distributed memory systems with MPI support (with GPUs),  and **Block Distribution**.

5. ``2_input_output_mgpu_block_cyclic``: ChASE for distributed memory systems with MPI support (with GPUs),  and **Block-Cyclic Distribution**.


This example uses `Boost` for parsing the parameters, thus the
required parameters and configuration can be gotten by the `help`
flag:

.. code-block:: sh

    ./2_input_output/2_input_output -h


Solving single problem
^^^^^^^^^^^^^^^^^^^^^^^^

Here we utilize ``2_input_output`` as an example to illustrate the way to use ChASE to solve single eigenproblem with loading external matrix.

The execution of this example through the command line is:

.. code-block:: sh

    mpirun -np ${NPROCS} ./2_input_output/2_input_output --path_in=${MATRIX_BINARY_FILE} --n=${RANK_OF_MATRIX} --nev=${NB_of_WANTED_EIGENPAIRS} --nex=${EXTERNAL_SEARCHNING_SPACE} --mode=R


Solving a sequence of problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we also utilize ``2_input_output`` as an example to illustrate the way to use ChASE to solve a sequence of eigenproblems with loading external matrix.

The execution of this example through the command line is:

.. code-block:: sh

    mpirun -np ${NPROCS} ./2_input_output/2_input_output --path_in=${DIRECTORY_STORE_MATRICES} --n=${RANK_OF_MATRIX} --nev=${NB_of_WANTED_EIGENPAIRS} --nex=${EXTERNAL_SEARCHNING_SPACE} --legacy=true --mode=R --bgn=2 --end=10 --sequence=true

In this example, for each physical system, a number (``N``) of matrices are available, which should be solved in sequence. 
All the matrices are named ``gmat/ /1/ ell``, with ``ell`` varying from ``1`` to ``N``. The flag ``--legacy=true`` enables
searching the matrices conforming this naming policies in the directory ``${DIRECTORY_STORE_MATRICES}``.

In the execution example above, the first matrix to be solved is ``gmat/ /1/ 2``, the the last matrix  to be solved is ``gmat/ /1/ 10``.

Parameters of configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

========================= ===================================================================================================
Parameter (default value) Description
========================= ===================================================================================================
  -h [ --help ]           show this message
  --n arg                 Size of the Input Matrix
  --double arg (=1)       Is matrix complex double valued, false indicates the
                          single type
  --complex arg (=1)      Matrix is complex valued
  --nev arg               Wanted Number of Eigenpairs
  --nex arg (=25)         Extra Search Dimensions
  --deg arg (=20)         Initial filtering degree
  --bgn arg (=2)          Start ell
  --end arg (=2)          End ell
  --tol arg (=1e-10)      Tolerance for Eigenpair convergence
  --path_in arg           Path to the input matrix/matrices
  --mode arg (=A)         valid values are R(andom) or A(pproximate)
  --opt arg (=S)          Optimi(S)e degree, or do (N)ot optimise
  --path_eigp arg         Path to approximate solutions, only required when
                          mode is Approximate, otherwise not used
  --sequence arg (=0)     Treat as sequence of Problems. Previous ChASE solution
                          is used,when available
  --mbsize arg (=400)     block size for the row, it only matters for **Block-Cyclic Distribution**.
  --nbsize arg (=400)     block size for the column, it only matters for **Block-Cyclic Distribution**.
  --dim0 arg (=0)         row number of MPI proc grid, it only matters for **Block-Cyclic Distribution**.
  --dim1 arg (=0)         column number of MPI proc grid, it only matters for **Block-Cyclic Distribution**.
  --irsrc arg (=0)        The process row over which the first row of matrix
                          is distributed. It only matters for **Block-Cyclic Distribution**.
  --icsrc arg (=0)        The process column over which the first column of the
                          array A isdistributed. It only matters for **Block-Cyclic Distribution**.
  --major arg (=C)        Major of MPI proc grid, valid values are R(ow) or
                          C(olumn). It only matters for **Block-Cyclic Distribution**.
  --legacy arg (=0)       Use legacy naming scheme?
========================= ===================================================================================================


.. note:: 
  We have generated a few number of matrices defining (sequences of) eigenproblems from multiple material science simulation codes, if you want to
  test with these matrices, please feel free to contact us.


.. note::
  For the fine tuning of more parameters in ChASE, please visit :ref:`configuration_object`, in which we provide a class
  to set up all the parameters of eigensolvers. For the suggestion of selecting values of parameters, please visit :ref:`parameters_and_config`.
