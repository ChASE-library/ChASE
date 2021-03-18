Cloning ChASE source code
--------------------------

ChASE is an open source project and it is available on `GitHub
<https://github.com/>`_. In order to download the source code of ChASE
one needs to have the `git <http://git-scm.com/>`_ utility installed. 
To clone a local copy of the ChASE repository execute the command::

    git clone https://github.com/SimLabQuantumMaterials/ChASE.git


.. _build-label:

Building and Installing the ChASE library
------------------------------------------

On a Linux system with MPI and CMake installed in the standard
locations, ChASE can be build by executing in order the
following commands (after having cloned the repository)::

    cd ChASE/
    mkdir build
    cd build/
    cmake .. -DCMAKE_INSTALL_PREFIX=${ChASEROOT}
    make install

In the commands above, the variable ``${CHASEROOT}`` is the path to
install ChASE on user's laptops. In fact,
CMake will auto-detect the dependencies and select the default
installed modules. In order to select a specific module installation,
one can manually specify several build options,
especially when multiple versions of libraries or several different
compilers are available on the system. For instance, any ``C++``, ``C``, or
``Fortran`` compiler can be selected by setting the
``CMAKE_CXX_COMPILER``, ``CMAKE_C_COMPILER``, and
``CMAKE_Fortran_COMPILER`` variables, respectively. The following
provide an illustration of such setting. ::

    -D CMAKE_CXX_COMPILER=/usr/bin/g++ \
    -D CMAKE_C_COMPILER=/usr/bin/gcc   \
    -D CMAKE_Fortran_COMPILER=/usr/bin/gfortran

Analogously, it may be necessary to manually specify the paths to the
MPI implementation by, for example, setting the following variables. ::

    -D MPI_CXX_COMPILER=/usr/bin/mpicxx \
    -D MPI_C_COMPILER=/usr/bin/mpicc \
    -D MPI_Fortran_COMPILER=/usr/bin/mpif90

For instance, installing ChASE on an Apple computer with gcc and
Accelerate, one could execute the following command::

    cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_Fortran_COMPILER=gfortran ..


Quick Hands-on by Examples
------------------------------

For a quick test and usage of the library, we provide various ready-to-use
examples which use ChASE to solve eigenproblems. These examples make
the additional use of the 
``C++`` library ``Boost`` for the parsing of command line values. Thus
``Boost`` should also be provided before the installation of ChASE.
In order to build these examples together with ChASE
the sequence of building commands should be slightly modified as below::

  cd ChASE/
  mkdir build
  cd build/
  cmake .. -DCMAKE_INSTALL_PREFIX=${ChASEROOT} -DBUILD_WITH_EXAMPLES=ON
  make install

Executing ChASE using the ready-to-use examples is rather
straightforward. For instance, example 2 is executed by simply typing
the line below::

  ./2_input_output/2_input_output

One can call this example without any argument as above. In this case a
Clement matrix is generated and default values are used.  
One could also specify a subset or all arguments. The minimal
arguments that need to be provided are the input matrix and the size
of such matrix::

  ./2_input_output/2_input_output --n <MatrixSize> --path_in <YourOwnFolder/YourMatrixToSolve.bin>
  
To run this example with MPI, start the command with the mpi launcher of your choice, e.g. `mpirun` or `srun`.

All additional arguments can be listed with -h::

  ./2_input_output/2_input_output -h

For sake of completeness we provide a complete list below.

.. table::

  ========================= =================================================================
  Parameter (default value) Description
  ========================= =================================================================
  -h [ --help ]             Shows the full list of parameters
  --n arg (=1001)           Size of the Input Matrix
  --double arg (=1)         Are matrix entries of type double, false indicates type single
  --complex arg (=1)        Is matrix complex valued, false indicates a real matrix 
  --nev arg (=100)          Wanted Number of Eigenpairs
  --nex arg (=25)           Extra Search Dimensions
  --deg arg (=20)           Initial filtering degree
  --bgn arg (=2)            Start index of matrix sequence (if any) 
  --end arg (=2)            End index of matrix sequence (if any)
  --tol arg (=1e-10)        Minimum tolerance required to declare eigenpairs converged
  --path_in arg               Path to the input matrix/matrices
  --output arg (=eig.txt)   Path to the write the eigenvalues
  --mode arg (=R)           Valid values are ``R`` (Random) or ``A`` (Approximate)
  --opt arg (=N)            Valid values are Optimi ``S`` e, or do ``N`` ot optimise degree
  --path_eigp arg           Path to approximate solutions, only required when 
                            mode is ``A`` pproximate, otherwise not used
  ========================= =================================================================
