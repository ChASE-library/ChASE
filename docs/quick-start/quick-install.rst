Cloning ChASE source code
--------------------------

ChASE is an open source project and it is available on `GitHub
<https://github.com/>`_. In order to download the source code of ChASE
one needs to have the `git <http://git-scm.com/>`_ utility installed. 
To clone a local copy of the ChASE repository execute the command::

    git clone https://github.com/ChASE-library/ChASE


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
examples which use ChASE to solve eigenproblems. Some of these examples make
the additional use of the 
``C++`` library ``Boost`` for the parsing of command line values. Thus
``Boost`` should also be provided before the installation of ChASE if users
would like to build ChASE with these examples.
In order to build these examples together with ChASE
the sequence of building commands should be slightly modified as below::

  cd ChASE/
  mkdir build
  cd build/
  cmake .. -DCMAKE_INSTALL_PREFIX=${ChASEROOT} -DBUILD_WITH_EXAMPLES=ON
  make install

Executing ChASE using the ready-to-use examples is rather
straightforward. For instance, :ref:`hello-world-chase` is executed by simply typing
the line below::

  ./0_hello_world/0_hello_world

In this example, a Clement matrix is generated and default values of parameters are used.  

To run this example with MPI, start the command with the mpi launcher of your choice, e.g. `mpirun` or `srun`.

For sake of completeness we provide a complete list of parameters in this example below.

.. table::

  ========================= ===================================================================================================
  Parameter (default value) Description
  ========================= ===================================================================================================
  N (=1001)                 Size of the Input Matrix
  nev (=40)                 Wanted Number of Eigenpairs
  nex (=20)                 Extra Search Dimensions
  deg (=20)                 Initial filtering degree, value set by ``config.SetDeg(20)``
  tol (=1e-10)              Minimum tolerance required to declare eigenpairs converged, value set by ``config.SetTol(1e-10)``
  opt (=true)               If optimize the degree of filter internally by ChASE, value set by ``config.SetOpt(true)``
  ========================= ===================================================================================================

.. note::  
  For the quick test and benchmark, user can modify some of parameters, e.g., to change the size of matrix ``N`` which will generate 
  a clement matrix of different size, to change the number of wanted eigepairs ``nev``, etc.

.. note::
  For the fine tuning of more parameters in ChASE, please visit :ref:`configuration_object`, in which we provide a class
  to set up all the parameters of eigensolvers. For the suggestion of selecting values of parameters, please visit :ref:`parameters_and_config`.

.. note::
  For a complete explanation of all the examples, please visit :ref:`examples-chase`.
