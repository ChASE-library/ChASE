***********
Quick Start
***********

This section gives the opportunity to the prospective user to quickly
set-up the necessary environment to install and test the ChASE library
without the necessity of having access to a computing cluster. In the
following, we provide simple step-by-step instructions on how to
install ChASE on a laptop or a workstation equipped with a Linux or
Unix OS. Because ChASE uses `CMake <http://www.cmake.org>`__ for
auto-detecting dependencies and managing configuration options across
different platforms, it can be easily configured on any Linux and Unix
based operating systems. Multiple examples are also provided, and
user can utilize them to directly test ChASE on a matrix of his choice.


Getting ChASE
=============


ChASE can be easily obtained by cloning the repository directly using ``git``:

.. code-block:: sh

    git clone https://github.com/ChASE-library/ChASE

It is also recommended that you check out the latest stable tag:

.. code-block:: sh

    git checkout v1.3.0


Dependencies
============

In addition to a recent ``C++`` compiler ChASE's external dependencies are
`CMake <http://www.cmake.org/>`__ , `MPI
<http://en.wikipedia.org/wiki/Message_Passing_Interface>`__ , `BLAS <http://netlib.org/blas>`__ ,
`LAPACK <http://netlib.org/lapack>`__. To enhance the usability of the
ready-to-use examples, it is also necessary to install the `Boost
<https://www.boost.org/>`__ library. 

Installing dependencies on Linux
--------------------------------

The following instructions for the installation of the prerequisite
modules have been tested on the `Ubuntu <http://www.ubuntu.com/>`__
operating system (e.g., version 18.10), but they should work as well
with most modern Linux OS.

CMake
^^^^^

Install CMake by executing the following command::

    sudo apt-get install cmake

GNU compiler
^^^^^^^^^^^^

To install the GNU ``C`` compiler, GNU ``C++`` compiler, build utility for
the compilation, and add some development tools execute the following command::

    sudo apt-get install build-essential

Installing BLAS and LAPACK
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Basic Linear Algebra Subprograms (BLAS) and Linear Algebra PACKage
(LAPACK) are both used heavily within ChASE. On most installations of
`Ubuntu <http://www.ubuntu.com>`__, there are two optimized BLAS/LAPACK
version available: `OpenBLAS <http://www.openblas.net>`__ and `ATLAS
<http://math-atlas.sourceforge.net/>`__. To install them use either of
the following commands::

    sudo apt-get install libopenblas-dev
    or
    sudo apt-get install libatlas-dev liblapack-dev

Installing MPI
^^^^^^^^^^^^^^

ChASE requires an implementation of the Message Passing Interface
(MPI) communication protocol. The two most commonly used
MPI implementations are `MPICH <https://www.mpich.org>`_, and `OpenMPI
<http://www.open-mpi.org/>`_.

`MPICH <https://www.mpich.org>`_ can be installed by executing the
following command::

    sudo apt-get install libmpich2-dev

while `OpenMPI <http://www.open-mpi.org/>`_ can be installed by executing::

    sudo apt-get install libopenmpi-dev


Installing dependencies on Mac OS
---------------------------------

On any Apple computer running a Mac OS we warmly invite to use
`Macports <https://www.macports.org/>`_ (`XCode
<https://developer.apple.com/xcode/>`_ is required and can be
downloaded directly using the Apple Store application) to install the
required dependencies. The installation of MacPorts and any of the
supported *ports* requires administration privileges. 

CMake
^^^^^

Install CMake by executing the following command::

    sudo port install cmake

GNU compiler
^^^^^^^^^^^^

To install the GNU ``C`` compiler, GNU ``C++`` compiler, and GNU
``Fortran`` compiler execute the following commands::

    sudo port install gcc10
    sudo port select --set gcc mp-gcc10

    
Installing BLAS and LAPACK
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Basic Linear Algebra Subprograms (BLAS) and Linear Algebra PACKage
(LAPACK) are both used heavily within ChASE.  On an Apple computer one
can either use the `Accelerate
<https://developer.apple.com/documentation/accelerate>`_ framework,
which is provided out of the box, or one could install `OpenBLAS
<https://www.openblas.net/>`_ by executing the following command::

    sudo port install OpenBLAS +native



Installing MPI
^^^^^^^^^^^^^^

ChASE requires an implementation of the Message Passing Interface
(MPI) communication protocol. The two most commonly used
MPI implementations are `MPICH <https://www.mpich.org>`_, and `OpenMPI
<http://www.open-mpi.org/>`_.

`MPICH <https://www.mpich.org>`_ can be installed by executing the
following commands::

    sudo port install mpich
    sudo port select --set mpi mpich-mp-fortran

while `OpenMPI <http://www.open-mpi.org/>`_ can be installed by executing::

    sudo port install openmpi
    sudo port select --set mpi openmpi-mp-fortran

Quick Installation and Execution
=================================

Installing the ChASE library on Linux or Mac OS requires cloning the
source files from a public github repository and compile them in few
steps. An example, provided with the source files, can be used to
run ChASE on a single computing node for the solution of an isolated
Hermitian standard eigenproblem.

.. _build-label:

Building and Installing the ChASE library
------------------------------------------

On a Linux system with MPI and CMake installed in the standard
locations, ChASE can be build by executing in order the
following commands (after having cloned the repository):

.. code-block:: sh
      
    cd ChASE/
    mkdir build
    cd build/
    cmake .. -DCMAKE_INSTALL_PREFIX=${ChASEROOT}
    make install

In the commands above, the variable ``${CHASEROOT}`` is the path to
install ChASE on user's laptops.
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
Accelerate, one could execute the following command:

.. code-block:: sh

    cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_Fortran_COMPILER=gfortran ..


.. note::
   If you want to try with ChASE or use ChASE as standalone eigensolver, the CMake flag ``-DCMAKE_INSTALL_PREFIX=${ChASEROOT}`` is not mandatory. 

Quick Hands-on by Examples
------------------------------

For a quick test and usage of the library, we provide various ready-to-use
examples which use ChASE to solve eigenproblems. Some of these examples make
the additional use of the 
``C++`` library ``Boost`` for the parsing of command line values. Thus
``Boost`` should also be provided before the installation of ChASE if users
would like to build ChASE with these examples.
In order to build these examples together with ChASE
the sequence of building commands should be slightly modified as
below:

.. code-block:: sh

    cd ChASE/
    mkdir build
    cd build/
    cmake .. -DCMAKE_INSTALL_PREFIX=${ChASEROOT} -DBUILD_WITH_EXAMPLES=ON
    make install

Executing ChASE using the ready-to-use examples is rather
straightforward. For instance, :ref:`hello-world-chase` is executed by simply typing
the line below:

.. code-block:: sh

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






..
   note:: Some LOOSE suggestions for sections and content (SEBASTIAN)

   * A driver for the reference implementation

     * Snippets of code
     * Link to a driver in the repository

   * A driver for the MPI (Elemental) implementation

     * Snippets of code
     * Link to a driver in the repository

   * Examples of usage
   * Regression system: Boost
