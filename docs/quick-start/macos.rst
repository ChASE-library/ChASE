Installing dependencies on Mac OS
---------------------------------

On any Apple computer running a Mac OS we warmly invite to use
`Macports <https://www.macports.org/>`_ (`XCode
<https://developer.apple.com/xcode/>`_ is required and can be
downloaded directly using the Apple Store application) to install the
required dependencies. The installation of Macports and any of the
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

