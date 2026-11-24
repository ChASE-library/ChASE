Installing dependencies on Linux
--------------------------------

The following instructions for the intallation of the prerequisite
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
