Installation on a CPU-only Cluster
------------------------------------


The following snippet shows how to install ChASE on the JUWELS cluster
(the main general purpose cluster at the Juelich Supercomputing Centre):

.. code-block:: console

  git clone https://github.com/ChASE-library/ChASE.git
  cd ChASE/
  mkdir build
  cd build/
  ###       GCC      ###
  ml GCC/8.3.0  ParaStationMPI/5.4.4-1 imkl CMake
  cmake .. -DCMAKE_INSTALL_PREFIX=${ChASEROOT}
  make install
  ### Intel Compiler ###
  ml intel-para CMake
  cmake .. -DCMAKE_INSTALL_PREFIX=${ChASEROOT} -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc
  make install

.. note::

  For the installation with the ``Intel Compiler``, two additional flags ``-DCMAKE_C_FLAGS=-no-multibyte-chars`` and ``-DCMAKE_CXX_FLAGS=-no-multibyte-chars`` might be required if
  the following error ``Catastrophic error: could not set locale "" to
  allow processing of multibyte characters`` are encoutered, which is produced by an internal
  bug appearing in some versions of the Intel Compiler.

Installation with GPU Support
------------------------------------

In order to compile ChASE with GPU support one needs to have installed
and loaded a CUDA compiler, which will also enable the use of
``cuBLAS`` and ``cuSOLVER``. On the JUWELS cluster this can
be achieved by loading the module CUDA in addition to the
modules necessary to compile ChASE on a CPU-only cluster. Make sure, that
ChASE is executed on a computer/node with at least one GPU device
(e.g. check with ``nvidia-smi``) and that the correct  CUDA
compiler is loaded (e.g. check ``which nvcc`` or if you are using a module system
look at ``module list``). The following instruction snippet builds ChASE with CUDA
support on JUWELS:

.. code-block:: console

  git clone https://github.com/ChASE-library/ChASE.git
  cd ChASE/
  mkdir build
  cd build/
  ml GCC/8.3.0  ParaStationMPI/5.4.4-1 imkl CUDA CMake
  cmake .. -DCMAKE_INSTALL_PREFIX=${ChASEROOT}
  make install

Building ChASE with Examples
---------------------------------

To build and install ChASE with examples, the 
additional option to the cmake build process
``-DBUILD_WITH_EXAMPLES=ON`` has to be turned on. The following
instruction snippet builds ChASE with
examples on the JUWELS cluster:

.. code-block:: console

  git clone https://github.com/ChASE-library/ChASE.git
  cd ChASE/
  mkdir build
  cd build/
  ml intel-para CMake Boost
  ##### If you want to install ChASE with GPU supporting, make sure CUDA is loaded #####
  ml load CUDA
  cmake .. -DCMAKE_INSTALL_PREFIX=${ChASEROOT} -DBUILD_WITH_EXAMPLES=ON
  make install
  ### Run example #0 ###
  ./examples/0_hello_world/0_hello_world

An MPI launcher has to be used to run an example in parallel. For
instance on the JUWELS cluster (or any other ``SLRUM`` based Cluster)
the following command line runs the "`hello world`" example in parallel.

.. code-block:: console

  srun -n 2 ./examples/0_hello_world/0_hello_world
