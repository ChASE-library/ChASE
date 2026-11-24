***********
Quick Start
***********

This section gives the opportunity to the prospective user to quickly
set-up the necessary environment to install and test the ChASE library
without the necessity of having access to a computing cluster. In the
following, we provide simple step-by-step instructions on how to
install ChASE on a laptop or a workstation equipped with a Linux or
Unix OS. Because ChASE uses `CMake <http://www.cmake.org>`__ for
autodecting dependencies and mananging configuration options across
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

    git checkout v1.0.0


Dependencies
============

In addition to a recent ``C++`` compiler ChASE's external dependencies are
`CMake <http://www.cmake.org/>`__ , `MPI
<http://en.wikipedia.org/wiki/Message_Passing_Interface>`__ , `BLAS <http://netlib.org/blas>`__ ,
`LAPACK <http://netlib.org/lapack>`__. To enhance the usability of the
ready-to-use examples, it is also necessary to install the `Boost
<https://www.boost.org/>`__ library. 

.. toctree::
   :maxdepth: 2

   quick-start/ubuntu
   quick-start/macos


Quick Installation and Execution
=================================

Installing the ChASE library on Linux or Mac OS requires cloning the
source files from a public github repository and compile them in few
steps. An example, provided with the source files, can be used to
run ChASE on a single computing node for the solution of an isolated
Hermitian standard eigenproblem.

.. toctree::
   :maxdepth: 2

   quick-start/quick-install
..
   quick-start/quick-run





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
