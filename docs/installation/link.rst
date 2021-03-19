Linking ChASE to other Application Software
---------------------------------------------

In this section, we give the guidelines for the integration
of the ChASE library into a given application software. 


Include headers
^^^^^^^^^^^^^^^

In order to use ChASE in an application code written in ``C/C++`` 
the following files have to be included in the header file.

.. code-block:: c++

    /*Performance Decorator of ChASE*/
    #include "algorithm/performance.hpp"

    /*Common interface of ChASE-MPI*/
    #include "ChASE-MPI/chase_mpi.hpp"

    /*USE ChASE-MPI without GPUs*/
    /*With MPI support for distributed-memory system*/
    #include "ChASE-MPI/impl/chase_mpihemm_blas.hpp"
    /*Without MPI support for single-node system*/
    #include "ChASE-MPI/impl/chase_mpihemm_blas_seq.hpp"
    #include "ChASE-MPI/impl/chase_mpihemm_blas_seq_inplace.hpp"

    /*USE ChASE-MPI with GPUs*/
    /*With MPI support for distributed-memory system*/
    #include "ChASE-MPI/impl/chase_mpihemm_mgpu.hpp"
    /*Without MPI support for single-node system*/    
    #include "ChASE-MPI/impl/chase_mpihemm_cuda_seq.hpp"


.. _link_by_cmake:

Linking to ChASE by using CMake
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``CMakeLists.txt`` (see code window below) is an example on how to link ChASE installation
using CMake. In this example ChASE is linked to a source file named ``chase_app.cpp``.
The ``CMakeLists.txt`` should then be included in the main directory
of the application software as well as the ``chase_app.cpp`` file.

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.8)

   project(chase-app VERSION 0.0.1 LANGUAGES CXX)

   #find installation of ChASE
   find_package( chase REQUIRED CONFIG)
   # find other dependencies of ChASE: BLAS, LAPACK, OpenMP, MPI, etc
   find_package( BLAS   REQUIRED )
   find_package( LAPACK REQUIRED )
   find_package( OpenMP)

   find_package(MPI REQUIRED)
   if(MPI_FOUND)
     SET(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
   endif()

   # enable OpenMP multi-threading
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

   add_executable(${PROJECT_NAME})

   # add the source file of application
   target_sources(${PROJECT_NAME} PRIVATE chase_app.cpp)

   target_compile_features(${PROJECT_NAME} INTERFACE cxx_auto_type)

   target_include_directories( ${PROJECT_NAME} INTERFACE
     ${MPI_CXX_INCLUDE_PATH}
     )

   # link to BLAS, LAPACK and MPI
   target_link_libraries( ${PROJECT_NAME} INTERFACE
     ${BLAS_LIBRARIES}
     ${LAPACK_LIBRARIES}
     ${MPI_CXX_LIBRARIES}
     )

   # link to ChASE
   target_link_libraries(${PROJECT_NAME} PUBLIC ChASE::chase_mpi)

   # if users want to compile the application with multi-GPU version of ChASE
   # the target should be linked to the both the library ChASE::chase_mpi
   # and the library ChASE::chase_cuda
   target_link_libraries(${PROJECT_NAME} PUBLIC ChASE::chase_cuda)

With CMake, the application software can be compiled by the following commands:

.. code-block:: console

   mkdir build & cd build
   cmake .. -DCMAKE_PREFIX_PATH=${ChASEROOT}
   make

`example: 3_installation <https://github.com/ChASE-library/ChASE/tree/master/examples/3_installation>`_
provides an example which illustrates the way to link ChASE by CMake with or without GPU supports. (This link should be replaced by the link on github later.)

.. note::
  We highly recommand to link ChASE with CMake. The installation of ChASE allows to use CMake to find and link it easily.

Direct linking
^^^^^^^^^^^^^^^

If users want to manually link to ChASE, it is necessary to link to
the ChASE installation, the ``BLAS/LAPACK`` libraries, ``OpenMP``
programming interface (Optional: if multi-threading is required), and
``CUDA`` programming interface (Optional: if CUDA is required).


Where to Find Useful Linking Information
"""""""""""""""""""""""""""""""""""""""""

The standard installation of ChASE can already provide some
information about the linking, which can be extracted when it
generates the `CMake configuration files`. More details, the linking information 
can be obtained from the
**lines 56-59** of the configuration file ``${ChASEROOT}/lib/cmake/ChASE/chase-mpi.cmake``.


Pure CPU version
"""""""""""""""""

For the pure CPU version, the installation of ChASE is header-only, so it is only necessary to include the ChASE header files and other external libraries such as ``BLAS/LAPACK``.

.. code-block:: console

    mpicxx chase-app.cpp -o chase-app -I${ChASEROOT}/include ${BLASLIBRARIES} ${LAPACKLIBRARIES}


Multi-GPU version
""""""""""""""""""

For the GPU version, apart from including the ChASE header files and other external libraries, it is also necessary to link against to the libraries ``libchase_cuda.a`` ``CUDA runtime``, ``cuBLAS`` and ``cuSOLVER``.

.. code-block:: console

    mpicxx chase-app-gpu.cpp -o chase-app-gpu -I${ChASEROOT}/include -L${ChASEROOT}/lib/libchase_cuda.a ${BLASLIBRARIES} ${LAPACKLIBRARIES} ${CUBLASLIBRARIES} ${CUSOLVERLIBRARIES} ${CUDA_RUNTIME_LIBRARIES}

.. note::
    For the users of Intel MKL as the BLAS/LAPACK implementation for
    ChASE, useful guidelines for linking can be found in `Intel® Math Kernel Library Link Line Advisor <https://software.intel.com/content/www/us/en/develop/articles/intel-mkl-link-line-advisor.html>`_ .


Linking by Makefile
^^^^^^^^^^^^^^^^^^^^

Similar as the direct linking, it is also possible to link ChASE by ``Makefile``. 
Here below is a template of this ``Makefile`` for `example: 3_installation <https://github.com/ChASE-library/ChASE/tree/master/examples/3_installation>`_.

.. code-block:: Makefile

  ChASEROOT = /The/installation/path/of/ChASE/on/your/platform

  CXX = mpicxx #or other mpi CXX compiler

  CXXFLAGS = \
      -Wall -fopenmp -MMD \

  INCLUDE_DIR = ${ChASEROOT}/include #include the headers of ChASE

  LIBS_BLASLAPACK = /your/BLAS/LAPACK/LIBRARIES

  ## Below is an example which uses MKL as BLAS/LAPACK ##
  #LIBS_BLASLAPACK = -lmkl_gf_lp64 \
  #                  -lmkl_gnu_thread \
  #                  -lmkl_core -lgomp \
  #                  -lpthread -lm \
  #                  -lmkl_gf_lp64 \
  #                  -lmkl_gnu_thread \
  #                  -lmkl_core

  ## Optional for multi-GPU version of ChASE ##
  LIBS_CUDA = -lcublas -lcusolver -lcudart ## link to the libraries of cuBLAS, cuSOLVER and CUDA runtime

  ## Optional for multi-GPU version of ChASE ##
  LIBS_CHASE_CUDA = ${ChASEROOT}/lib64/libchase_cuda.a

  chase-app: LIBS = ${LIBS_BLASLAPACK} #executable generated by chase-app.cpp

  chase-app-gpu: LIBS = ${LIBS_BLASLAPACK} -L${LIBS_CHASE_CUDA} ${LIBS_CUDA} #executable generated by chase-app-gpu.cpp

  src = ${wildcard *.cpp}
  exe = ${basename ${src}}

  all: $(exe)

  .SUFFIXES:

  %: %.cpp
          ${CXX} ${CXXFLAGS} ${LIBS} -I${INCLUDE_DIR} -o $@ $<

  clean:
          -rm -f $(exe) *.o

  -include *.d


.. Interface of ChASE to Fortran & C
   ----------------------------------

   ChASE is written in ``C++`` with templates, which is able to support the computation with
   multiple scalar types and precisions. In order to integrate ChASE into ``Fortran`` or ``C``
   based applications, we provide its interfaces to both ChASE and C.



.. Interface to Fortran
   ^^^^^^^^^^^^^^^^^^^^^

   Interface to Fortran

.. Interface to C
   ^^^^^^^^^^^^^^^

