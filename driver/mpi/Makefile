BUILDDIR=../../build
TESTPROBLEMDIR=~


${BUILDDIR}/chase_mpi.x: | mk_builddir
	mpicxx -cxx=icpc -DOUTPUT -std=c++11 -qopenmp -g \
	  -I../../testframework -I../../genera/mpi \
	  -I../../algorithm \
	  main.cpp \
	  -mkl -lboost_program_options -lboost_serialization \
	  -o ${BUILDDIR}/chase_mpi.x

clean:
	rm -fr ${BUILDDIR}/chase_mpi.x ${BUILDDIR}/chase_mpi.done

test: ${BUILDDIR}/chase_mpi.done

mk_builddir:
	mkdir -p ${BUILDDIR}

build: ${BUILDDIR}/chase_mpi.x
