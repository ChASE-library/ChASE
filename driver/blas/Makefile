BUILDDIR=../../build
TESTPROBLEMDIR=~


${BUILDDIR}/chase_blas.x: | mk_builddir
	icpc -DOUTPUT -std=c++11 -qopenmp -g \
	  -I../../testframework -I../../genera/blas \
	  -I../../algorithm \
	  main.cpp \
	  -mkl -lboost_program_options -lboost_serialization \
	  -o ${BUILDDIR}/chase_blas.x
	h5pcc -g -c hdf5.c -o ${BUILDDIR}/hdf5.o
	icpc -DOUTPUT -std=c++11 -qopenmp -g \
	  -I../../genera/blas  -I../../algorithm \
	  lib.cpp \
	  -c -o ${BUILDDIR}/chase_blas.o
	ar rvs ${BUILDDIR}/libchase.a ${BUILDDIR}/hdf5.o ${BUILDDIR}/chase_blas.o

${BUILDDIR}/chase_blas.done: ${BUILDDIR}/chase_blas.x
	${BUILDDIR}/chase_blas.x --n 3893 --mode "R" --opt "S" --name nacl   \
	  --path_in ${TESTPROBLEMDIR}/ --legacy on --nev 256 --bgn 1 --end 1 \
	  --nex 50
	${BUILDDIR}/chase_blas.x --n 3893 --mode "R" --opt "n" --name nacl   \
	  --path_in ${TESTPROBLEMDIR}/ --legacy on --nev 256 --bgn 1 --end 1 \
	  --nex 50
	touch ${BUILDDIR}/chase_blas.done

clean:
	rm -fr ${BUILDDIR}/chase_blas.x ${BUILDDIR}/chase_blas.done

test: ${BUILDDIR}/chase_blas.done

mk_builddir:
	mkdir -p ${BUILDDIR}

build: ${BUILDDIR}/chase_blas.x
