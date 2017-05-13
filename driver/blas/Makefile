BUILDDIR=../../build
TESTPROBLEMDIR=~
GENUS=blas

.PHONY: lib clean test mk_builddir

lib: ${BUILDDIR}/libchase_${GENUS}.a

clean:
	:

test: ${BUILDDIR}/chase_${GENUS}.done

mk_builddir:
	mkdir -p ${BUILDDIR}

build: | ${BUILDDIR}/chase_${GENUS}.x

${BUILDDIR}/chase_${GENUS}.x: mk_builddir
	icpc -DOUTPUT -std=c++11 -qopenmp -g \
	  -I../../testframework -I../../genera/${GENUS} \
	  -I../../algorithm \
	  main.cpp \
	  -mkl -lboost_program_options -lboost_serialization \
	  -o ${BUILDDIR}/chase_${GENUS}.x

${BUILDDIR}/libchase_${GENUS}.a: mk_builddir
	h5pcc -g -c hdf5.c -o ${BUILDDIR}/hdf5.o
	mpicxx -DOUTPUT -std=c++11 -fopenmp -g \
	  -I../../genera/${GENUS}  -I../../algorithm \
	  lib.cpp \
	  -c -o ${BUILDDIR}/chase_${GENUS}.o
	ar rvs ${BUILDDIR}/libchase_${GENUS}.a ${BUILDDIR}/hdf5.o ${BUILDDIR}/chase_${GENUS}.o

${BUILDDIR}/chase_${GENUS}.done: ${BUILDDIR}/chase_${GENUS}.x
	${BUILDDIR}/chase_${GENUS}.x --n 3893 --mode "R" --opt "S" --name nacl   \
	  --path_in ${TESTPROBLEMDIR}/ --legacy on --nev 256 --bgn 1 --end 1 \
	  --nex 50
	${BUILDDIR}/chase_${GENUS}.x --n 3893 --mode "R" --opt "n" --name nacl   \
	  --path_in ${TESTPROBLEMDIR}/ --legacy on --nev 256 --bgn 1 --end 1 \
	  --nex 50
	touch ${BUILDDIR}/chase_${GENUS}.done
