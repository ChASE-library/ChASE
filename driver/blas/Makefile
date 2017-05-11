BUILDDIR=../../build
TESTPROBLEMDIR=~
GENUS=blas

.PHONY: lib clean test mk_builddir

lib: ${BUILDDIR}/libchase_${GENUS}.a

clean:
	rm -fr ${BUILDDIR}/chase_${GENUS}.x ${BUILDDIR}/chase_${GENUS}.done

test: ${BUILDDIR}/chase_${GENUS}.done

mk_builddir:
	mkdir -p ${BUILDDIR}

build: ${BUILDDIR}/chase_${GENUS}.x

${BUILDDIR}/chase_${GENUS}.x: | mk_builddir
	icpc -DOUTPUT -std=c++11 -qopenmp -g \
	  -I../../testframework -I../../genera/blas \
	  -I../../algorithm \
	  main.cpp \
	  -mkl -lboost_program_options -lboost_serialization \
	  -o ${BUILDDIR}/chase_blas.x

${BUILDDIR}/libchase_${GENUS}.a: mk_builddir
	h5pcc -g -c hdf5.c -o ${BUILDDIR}/hdf5.o
	mpicxx -DOUTPUT -std=c++11 -fopenmp -g \
	  -I../../genera/${GENUS}  -I../../algorithm \
	  lib.cpp \
	  -c -o ${BUILDDIR}/chase_${GENUS}.o
	ar rvs ${BUILDDIR}/libchase_${GENUS}.a ${BUILDDIR}/hdf5.o ${BUILDDIR}/chase_${GENUS}.o

${BUILDDIR}/chase_blas.done: ${BUILDDIR}/chase_blas.x
	${BUILDDIR}/chase_blas.x --n 3893 --mode "R" --opt "S" --name nacl   \
	  --path_in ${TESTPROBLEMDIR}/ --legacy on --nev 256 --bgn 1 --end 1 \
	  --nex 50
	${BUILDDIR}/chase_blas.x --n 3893 --mode "R" --opt "n" --name nacl   \
	  --path_in ${TESTPROBLEMDIR}/ --legacy on --nev 256 --bgn 1 --end 1 \
	  --nex 50
	touch ${BUILDDIR}/chase_blas.done


# ${BUILDDIR}/chase_blas.x: | mk_builddir
# 	icpc -DOUTPUT -std=c++11 -qopenmp -g \
# 	  -I../../testframework -I../../genera/blas \
# 	  -I../../algorithm \
# 	  main.cpp \
# 	  -mkl -lboost_program_options -lboost_serialization \
# 	  -o ${BUILDDIR}/chase_blas.x

# ${BUILDDIR}/libchase_blas.a:
# 	h5pcc -g -c hdf5.c -o ${BUILDDIR}/hdf5.o
# 	mpicxx -cxx=icpc -DOUTPUT -std=c++11 -qopenmp -g \
# 	  -I../../genera/blas  -I../../algorithm \
# 	  lib.cpp \
# 	  -c -o ${BUILDDIR}/chase_blas.o
# 	ar rvs ${BUILDDIR}/libchase_blas.a ${BUILDDIR}/hdf5.o ${BUILDDIR}/chase_blas.o


# clean:
# 	rm -fr ${BUILDDIR}/chase_blas.x ${BUILDDIR}/chase_blas.done

# test: ${BUILDDIR}/chase_blas.done

# mk_builddir:
# 	mkdir -p ${BUILDDIR}

# build: ${BUILDDIR}/chase_blas.x
