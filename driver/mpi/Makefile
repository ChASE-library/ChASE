BUILDDIR=../../build
TESTPROBLEMDIR=~
GENUS=mpi

.PHONY: lib clean test mk_builddir

lib: ${BUILDDIR}/libchase_${GENUS}.a

clean:
	:

test: ${BUILDDIR}/chase_${GENUS}.done

mk_builddir:
	mkdir -p ${BUILDDIR}

build: | ${BUILDDIR}/chase_${GENUS}.x # ${BUILDDIR}/chase_${GENUS}_gpu.x

${BUILDDIR}/chase_${GENUS}_gpu.x: mk_builddir
	h5pcc -g -c hdf5.c -o ${BUILDDIR}/hdf5_${GENUS}.o
	mpicxx -cxx=icpc -DOUTPUT -DGPU_MODE -std=c++11 -g \
	  -I../../testframework -I../../genera/${GENUS} \
	  -I../../algorithm -I/usr/include/cuda/\
	  main.cpp -c -o ${BUILDDIR}/main_${GENUS}.o
	nvcc ../../genera/mpi/shift.cu -c -o ${BUILDDIR}/shift_${GENUS}.o
	h5pcc -qopenmp -g \
	  ${BUILDDIR}/main_${GENUS}.o ${BUILDDIR}/hdf5_${GENUS}.o ${BUILDDIR}/shift_${GENUS}.o\
	    -lcudart -lcublas \
	  -mkl -lboost_program_options -lboost_serialization -lstdc++ \
	  -o ${BUILDDIR}/chase_${GENUS}_gpu.x

${BUILDDIR}/chase_${GENUS}.x: mk_builddir
#	h5pcc -g -c hdf5.c -o ${BUILDDIR}/hdf5_${GENUS}.o
	mpicxx -cxx=g++ -DOUTPUT -fopenmp -std=c++11 -g \
	  -I../../testframework -I../../genera/${GENUS} \
	  -I../../algorithm -I/usr/include/cuda/\
	  main.cpp -c -o ${BUILDDIR}/main_${GENUS}.o
# echo ${BUILDDIR}/hdf5_${GENUS}.o 
	mpicxx -cxx=icpc -fopenmp -g \
	  ${BUILDDIR}/main_${GENUS}.o \
	  -mkl -lboost_program_options -lboost_serialization -lstdc++ \
	  -o ${BUILDDIR}/chase_${GENUS}.x


${BUILDDIR}/libchase_${GENUS}.a: mk_builddir
	h5pcc -g -c hdf5.c -o ${BUILDDIR}/hdf5_${GENUS}.o
	mpicxx -cxx=icpc -DOUTPUT -std=c++11 -qopenmp -g \
	  -I../../genera/${GENUS}  -I../../algorithm \
	  lib.cpp  -c -o ${BUILDDIR}/chase_${GENUS}.o
	  ar rvs ${BUILDDIR}/libchase_${GENUS}.a ${BUILDDIR}/hdf5_${GENUS}.o ${BUILDDIR}/chase_${GENUS}.o

${BUILDDIR}/chase_${GENUS}.done: ${BUILDDIR}/chase_${GENUS}.x
	mpirun -np 1 ${BUILDDIR}/chase_mpi.x --n 3893 --mode "R" --opt "S" --name nacl   \
	  --path_in ${TESTPROBLEMDIR}/ --legacy on --nev 256 --bgn 1 --end 1 \
	  --nex 50
	mpirun -np 4 ${BUILDDIR}/chase_mpi.x --n 3893 --mode "R" --opt "S" --name nacl   \
	  --path_in ${TESTPROBLEMDIR}/ --legacy on --nev 256 --bgn 1 --end 1 \
	  --nex 50
	mpirun -np 6 ${BUILDDIR}/chase_mpi.x --n 3893 --mode "R" --opt "n" --name nacl   \
	  --path_in ${TESTPROBLEMDIR}/ --legacy on --nev 256 --bgn 1 --end 1 \
	  --nex 50
	mpirun -np 7 ${BUILDDIR}/chase_mpi.x --n 3893 --mode "R" --opt "S" --name nacl   \
	  --path_in ${TESTPROBLEMDIR}/ --legacy on --nev 256 --bgn 1 --end 1 \
	  --nex 50
	touch ${BUILDDIR}/chase_${GENUS}.done
