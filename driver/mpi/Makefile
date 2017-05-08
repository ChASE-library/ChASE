BUILDDIR=../../build
TESTPROBLEMDIR=~


${BUILDDIR}/chase_mpi.x: | mk_builddir
	mpicxx -cxx=icpc -DOUTPUT -std=c++11 -qopenmp -g \
	  -I../../testframework -I../../genera/mpi \
	  -I../../algorithm \
	  main.cpp \
	  -mkl -lboost_program_options -lboost_serialization \
	  -o ${BUILDDIR}/chase_mpi.x

${BUILDDIR}/chase_mpi.done: ${BUILDDIR}/chase_mpi.x
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
	touch ${BUILDDIR}/chase_blas.done


clean:
	rm -fr ${BUILDDIR}/chase_mpi.x ${BUILDDIR}/chase_mpi.done

test: ${BUILDDIR}/chase_mpi.done

mk_builddir:
	mkdir -p ${BUILDDIR}

build: ${BUILDDIR}/chase_mpi.x
