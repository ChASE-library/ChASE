BUILDDIR=../../build
TESTPROBLEMDIR=~


${BUILDDIR}/chase_blas.x: | mk_builddir
	icpc -DOUTPUT -std=c++11 -qopenmp \
	-I../../testframework -I../../genera/blas \
	-I../../algorithm \
	 main.cpp \
	-mkl -lboost_program_options -lboost_serialization \
	-o ${BUILDDIR}/chase_blas.x

${BUILDDIR}/chase_blas.done: ${BUILDDIR}/chase_blas.x
	${BUILDDIR}/chase_blas.x --n 3893 --mode "R" --opt "n" --name nacl --path_in ${TESTPROBLEMDIR}/ --legacy on --nev 50 --bgn 1 --end 1 --nex 25
	touch ${BUILDDIR}/chase_blas.done

clean:
	rm -fr ${BUILDDIR}/chase_blas.x

test: ${BUILDDIR}/chase_blas.done

mk_builddir:
	mkdir -p ${BUILDDIR}

build: ${BUILDDIR}/chase_blas.x
