GPP=g++
GF=gfortran

GFFLAGS+=-cpp -ffree-line-length-none -ffree-form
GFLAGS+=-lstdc++

all: clean
	${GF} -c get_exciton.F ${GFFLAGS}
	${GPP} -c write.c
	${GF} -o get_exciton.x write.o get_exciton.o ${GFLAGS}

clean:
	rm -f get_exciton.o write.o get_exciton.x excivar.mod excihelper.mod
