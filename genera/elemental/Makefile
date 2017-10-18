include /usr/local/software/jureca/Stages/2017a/software/Elemental/0.87.7-intel-para-2017a/conf/ElVars

all: test

test: HermitianEig.cpp
	${CC} ${EL_COMPILE_FLAGS} $^ ${EL_LIBS} -o $@
