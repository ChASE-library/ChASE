# Ensure that the obj/ folder exists
MKDIR_P = mkdir -p
.PHONY: obj_dir
all: main

CXX=icpc
CFLAGS+=-O2 -std=c++11 -g
LDFLAGS+=-lboost_program_options -lboost_serialization

CFLAGS+=-qopenmp
LDFLAGS+=-mkl


#LDFLAGS+=-lgomp /usr/lib64/liblapack.a /usr/lib64/libblas.a  -lm

OBJS=obj/chase.o obj/main.o obj/lanczos.o obj/filter.o obj/testresult.o

obj_dir:
	${MKDIR_P} obj/

main: obj_dir ${OBJS}
	${CXX} ${OBJS} ${LDFLAGS} -o $@.x

obj/%.o: src/%.cpp
	${CXX} ${CFLAGS} -c $< -o $@

clean:
	rm -f obj/* main.x
