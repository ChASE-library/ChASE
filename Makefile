# Ensure that the obj/ folder exists
MKDIR_P = mkdir -p
BUILD_CUDA=1

CXX=icpc
CPPFLAGS+=-qopenmp

CFLAGS+=-O2 -std=c++11 -g
LDFLAGS+=-mkl

CUDACXX=nvcc
CUDAFLAGS+=

CUDA_INCLUDE_DIR=${CUDA_ROOT}/include
CUDA_LIBS_DIR=${CUDA_ROOT}/lib64

BOOST_INCLUDE_DIR=${BOOST_ROOT}/include
BOOST_LIBS_DIR=${BOOST_ROOT}/lib

################################################################################
CUDA_FILE_DIR=src
CPP_FILE_DIR=src
HEADER_FILES := $(wildcard include/*.h)
CPP_FILES := $(wildcard ${CPP_FILE_DIR}/*.cpp)

ifdef BUILD_CUDA
CUDA_FILES := $(wildcard ${CUDA_FILE_DIR}/*.cu)
CFLAGS+=-DCHASE_BUILD_CUDA
CFLAGS_CUDA=-I${CUDA_INCLUDE_DIR}
LDFLAGS_CUDA=-L${CUDA_LIBS_DIR} -lcudart -lcublas -lnvToolsExt
else
CUDA_FILES =
CFLAGS_CUDA=
LDFLAGS_CUDA=
endif


CFLAGS_BOOST:=-I${BOOST_INCLUDE_DIR}
LDFLAGS_BOOST=-L${BOOST_LIBS_DIR} -lboost_program_options -lboost_serialization

CFLAGS+=${CFLAGS_BOOST} ${CFLAGS_CUDA}
LDFLAGS+=${LDFLAGS_BOOST} ${LDFLAGS_CUDA}

CPPFLAGS+=${CFLAGS} ${FLAG_OPENMP}
CUDAFLAGS+=${CFLAGS}

CPP_OBJS := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))
CUDA_OBJS := $(addprefix obj/,$(notdir $(CUDA_FILES:.cu=.o)))
OBJS=${CUDA_OBJS} ${CPP_OBJS}

all: main

${OBJS}: ${HEADER_FILES}

main: obj_dir ${OBJS}
	${CXX} ${OBJS} ${LDFLAGS} -o $@.x

obj/%.o: ${CPP_FILE_DIR}/%.cpp
	${CXX} ${CPPFLAGS} -c $< -o $@

obj/%.o: ${CUDA_FILE_DIR}/%.cu
	${CUDACXX} ${CUDAFLAGS}  -c $< -o $@

.PHONY: obj_dir clean test

obj_dir:
	${MKDIR_P} obj/

clean:
	rm -f obj/* main.x
