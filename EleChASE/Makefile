ifdef EL_HOME
 include ${EL_HOME}/conf/ElVars
else
 include ${HOME}/libs/conf/ElVars
endif

ifndef FLEUR_INSTALL
  FLEUR_INSTALL=~/FleurChase/libs/lib/
endif

# Ensure that the obj/ folder exists
MKDIR_P = mkdir -p
.PHONY: obj_dir install_dir
all: libchase.a filter main

obj_dir:
	${MKDIR_P} obj/

install_dir:
	${MKDIR_P} ${FLEUR_INSTALL}

main: obj_dir obj/chase.o obj/main.o
	${CXX} ${EL_COMPILE_FLAGS} obj/main.o obj/chase.o -o $@ ${EL_LINK_FLAGS} ${EL_LIBS}

libchase.a: obj_dir obj/chase.o obj/elechfsi_fleur.o
	ar -rcs $@ obj/chase.o obj/elechfsi_fleur.o

obj/chase.o: src/chase.cpp
	${CXX} ${EL_COMPILE_FLAGS} -c $< -o $@

obj/main.o: src/main.cpp
	${CXX} ${EL_COMPILE_FLAGS} -c $< -o $@

obj/elechfsi_fleur.o: src/elechfsi_fleur.cpp
	${CXX} ${EL_COMPILE_FLAGS} -c $< -o $@

filter: src/driver_filter.cpp
	${CXX} ${EL_COMPILE_FLAGS} $< -o $@ ${EL_LINK_FLAGS} ${EL_LIBS}

clean:
	rm -f obj/* main filter libchase.a

obj/chase_f.o: chase.F90
	gfortran -c chase.F90  -o $@

install: install_dir
	cp libchase.a ${FLEUR_INSTALL}
