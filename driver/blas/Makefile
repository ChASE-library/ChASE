build:
	module load Intel/2017.0.098-GCC-5.4.0  IntelMPI/2017.0.098 Boost/1.61.0-Python-2.7.12 imkl/2017.0.098 || true
	icpc -DOUTPUT -std=c++11  -I../../testframework -I../../genera/blas  -I../../algorithm -qopenmp  main.cpp -mkl -lboost_program_options -lboost_serialization -o main.x

clean:
	rm -fr *.x

test: main.x
	ln -sf ~/gmat\ \ 1\ \ 1.bin ../../testframework/gmat\ \ 1\ \ 1.bin
	./main.x --n 3893 --mode "R" --opt "n" --name nacl --path_in ../../testframework/ --legacy on --nev 50 --bgn 1 --end 1 --nex 25
