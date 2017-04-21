build:
	module load Intel Boost || true
	icpc -DOUTPUT -std=c++11  -I../../testframework -I../../genera/blas  -I../../algorithm -qopenmp  main.cpp -mkl -lboost_program_options -lboost_serialization -o main.x

clean:
	rm -fr *.x

test: main.x
	ln -sf ~/gmat\ \ 1\ \ 1.bin ../../testframework/gmat\ \ 1\ \ 1.bin
	./main.x --n 3893 --mode "R" --opt "n" --name nacl --path_in ../../testframework/ --legacy on --nev 50 --bgn 1 --end 1 --nex 25
