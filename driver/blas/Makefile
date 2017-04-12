all:
	rm -f *.o
	icpc -DOUTPUT  -I../../testframework -I../../genera/blas  -I../../algorithm -qopenmp  main.cpp -c -o main.o
	icpc -DOUTPUT -I../../testframework -I../../genera/blas  -I../../algorithm -qopenmp  ../../genera/blas/template_wrapper.cpp -c -o twrapper.o
	icpc -DOUTPUT -I../../testframework -I../../genera/blas  -I../../algorithm -qopenmp  ../../testframework/testresult.cpp -c -o test.o
	icpc -DOUTPUT main.o test.o twrapper.o -mkl -lboost_program_options -lboost_serialization -o main.x
	./main.x --n 3893 --mode "R" --opt "n" --name nacl --path_in ../../testframework/ --legacy on --nev 50 --bgn 1 --end 1 --nex 25
