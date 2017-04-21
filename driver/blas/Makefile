all:
	icpc -DOUTPUT  -I../../testframework -I../../genera/blas  -I../../algorithm -qopenmp  main.cpp -mkl -lboost_program_options -lboost_serialization -o main.x
	./main.x --n 3893 --mode "R" --opt "n" --name nacl --path_in ../../testframework/ --legacy on --nev 50 --bgn 1 --end 1 --nex 25
