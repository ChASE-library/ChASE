// This file is a part of the ChASE library.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials,
// Forschungszentrum Juelich GmbH, Germany.
// All rights reserved.
// ChASE is licensed under the 3-clause BSD license (BSD 2.0).
// https://github.com/ChASE-library/ChASE/
#include <iostream>
#include <fstream>
#include <complex>
#include <iomanip>
#include <typeinfo>
#include <vector>
#include <chrono>
#include <random>
#include <sys/stat.h>

#include "algorithm/types.hpp"

bool IsPathExist(const std::string &s)
{
  struct stat buffer;
  return (stat (s.c_str(), &buffer) == 0);
}

std::string getCmdOption(int argc, char* argv[], const std::string& option)
{
    std::string cmd;
     for( int i = 0; i < argc; ++i)
     {
          std::string arg = argv[i];
          if(0 == arg.find(option))
          {
	       cmd = argv[i + 1];
               return cmd;
          }
     }
     return cmd;
}

template<typename T>
std::vector<T> generateRandomVec(std::size_t size){
  std::chrono::high_resolution_clock::time_point start, end;
  std::chrono::duration<double> elapsed;

  start = std::chrono::high_resolution_clock::now();

  std::vector<T> rnd(size);
  std::mt19937 gen(1337.0);
  std::normal_distribution<> d;

  for (std::size_t k = 0; k < size; ++k){
    rnd[k] = getRandomT<T>([&]() { return d(gen); });
  }

  end = std::chrono::high_resolution_clock::now();

  elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

  std::string scalartype;
  std::string precision;

  if(sizeof(chase::Base<T>) == 8){
    precision = "double";
  }else if(sizeof(chase::Base<T>) == 4){
    precision = "single";
  }

  if(sizeof(T) / sizeof(chase::Base<T>) == 1){
    scalartype = "real";
  }else if(sizeof(T) / sizeof(chase::Base<T>) == 2){
    scalartype = "complex";
  }

  std::cout << "] generating matrix of size " << size << " in " << scalartype << " " << precision << " in " << elapsed.count() << "s.\n";

  return rnd;
}

template <typename T>
void wrtMatIntoBinary(T *H, std::string path_out, std::size_t size){
  std::chrono::high_resolution_clock::time_point start, end;
  std::chrono::duration<double> elapsed;

  start = std::chrono::high_resolution_clock::now();

  std::ostringstream problem(std::ostringstream::ate);
  problem << path_out;

  std::cout << "]> writing matrix into ";
  std::cout << problem.str();
  std::cout << " of size = " << size;

  auto outfile = std::fstream(problem.str().c_str(), std::ios::out | std::ios::binary);

  outfile.write((char*)&H[0], size * sizeof(T));

  end = std::chrono::high_resolution_clock::now();

  elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

  std::cout << " in " << elapsed.count() << "s.\n";
  outfile.close();
}

template <typename T>
void readMatFromBinary(T *H, std::string path_in, std::size_t size){
  std::ostringstream problem(std::ostringstream::ate);
  problem << path_in;

  std::cout << "]> start reading matrix from binary file ";
  std::cout << problem.str();
  std::cout << " of size = " << size << std::endl;

  std::ifstream infile(problem.str().c_str(), std::ios::binary);

  infile.read((char*)H, sizeof(T) * size);

  infile.close();
}


int main (int argc, char *argv[]){

    std::size_t N = 1000000;
    std::string path_out;
    //parser
    std::string N_str = getCmdOption(argc, argv, "--N");
    if(!N_str.empty()){
    	N = std::stoi(N_str);
    }
    path_out = getCmdOption(argc, argv, "--path_out");
    if(path_out.empty()){
    	path_out = "./tmp/";
    }

    if(!IsPathExist(path_out)){
        mkdir(path_out.c_str(), 0700);
    }

    std::ostringstream drnd_str, zrnd_str, srnd_str, crnd_str;
    drnd_str << path_out << "rnd_d.bin";
    zrnd_str << path_out << "rnd_z.bin";
    crnd_str << path_out << "rnd_c.bin";
    srnd_str << path_out << "rnd_s.bin";

    std::mt19937 gen(2342.0);
    std::normal_distribution<> d;
    
    std::vector<double> dbuf = generateRandomVec<double>(N);

    wrtMatIntoBinary<double>(dbuf.data(), drnd_str.str(), N);
    //
    std::vector<float> sbuf = generateRandomVec<float>(N);

    wrtMatIntoBinary<float>(sbuf.data(), srnd_str.str(), N);
    //
    std::vector<std::complex<double>> zbuf = generateRandomVec<std::complex<double>>(N);

    wrtMatIntoBinary<std::complex<double>>(zbuf.data(), zrnd_str.str(), N);
    //
    std::vector<std::complex<float>> cbuf = generateRandomVec<std::complex<float>>(N);

    wrtMatIntoBinary<std::complex<float>>(cbuf.data(), crnd_str.str(), N);

    return 0;
}
