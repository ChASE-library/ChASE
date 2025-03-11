// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <string>
#include <iostream>
#include <complex>

template<typename T>
struct TypeName;

template<>
struct TypeName<float> {
    static std::string Get() { return "float_"; }
    static std::string GetLong() { return "simple"; }
};

template<>
struct TypeName<double> {
    static std::string Get() { return "double_"; }
    static std::string GetLong() { return "double"; }
};

template<>
struct TypeName<std::complex<float>> {
    static std::string Get() { return "cfloat_"; }
    static std::string GetLong() { return "complex_simple"; }
};

template<>
struct TypeName<std::complex<double>> {
    static std::string Get() { return "cdouble_"; }
    static std::string GetLong() { return "complex_double"; }
};


template<typename T>
std::string GetQRFileName() {
    return "../QR_matrices/matrix_" + TypeName<T>::Get();
}

template<typename T>
std::string GetBSEPath() {
    return "../BSE_matrices";
}

template<typename T>
std::string GetBSETiny_MatrixFileName(){
    return GetBSEPath<T>() + "/tiny_random_quasihermitian_" + TypeName<T>::GetLong() + ".bin";
}

template<typename T>
std::string GetBSETiny_EigenvaluesFileName(){
    return GetBSEPath<T>() + "/eigenvalues_tiny_random_quasihermitian_" + TypeName<T>::GetLong() + ".bin";
}

template<typename T>
std::string GetBSETiny_SH_EigenvaluesFileName(){
    return GetBSEPath<T>() + "/SH_eigenvalues_tiny_random_quasihermitian_" + TypeName<T>::GetLong() + ".bin";
}

template<typename T>
std::string GetBSE_MatrixFileName(){
    return GetBSEPath<T>() + "/random_quasihermitian_" + TypeName<T>::GetLong() + ".bin";
}

template<typename T>
std::string GetBSE_EigenvaluesFileName(){
    return GetBSEPath<T>() + "/eigenvalues_random_quasihermitian_" + TypeName<T>::GetLong() + ".bin";
}

template<typename T>
std::string GetBSE_SH_EigenvaluesFileName(){
    return GetBSEPath<T>() + "/SH_eigenvalues_random_quasihermitian_" + TypeName<T>::GetLong() + ".bin";
}

template<typename T>
struct MachineEpsilon {
    static T value() { return std::numeric_limits<T>::epsilon(); }
};

template<typename T>
struct MachineEpsilon<std::complex<T>> {
    static T value() { return std::numeric_limits<T>::epsilon(); }
};

template <typename T>
void read_vectors(T* Vec, std::string path_in, 
                  std::size_t xoff, std::size_t xlen,
                  std::size_t M, std::size_t nevex, int rank)
{
    std::ifstream file(path_in.data(), std::ios::binary);

    if (!file) {
        std::cerr << "Unable to open file "<< path_in.data() << std::endl;
        return;
    }
    std::cout << "READING MATRIX: " << path_in << "\n";
    int j;
    for (j = 0; j < nevex; ++j) {
        file.seekg(((xoff) +  j*M) * sizeof(T));
        file.read(reinterpret_cast<char*>(Vec + xlen * j), xlen * sizeof(T));
    }

    file.close();
}
