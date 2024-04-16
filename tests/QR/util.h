#pragma once

#include <string>
#include <iostream>
#include <complex>

template<typename T>
struct TypeName;

template<>
struct TypeName<float> {
    static std::string Get() { return "float_"; }
};

template<>
struct TypeName<double> {
    static std::string Get() { return "double_"; }
};

template<>
struct TypeName<std::complex<float>> {
    static std::string Get() { return "cfloat_"; }
};

template<>
struct TypeName<std::complex<double>> {
    static std::string Get() { return "cdouble_"; }
};


template<typename T>
std::string GetFileName() {
    return "QR_matrices/matrix_" + TypeName<T>::Get();
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
    }
    std::cout << "READING MATRIX: " << path_in << "\n";
    int i,j;
    for (j = 0; j < nevex; ++j) {
        file.seekg(((xoff) +  j*M) * sizeof(T));
        file.read(reinterpret_cast<char*>(Vec + xlen * j), xlen * sizeof(T));
    }

    file.close();
}
