#pragma once

#include <mpi.h>

namespace chase
{
namespace mpi
{
    template <typename T>
    MPI_Datatype getMPI_Type();

    template <>
    MPI_Datatype getMPI_Type<float>()
    {
        return MPI_FLOAT;
    }

    template <>
    MPI_Datatype getMPI_Type<double>()
    {
        return MPI_DOUBLE;
    }

    template <>
    MPI_Datatype getMPI_Type<std::complex<float>>()
    {
        return MPI_COMPLEX;
    }

    template <>
    MPI_Datatype getMPI_Type<std::complex<double>>()
    {
        return MPI_DOUBLE_COMPLEX;
    }

    
} // namespace Impl

    
} // namespace chase

