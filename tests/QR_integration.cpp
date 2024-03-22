#include <limits>

#include <gtest/gtest.h>

#include "ChASE-MPI/chase_mpi.hpp"
#include "algorithm/performance.hpp"

#include "ChASE-MPI/impl/chase_mpidla_blaslapack_seq.hpp"
#include "ChASE-MPI/impl/chase_mpidla_blaslapack_seq_inplace.hpp"

#include "ChASE-MPI/impl/chase_mpidla_blaslapack.hpp"

#include "util.h"

using namespace chase::mpi;

template <typename T>
class QRfixture : public testing::Test {
    protected:
    void SetUp() override {   
    
        properties = new ChaseMpiProperties<T>(N, nev, nex, MPI_COMM_WORLD);
        n = properties->get_n();
        m = properties->get_m();

        properties->get_off(&xoff, &yoff, &xlen, &ylen);
        column_comm = properties->get_col_comm();
        
        MPI_Comm_rank(column_comm, &column_rank);

        H.resize(n*m);
        V1.resize(m*(nev+nex));
        ritzv.resize(nev+nex);

        DLAblaslapack = new ChaseMpiDLABlaslapack<T>(properties, H.data(), m, V1.data(), ritzv.data());
        DLA = new ChaseMpiDLA<T>(properties, DLAblaslapack);
        chasempi = new ChaseMpi<ChaseMpiDLABlaslapack, T>(properties, H.data(), m, V1.data(), ritzv.data());
    }

    void TearDown() override {
        delete properties;
        delete DLA;
    }

    ChaseMpiProperties<T>* properties;
    ChaseMpiDLABlaslapack<T>* DLAblaslapack;
    ChaseMpiDLA<T>* DLA;
    ChaseMpi<ChaseMpiDLABlaslapack, T>* chasempi;

    // I need ChaseMpiProperties class for constructor of chase_
    std::size_t N   = 100;
    std::size_t nev = 30;
    std::size_t nex = 20;
    std::size_t n,m;

    std::size_t xoff;
    std::size_t yoff;
    std::size_t xlen;
    std::size_t ylen;

    // memory allocation
    std::vector<T> H;
    std::vector<T> V1;
    std::vector<chase::Base<T>> ritzv;

    MPI_Comm column_comm;
    int column_rank;
};

typedef ::testing::Types<float, double, std::complex<float>, std::complex<double>> MyTypes;
TYPED_TEST_SUITE(QRfixture, MyTypes);

TYPED_TEST(QRfixture, NumberOfProcs)
{
    ASSERT_EQ(this->chasempi->get_nprocs(), 4);    
}

TYPED_TEST(QRfixture, firsttest)
{
    auto machineEpsilon = MachineEpsilon<TypeParam>::value();

    read_vectors(this->V1.data(), GetFileName<TypeParam>() + "cond_10.bin", this->xoff, this->xlen, this->N, this->nev + this->nex, this->column_rank);
    this->chasempi->QR(0,0);
}
