#include <gtest/gtest.h>
#include <complex>
#include <random>
#include <cmath>
#include <cstring>
#include "linalg/internal/cpu/rayleighRitz.hpp"
#include "tests/linalg/internal/utils.hpp"

template <typename T>
class rayleighRitzCPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        H.resize(N * N);
        Q.resize(N * n);
        W.resize(N * n);
        ritzv.resize(n);
        evals.resize(N);
        resids.resize(N);
    }

    void TearDown() override {}

    std::size_t N = 50;
    std::size_t n = 10;
    std::vector<T> H;
    std::vector<T> Q;
    std::vector<T> W;
    std::vector<chase::Base<T>> ritzv;
    std::vector<chase::Base<T>> evals;
    std::vector<chase::Base<T>> resids;


};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(rayleighRitzCPUTest, TestTypes);

TYPED_TEST(rayleighRitzCPUTest, eigenpairs) {
    using T = TypeParam;  // Get the current type
    auto machineEpsilon = MachineEpsilon<T>::value();
    T One = T(1.0);
    T Zero = T(0.0);
    for (int i = 0; i < this->N; ++i) {
        this->H[i * this->N + i] = T(0.1 * i + 0.1);
    }

    std::mt19937 gen(1337.0);
    std::normal_distribution<> d;
    std::vector<T> V(this->N * this->N);
    std::vector<T> H2(this->N * this->N);

    for(auto i = 0; i < this->N * this->N; i++)
    {
        V[i] =  getRandomT<T>([&]() { return d(gen); });
    }

    std::unique_ptr<T[]> tau(new T[this->N]);

    chase::linalg::lapackpp::t_geqrf(LAPACK_COL_MAJOR, this->N, this->N, V.data(), this->N, tau.get());
    chase::linalg::lapackpp::t_gqr(LAPACK_COL_MAJOR, this->N, this->N, this->N, V.data(), this->N, tau.get());

    chase::linalg::blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, this->N, this->N, this->N,
               &One, this->H.data(), this->N, V.data(), this->N, &Zero,
               H2.data(), this->N);

    chase::linalg::blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasConjTrans, this->N, this->N, this->N,
               &One, V.data(), this->N, H2.data(), this->N, &Zero,
               this->H.data(), this->N);

    std::memcpy(H2.data(), this->H.data(), sizeof(T) * this->N * this->N);

    chase::linalg::lapackpp::t_heevd(CblasColMajor, 'V', 'U', this->N,
                    this->H.data(), this->N, this->evals.data());
    
    chase::linalg::internal::cpu::rayleighRitz(this->N, H2.data(), this->N, this->n, this->H.data(), this->N,
                        this->W.data(), this->N, this->ritzv.data());

    //check the eigenvalues
    for(auto i = 0; i < this->n; i++)
    {
        EXPECT_NEAR(this->ritzv[i], this->evals[i], 100 * machineEpsilon);
    }

    // check the residuals
    T beta;
    chase::linalg::blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, this->N, this->n, this->N,
            &One, H2.data(), this->N, this->W.data(), this->N, &Zero, this->Q.data(), this->N);

    for (std::size_t i = 0; i < this->n; ++i)
    {
        beta = -T(this->ritzv[i]);
        chase::linalg::blaspp::t_axpy(this->N, &beta, this->W.data() + this->N * i, 1,
                this->Q.data() + this->N * i, 1);

        this->resids[i] = chase::linalg::blaspp::t_nrm2(this->N, this->Q.data() + this->N * i, 1);
        EXPECT_NEAR(this->resids[i], machineEpsilon, machineEpsilon * 1e2);
    }    

}
