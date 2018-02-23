/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set ts=8 sts=2 et sw=2 tw=80: */
/* TODO License */
#pragma once

#include <complex>
#include <cstring>  //memcpy
#include <iomanip>
#include <random>

namespace chase {

namespace chase_config_helper {
template <typename T>
std::size_t initMaxDeg(bool approx, bool optimization);

template <typename T>
std::size_t initDeg(bool approx, bool optimization);

template <typename T>
std::size_t initLanczosIter(bool approx, bool optimization);

template <typename T>
double initTolerance(bool approx, bool optimization);

template <>
std::size_t initMaxDeg<double>(bool approx, bool optimization) {
  return 36;
};

template <>
std::size_t initDeg<double>(bool approx, bool optimization) {
  return 20;
};

template <>
std::size_t initLanczosIter<double>(bool approx, bool optimization) {
  return 25;
};

template <>
double initTolerance<double>(bool approx, bool optimization) {
  return 1e-10;
};

template <>
double initTolerance<float>(bool approx, bool optimization) {
  return 1e-5;
}

template <>
std::size_t initMaxDeg<std::complex<double> >(bool approx, bool optimization) {
  return initMaxDeg<double>(approx, optimization);
};

template <>
std::size_t initMaxDeg<float>(bool approx, bool optimization) {
  return initMaxDeg<double>(approx, optimization) / 2;
};

template <>
std::size_t initMaxDeg<std::complex<float> >(bool approx, bool optimization) {
  return 12;
  return initMaxDeg<double>(approx, optimization) / 2;
};

template <>
std::size_t initDeg<std::complex<double> >(bool approx, bool optimization) {
  return initDeg<double>(approx, optimization);
};

template <>
std::size_t initDeg<float>(bool approx, bool optimization) {
  return initDeg<double>(approx, optimization) / 2;
};

template <>
std::size_t initDeg<std::complex<float> >(bool approx, bool optimization) {
  return initDeg<double>(approx, optimization) / 2;
};

template <>
std::size_t initLanczosIter<std::complex<double> >(bool approx,
                                                   bool optimization) {
  return initLanczosIter<double>(approx, optimization);
};

template <>
std::size_t initLanczosIter<float>(bool approx, bool optimization) {
  return initLanczosIter<double>(approx, optimization) / 2;
};

template <>
std::size_t initLanczosIter<std::complex<float> >(bool approx,
                                                  bool optimization) {
  return initLanczosIter<double>(approx, optimization) / 2;
};

template <>
double initTolerance<std::complex<double> >(bool approx, bool optimization) {
  return initTolerance<double>(approx, optimization);
}

template <>
double initTolerance<std::complex<float> >(bool approx, bool optimization) {
  return initTolerance<float>(approx, optimization);
}

static const std::size_t key_width = 30;
static const std::size_t val_width = 8;

template <typename T>
void pretty_print(std::ostream& oss, std::string key, T value) {
  oss << "    " << std::left << std::setfill('.') << std::setw(key_width) << key
      << std::right << std::setw(val_width) << value << '\n';
}

}  // namespace chase_config_helper

template <class T>
class ChaseConfig {
 public:
  ChaseConfig(std::size_t _N, std::size_t _nev, std::size_t _nex)
      : N(_N),
        nev(_nev),
        nex(_nex),
        optimization(false),
        approx(false),
        mMaxIter(25),
        mDegExtra(2),
        num_lanczos_(6) {
    mMaxDeg = chase_config_helper::initMaxDeg<T>(approx, optimization);
    deg = chase_config_helper::initDeg<T>(approx, optimization);
    mLanczosIter =
        chase_config_helper::initLanczosIter<T>(approx, optimization);
    tol = chase_config_helper::initTolerance<T>(approx, optimization);

    // TODO: we require even degree!
    // matrixFree/blasSkewed requires it
  }

  bool use_approx() const { return approx; }
  void setApprox(bool flag) { approx = flag; }

  bool do_optimization() const { return optimization; }
  void setOpt(bool flag) { optimization = flag; }

  std::size_t getDeg() const { return deg; }
  void setDeg(std::size_t _deg) { deg = _deg; }

  double getTol() const { return tol; }
  void setTol(double _tol) { tol = _tol; }

  std::size_t getMaxDeg() const { return mMaxDeg; }
  void setMaxDeg(std::size_t maxDeg_) { mMaxDeg = maxDeg_; }

  std::size_t getDegExtra() const { return mDegExtra; }
  void setDegExtra(std::size_t degExtra) { mDegExtra = degExtra; }

  std::size_t getMaxIter() const { return mMaxIter; }
  void setMaxIter(std::size_t maxIter) { mMaxIter = maxIter; }

  std::size_t getLanczosIter() const { return mLanczosIter; }
  void setLanczosIter(std::size_t lanczosIter) { mLanczosIter = lanczosIter; }

  std::size_t getNumLanczos() const { return num_lanczos_; }
  void setNumLanczos(std::size_t lanczosIter) { num_lanczos_ = lanczosIter; }

  std::size_t getN() const { return N; }

  std::size_t getNev() const { return nev; }

  std::size_t getNex() const { return nex; }

 private:
  std::size_t const N, nev, nex;

  bool optimization;
  bool approx;
  std::size_t deg;

  std::size_t mDegExtra;
  std::size_t mMaxIter;
  std::size_t mLanczosIter;
  std::size_t mMaxDeg;
  std::size_t num_lanczos_;

  double tol;
};

template <typename T>
std::ostream& operator<<(std::ostream& oss_, const ChaseConfig<T>& rhs) {
  using namespace chase_config_helper;
  std::ostringstream oss;

  oss << "ChASE Configuration:\n";
  oss << "  "
      << "General Parameters"
      << "\n";
  pretty_print(oss, "N:", rhs.getN());
  pretty_print(oss, "nev:", rhs.getNev());
  pretty_print(oss, "nex:", rhs.getNex());
  pretty_print(oss, "Optimize Degree?", rhs.do_optimization());
  pretty_print(oss, "Have approximate Solution?", rhs.use_approx());
  pretty_print(oss, "Target residual tolerance:", rhs.getTol());
  pretty_print(oss, "Max # of Iterations:", rhs.getMaxIter());
  oss << "  "
      << "Filter Parameters"
      << "\n";
  pretty_print(oss, "Initial filter degree:", rhs.getDeg());
  pretty_print(oss, "Extra filter degree:", rhs.getDegExtra());
  pretty_print(oss, "Maximum filter degree:", rhs.getMaxDeg());
  oss << "  "
      << "Parameters for Spectral Estimates"
      << "\n";
  pretty_print(oss, "# of Lanczos Iterations:", rhs.getLanczosIter());
  oss << "\n";

  oss_ << oss.str();
  return oss_;
};

}  // namespace chase
