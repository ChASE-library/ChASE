/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2018, Simulation Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany
// and
// Copyright (c) 2016-2018, Aachen Institute for Advanced Study in Computational
//   Engineering Science, RWTH Aachen University, Germany All rights reserved.
// License is 3-clause BSD:
// https://github.com/SimLabQuantumMaterials/ChASE/

#ifndef CHASE_ALGORITHM_CONFIGURATION_HPP
#define CHASE_ALGORITHM_CONFIGURATION_HPP

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

  //! A class to set up all the parameters of the eigensolver
  /*!
      Besides setting up the standard parameters such as size of the
      matrix `_N` defining the eigenproblem, number of wanted
      eigenvalues `_nev`, the public functions of this class
      initialize all internal parameters and allow the experienced
      user to set up the values of parameters of core functionalities
      (e.g. lanczos DoS). The aim is to influence the behavior of the
      library in special cases when the default values of the
      parameters return a suboptimal efficiency in terms of
      performance and/or accuracy.
   */
template <class T>
class ChaseConfig {
 public:
  ChaseConfig(std::size_t _N, std::size_t _nev, std::size_t _nex)
      : N_(_N),
        nev_(_nev),
        nex_(_nex),
        optimization_(true),
        approx_(false),
        max_iter_(25),
        deg_extra_(2),
        num_lanczos_(4) {
    SetMaxDeg(chase_config_helper::initMaxDeg<T>(approx_, optimization_));
    SetDeg(chase_config_helper::initDeg<T>(approx_, optimization_));
    lanczos_iter_ =
        chase_config_helper::initLanczosIter<T>(approx_, optimization_);
    tol_ = chase_config_helper::initTolerance<T>(approx_, optimization_);
  }

  bool UseApprox() const { return approx_; }
  void SetApprox(bool flag) { approx_ = flag; }

  bool DoOptimization() const { return optimization_; }
  void SetOpt(bool flag) { optimization_ = flag; }

  //! Gets the degree of the Chebyshev filter used by ChASE
  /*!
      The value returned is the degree used by the filter when it is
      called (when ``optimization == 'true'`` this value is used only the
      first time the filter is called)
      \return The value used by the Chebyshev filter
   */
  std::size_t GetDeg() const { return deg_; }
  //! Set the value of the initial degree of the Chebyshev filter.
  /*!
      Depending if the `optimization` parameter is set to `false` or `true`
      (it is `true` by default), the value of `_deg` is used by the Chebyshev
      filter respectively every time or just the first time it is called.
      \param _deg Value set by the expert user and should in general be between *10* and *25*. The default value is *20*. If a odd value is inserted, the function makes it even. This is necessary due to the swapping of the matrix of vectors within the filter. It is strongly suggested to avoid values above the higher between *40* and the value returned by `GetMaxDeg`.
      \sa GetMaxDeg
   */
  void SetDeg(std::size_t _deg) {
    deg_ = _deg;
    deg_ += deg_ % 2;
  }

  double GetTol() const { return tol_; }
  void SetTol(double _tol) { tol_ = _tol; }

  std::size_t GetMaxDeg() const { return max_deg_; }
  //! Set the maximum value of the degree of the Chebyshev filter
  /*!
      When ``optimization = true``, the Chebyshev filter degree is computed automatically. Because the computed values could be quite large for eigenvectors at the end of the sought after spectrum, a maximum value is set to avoid numerical instabilities that may trigger eigenpairs divergence.
      \param _maxDeg This value should be set by the expert user. It is set to 36 by default. It can be lowered in case of the onset of early instabilities but it should not be lower than 20-25 to avoid the filter becomes ineffective. It can be increased whenever it is known there is a spectral gap between the value of `nev_` and the value of `nev_ + nex_`. It is strongly suggested to never exceed the value of 70.  
   */
  void SetMaxDeg(std::size_t _maxDeg) {
    max_deg_ = _maxDeg;
    max_deg_ += max_deg_ % 2;
  }

  std::size_t GetDegExtra() const { return deg_extra_; }
  void SetDegExtra(std::size_t degExtra) { deg_extra_ = degExtra; }

  std::size_t GetMaxIter() const { return max_iter_; }
  void SetMaxIter(std::size_t maxIter) { max_iter_ = maxIter; }

  std::size_t GetLanczosIter() const { return lanczos_iter_; }
  void SetLanczosIter(std::size_t lanczosIter) { lanczos_iter_ = lanczosIter; }

  std::size_t GetNumLanczos() const { return num_lanczos_; }
  void SetNumLanczos(std::size_t lanczosIter) { num_lanczos_ = lanczosIter; }

  std::size_t GetN() const { return N_; }

  std::size_t GetNev() const { return nev_; }

  std::size_t GetNex() const { return nex_; }

 private:
  std::size_t const N_, nev_, nex_;

  bool optimization_;
  bool approx_;
  std::size_t deg_;

  std::size_t deg_extra_;
  std::size_t max_iter_;
  std::size_t lanczos_iter_;
  std::size_t max_deg_;
  std::size_t num_lanczos_;

  double tol_;
};

template <typename T>
std::ostream& operator<<(std::ostream& oss_, const ChaseConfig<T>& rhs) {
  using namespace chase_config_helper;
  std::ostringstream oss;

  oss << "ChASE Configuration:\n";
  oss << "  "
      << "General Parameters"
      << "\n";
  pretty_print(oss, "N:", rhs.GetN());
  pretty_print(oss, "nev:", rhs.GetNev());
  pretty_print(oss, "nex:", rhs.GetNex());
  pretty_print(oss, "Optimize Degree?", rhs.DoOptimization());
  pretty_print(oss, "Have approximate Solution?", rhs.UseApprox());
  pretty_print(oss, "Target residual tolerance:", rhs.GetTol());
  pretty_print(oss, "Max # of Iterations:", rhs.GetMaxIter());
  oss << "  "
      << "Filter Parameters"
      << "\n";
  pretty_print(oss, "Initial filter degree:", rhs.GetDeg());
  pretty_print(oss, "Extra filter degree:", rhs.GetDegExtra());
  pretty_print(oss, "Maximum filter degree:", rhs.GetMaxDeg());
  oss << "  "
      << "Parameters for Spectral Estimates"
      << "\n";
  pretty_print(oss, "# of Lanczos Iterations:", rhs.GetLanczosIter());
  oss << "\n";

  oss_ << oss.str();
  return oss_;
};

}  // namespace chase
#endif
