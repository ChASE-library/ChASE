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
      matrix `N` defining the eigenproblem, number of wanted
      eigenvalues `nev`, the public functions of this class
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
  //! Constructor for the ChaseConfig class
  /*!
      Requires the explicit values for the initalization of the size `N`
      of the matrix *A*, the number of sought after extremal
      eigenvalues `nev`, and the number of extra eigenvalue `nex` which
      defines, together with `nev`, the search space. All the other
      private members of the class are initialized using default values
      either specified directly (e.g. `max_iter`) or specified using a
      function that is part of the ChaseConfig namespace (e.g. `initMaxDeg`).

      \param _N Size of the square matrix defining the eigenproblem.
      \param _nev Number of desired extremal eigenvalues.
      \param _nex Number of eigenvalues augmenting the search space. Usually a relatively small fraction of `nev`.
   */
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

  //! Returns the value of the `approx_` flag.
  /*! The value of the `approx_` flag indicates if ChASE has been used
      with the engagement of approximate solutions as it is typical
      when solving for sequences of eigenvalue problems occurring in
      Density Functional Theory.
      \return The value of the `approx_` flag.
   */
  bool UseApprox() const { return approx_; }

  //! Sets the `approx_` flag  to either `'true'` or `'false'`.
  /*! This function is used to change the value of `approx_` so that
      the eigensolver can use approximate solutions inputed through a
      matrix of initial vectors.
      \param flag A boolean parameter which admits either a `'true'` or `'false'` value. 
   */
  void SetApprox(bool flag) { approx_ = flag; }

  //! Returns the value of the `optimization_` flag.
  /*! The value of the `optimization_` flag indicates when ChASE
      computes a polynomial degree optimized for each single desired
      eigenpairs. The optimization minimizes the number of operations
      required for the eigenpairs to have a residual which is just
      below the specified tolerance threshold.
      \return The value of the `optimization_` flag
   */
  bool DoOptimization() const { return optimization_; }

  //! Sets the `optimization_` flag to either `'true'` or `'false'`.
  /*! This function is used to change the value of `optimization_` so
      that the eigensolver minimizes the number of FLOPs needed to
      reach convergence for the entire sought after subspace of the
      spectrum.
      \param flag A boolean parameter which admits either a `'true'` or `'false'` value.
   */
  void SetOpt(bool flag) { optimization_ = flag; }

  //! Returns the degree of the Chebyshev filter used by ChASE
  /*!
      The value returned is the degree used by the filter when it is
      called (when ``optimization == 'true'`` this value is used only the
      first time the filter is called)
      \return The value used by the Chebyshev filter
   */
  std::size_t GetDeg() const { return deg_; }
  
  //! Set the value of the initial degree of the Chebyshev filter.
  /*!
      Depending if the `optimization` parameter is set to `false` or
      `true`, the value of `_deg` is used by the Chebyshev filter
      respectively every time or just the first time it is called.
      \param _deg Value set by the expert user and should in general be between *10* and *25*. The default value is *20*. If a odd value is inserted, the function makes it even. This is necessary due to the swapping of the matrix of vectors within the filter. It is strongly suggested to avoid values above the higher between *40* and the value returned by `GetMaxDeg`.
   */
  void SetDeg(std::size_t _deg) {
    deg_ = _deg;
    deg_ += deg_ % 2;
  }

  //! Returns the threshold value of the eigenpair's residual tolerance.
  /*! The value of the tolerance is used as threshold for all the
      residuals of the desired eigenpaits. Whenever an eigenpair's
      residual decreases below such a value it is declared as
      converged, and is consequently deflated and locked.
      \return The value of the `tol_` parameter.
   */
  double GetTol() const { return tol_; }

  //! Sets the value of the threshold of the eigenpair's residual tolerance.
  /*! The value of the tolerance should be set carefully keeping in
      mind that the residual of the eigenpairs is limited by the
      accuracy of the dense eigensolver used within the Rayleigh-Ritz
      procedure. As such it should hardly be set below *1e-14* in
      double precision. As a rule of thumb a minimum value of *1e-04*
      and *1e-08* should be used respectively in single and double
      precision.
      \param _tol A type double number usually specified in scientific notation (e.g. *1e-10*).
   */
  void SetTol(double _tol) { tol_ = _tol; }

  //! Returns the integer value of the maximum degree used by the polynomial filter.
  /*! The value of `max_deg_` indicates the upper bound for the degree
      of the polynomial for any of the vectors filtered. Such bound is
      important to avoid potential numerical instabilities that may
      occur and impede the convergence of the eigenpairs,
      expecially the one close to the upper end of the desired
      subspace of the spectrum.
      \return The value of the maximum degree of the Chebyshev filter.
   */
  std::size_t GetMaxDeg() const { return max_deg_; }
  
  //! Sets the maximum value of the degree of the Chebyshev filter
  /*! When ``optimization = true``, the Chebyshev filter degree is
      computed automatically. Because the computed values could be
      quite large for eigenvectors at the end of the sought after
      spectrum, a maximum value is set to avoid numerical
      instabilities that may trigger eigenpairs divergence.
      \param _maxDeg This value should be set by the expert user. It is set to *36* by default. It can be lowered in case of the onset of early instabilities but it should not be lower than *20-25* to avoid the filter becomes ineffective. It can be increased whenever it is known there is a spectral gap between the value of `nev_` and the value of `nev_ + nex_`. It is strongly suggested to never exceed the value of *70*.  
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

  ///////////////////////////////////////////////////
  // General parameters of the eigensolver
  //////////////////////////////////////////////////

  //! Rank of the matrix *A* defining the eigenproblem.
  /*!    This variable is initialized by the constructor using the value of the first
      of its input parameters `_N`. There is no default value.
   */
  std::size_t const N_;

  //! Number of desired extremal eigenpairs
  /*!
      This variable is initialized by the constructor using the value of the second
      of its input parameters `_nev`. There is no default value.
   */
  std::size_t const nev_;

  //! Increment of the search subspace so that its total size is `_nev + _nex`.
  /*!
      This variable is initialized by the constructor using the value of the third
      of its input parameters `_nex`. There is no default value.
   */
  std::size_t const nex_;

  //! An optional flag indicating if approximate eigenvectors are provided by the user.
  /*!
      This variable is initialized by the constructor. Its default value is set to `'false'`
   */  
  bool approx_;

  //! An optional parameters limiting the total number of internal while loops ChASE executes.
  /*!
      This variable is initialized by the constructor. Its default value is *25*
   */  
  std::size_t max_iter_;

  //! An optional parameters indicating the minimal value of the threshold below which the desired eigenpairs are declared converged.
  /*!
      This variable is initialized by the constructor. Its default
      value is set to *1e-10* and *1e-05* respectively in double and single
      precision.
   */  
  double tol_;

  ///////////////////////////////////////////////////
  // Chebyshev filter parameters
  //////////////////////////////////////////////////

  //! An optional parameter indicating the degree of the polynomial filter.
  /*!
      When the flag `optimization_` is set to `'true'`, its value is
      used only the first first time the filter routine is
      called. Otherwise this value is used for each vector and all
      subsequent subspace iterations. This variable is initialized by
      the constructor. Its default value is set to *20* and *10* in double
      and single precision, respectively.
   */  
  std::size_t deg_;

  //! An optional flag indicating if the filter uses a polynomial degree optimized for each single vector.
  /*!
      This variable is initialized by the constructor. Its default value is set to `'true'`
   */
  bool optimization_;

  //! An optional parameter that limits from above the value of the allowed polynomial filter.
  /*!
      When the flag `optimization_` is set to `'true'`, it avoids that
      a vector is filtered with a too high of a degree which may
      introduce numerical instabilities and slow or even impede
      convergence. This variable is initialized by the
      constructor. Its default value is set to *36* and *18* in double and
      single precision, respectively.
   */  
  std::size_t max_deg_;

  //! An optional parameter augmenting of few units the polynomial degree automatic computed by ChASE.
  /*!
      This parameter is exclusively used when the flag `optimization_`
      is set to `'true'` and should never be larger than a single
      digit. This variable is initialized by the constructor. Its
      default value is set to *2*.
   */  
  std::size_t deg_extra_;

  ///////////////////////////////////////////////////
  // Lanczos DoS parameters
  //////////////////////////////////////////////////

  //! Optional parameter indicating the total number of steps executed by the Lanczos DoS. 
  /*!
      This variable is initialized by the constructor. Its
      default value is set to *25*.
   */
  std::size_t lanczos_iter_;

  //! Optional parameter indicating the total number of vectors used for the vector estimate in the Lanczos DoS. 
  /*!
      This variable is initialized by the constructor. Its
      default value is set to *4*.
   */
  std::size_t num_lanczos_;

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
