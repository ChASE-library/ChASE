// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#ifndef CHASE_ALGORITHM_CONFIGURATION_HPP
#define CHASE_ALGORITHM_CONFIGURATION_HPP

#include <complex>
#include <cstring>
#include <iomanip>
#include <random>

namespace chase
{

namespace chase_config_helper
{

template <typename T>
std::size_t initMaxDeg(bool approx, bool optimization);

template <typename T>
std::size_t initDeg(bool approx, bool optimization);

template <typename T>
std::size_t initLanczosIter(bool approx, bool optimization);

template <typename T>
double initTolerance(bool approx, bool optimization);

template <>
std::size_t initMaxDeg<double>(bool approx, bool optimization)
{
    return 36;
};

template <>
std::size_t initDeg<double>(bool approx, bool optimization)
{
    return 20;
};

template <>
std::size_t initLanczosIter<double>(bool approx, bool optimization)
{
    return 25;
};

template <>
double initTolerance<double>(bool approx, bool optimization)
{
    return 1e-10;
};

template <>
double initTolerance<float>(bool approx, bool optimization)
{
    return 1e-5;
}

template <>
std::size_t initMaxDeg<std::complex<double>>(bool approx, bool optimization)
{
    return initMaxDeg<double>(approx, optimization);
};

template <>
std::size_t initMaxDeg<float>(bool approx, bool optimization)
{
    return initMaxDeg<double>(approx, optimization) / 2;
};

template <>
std::size_t initMaxDeg<std::complex<float>>(bool approx, bool optimization)
{
    return initMaxDeg<double>(approx, optimization) / 2;
};

template <>
std::size_t initDeg<std::complex<double>>(bool approx, bool optimization)
{
    return initDeg<double>(approx, optimization);
};

template <>
std::size_t initDeg<float>(bool approx, bool optimization)
{
    return initDeg<double>(approx, optimization) / 2;
};

template <>
std::size_t initDeg<std::complex<float>>(bool approx, bool optimization)
{
    return initDeg<double>(approx, optimization) / 2;
};

template <>
std::size_t initLanczosIter<std::complex<double>>(bool approx,
                                                  bool optimization)
{
    return initLanczosIter<double>(approx, optimization);
};

template <>
std::size_t initLanczosIter<float>(bool approx, bool optimization)
{
    return initLanczosIter<double>(approx, optimization) / 2;
};

template <>
std::size_t initLanczosIter<std::complex<float>>(bool approx, bool optimization)
{
    return initLanczosIter<double>(approx, optimization) / 2;
};

template <>
double initTolerance<std::complex<double>>(bool approx, bool optimization)
{
    return initTolerance<double>(approx, optimization);
}

template <>
double initTolerance<std::complex<float>>(bool approx, bool optimization)
{
    return initTolerance<float>(approx, optimization);
}

static const std::size_t key_width = 30;
static const std::size_t val_width = 8;

template <typename T>
void pretty_print(std::ostream& oss, std::string key, T value)
{
    oss << "    " << std::left << std::setfill('.') << std::setw(key_width)
        << key << std::right << std::setw(val_width) << value << '\n';
}

} // namespace chase_config_helper

//! A class to set up all the parameters of the eigensolver
/*!
    Besides setting up the standard parameters such as size of the
    matrix `N_` defining the eigenproblem, number of wanted
    eigenvalues `nev_`, the public functions of this class
    initialize all internal parameters and allow the experienced
    user to set up the values of parameters of core functionalities
    (e.g. lanczos DoS). The aim is to influence the behavior of the
    library in special cases when the default values of the
    parameters return a suboptimal efficiency in terms of
    performance and/or accuracy.
 */
template <class T>
class ChaseConfig
{
public:
    //! Constructor for the ChaseConfig class
    /*!
        Requires the explicit values for the initalization of the size `N_`
        of the matrix *A*, the number of sought after extremal
        eigenvalues `nev_`, and the number of extra eigenvalue `nex_` which
        defines, together with `nev_`, the search space. All the other
        private members of the class are initialized using default values
        either specified directly (e.g. `max_iter_`) or specified using a
        function that is part of the ChaseConfig namespace (e.g. `initMaxDeg`).

        \param _N Size of the square matrix defining the eigenproblem.
        \param _nev Number of desired extremal eigenvalues.
        \param _nex Number of eigenvalues augmenting the search space. Usually a
       relatively small fraction of `nev`.
     */
    ChaseConfig(std::size_t _N, std::size_t _nev, std::size_t _nex)
        : N_(_N), nev_(_nev), nex_(_nex), optimization_(true), approx_(false),
          max_iter_(25), deg_extra_(2), num_lanczos_(4), decaying_rate_(1.0),
          upperb_scale_rate_(1.0), cluster_aware_degrees_(true)
    {
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

    //! Sets the `approx_` flag  to either `true` or `false`.
    /*! This function is used to change the value of `approx_` so that
        the eigensolver can use approximate solutions inputed through a
        matrix of initial vectors.
        \param flag A boolean parameter which admits either a `true` or `false`
       value.
     */
    bool DoOptimization() const { return optimization_; }

    //! Sets the `optimization_` flag to either `true` or `false`.
    /*! This function is used to change the value of `optimization_` so
        that the eigensolver minimizes the number of FLOPs needed to
        reach convergence for the entire sought after subspace of the
        spectrum.
        \param flag A boolean parameter which admits either a `true` or `false`
       value.
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
    void SetOpt(bool flag) { optimization_ = flag; }

    //! Returns the integer value of the maximum degree used by the polynomial
    //! filter.
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
    /*! When ``optimization_ == 'true'``, the Chebyshev filter degree is
        computed automatically. Because the computed values could be
        quite large for eigenvectors at the end of the sought after
        spectrum, a maximum value is set to avoid numerical
        instabilities that may trigger eigenpairs divergence.
        \param _maxDeg This value should be set by the expert user.
        It is set to *36* by default. It can be lowered in case of
        the onset of early instabilities but it should not be lower
        than *20-25* to avoid the filter becomes ineffective.
        It can be increased whenever it is known there is a spectral gap
        between the value of `nev_` and the value of `nev_ + nex_`.
        It is strongly suggested to never exceed the value of *70*.
     */
    void SetMaxDeg(std::size_t _maxDeg)
    {
        max_deg_ = _maxDeg;
        max_deg_ += max_deg_ % 2;
    }

    //! Returns the extra degree added to the polynomial filter.
    /*! When ``optimization_ == 'true'``, each vector is filtered with a
        polynomial of a given calculated degree. Because the degree is
        predicted based on an heuristic fomula, such degree is augmented
        by a small value to ensure that the residual of the
        corresponding vector will be almost always lower than the
        required threshold tolerance.
        \return The extra value added to a computed vector of polynomial
       degrees.
     */
    std::size_t GetDegExtra() const { return deg_extra_; }

    //! Sets the value of the extra degree added to the polynomial filter.
    /*! The value of `degExtra` should be a single digit number usually
        not exceeding *5* or *6*. The expert user can modify the default
        value (which is *2*) in those cases where the heuristic to
        automatically compute the vector of optimal degrees seems to
        underestimate the value of the degree necessary for the
        eigenpairs to be declared converged.
        \param degExtra Value of the extra degree.
     */
    void SetDegExtra(std::size_t degExtra) { deg_extra_ = degExtra; }

    //! Returns the value of the maximum number of subspace iterations allowed
    //! within ChASE.
    /*! In order to avoid that the eigensolver would runoff unchecked,
        ChASE is given a upper bound on the number of subspace iteration
        it can execute. This is a normal safety mechanism when the
        algorithm fails to converge some eigenpairs whose residuals
        would continue to oscillate without above the tolerance
        threshold.
        \return The value of the maximum number of subspace iterations allowed.
     */
    std::size_t GetMaxIter() const { return max_iter_; }

    //! Sets the value of the maximum number of subspace iterations within
    //! ChASE.
    /*! Typically ChASE requires a number of single digit iterations to
        converge. In extreme cases such number can grow up to *10* or *12*.
        An increasing number of iterations usually implies that
        the eigensolver is not used for the intended purpose or that the
        spectrum of eigenproblem tackled is particularly
        problematic. The default value of `maxIter` is set to *25*.
        \param maxIter Value of the maximum number of subspace iterations.
     */
    void SetMaxIter(std::size_t maxIter) { max_iter_ = maxIter; }

    //! Returns the degree of the Chebyshev filter used by ChASE
    /*!
        The value returned is the degree used by the filter when it is
        called (when ``optimization_ == 'true'`` this value is used only the
        first time the filter is called)
        \return The value used by the Chebyshev filter
     */
    std::size_t GetDeg() const { return deg_; }

    //! Set the value of the initial degree of the Chebyshev filter.
    /*!
        Depending if the `optimization_` parameter is set to `false` or
        `true`, the value of `_deg` is used by the Chebyshev filter
        respectively every time or just the first time it is called.
        \param _deg Value set by the expert user and should in general be
       between *10* and *25*. The default value is *20*. If a odd value is
       inserted, the function makes it even. This is necessary due to the
       swapping of the matrix of vectors within the filter. It is strongly
       suggested to avoid values above the higher between *40* and the value
        returned by \ref GetMaxDeg().
     */
    void SetDeg(std::size_t _deg)
    {
        deg_ = _deg;
        deg_ += deg_ % 2;
    }

    //! Returns the threshold value of the eigenpair's residual tolerance.
    /*! The value of the tolerance is used as threshold for all the
        residuals of the desired eigenpaits. Whenever an eigenpair's
        residual decreases below such a value it is declared as
        converged, and is consequently deflated and locked.
        For an eigenpair \f$(\lambda_i,V_i)\f$, the residual in ChASE
        is defined as the Euclidean norm: \f$||AV_i-\lambda_iV_i||_2\f$.
        \return The value of the `tol_` parameter.
     */
    double GetTol() const { return tol_; }

    //! Sets the value of the threshold of the eigenpair's residual tolerance.
    /*! The value of the tolerance should be set carefully keeping in
        mind that the residual of the eigenpairs is limited by the
        accuracy of the dense eigensolver used within the Rayleigh-Ritz
        procedure. As such it should hardly be set below \f$1e-14\f$ in
        double precision. As a rule of thumb a minimum value of *1e-04*
        and *1e-08* should be used respectively in single and double
        precision.
        \param _tol A type double number usually specified in scientific
       notation (e.g. *1e-10*).
     */
    void SetTol(double _tol) { tol_ = _tol; }

    //! Returns the number of Lanczos iterations executed by ChASE.
    /*! In order to estimate the spectral bounds, ChASE executes a
        limited number of Lanczos steps. These steps are then used to
        compute a spectral estimate based on the Density of State (DoS)
        algorithm.
        \return The total number of the Lanczos iterations used by the DoS
       algorithm.
     */
    std::size_t GetLanczosIter() const { return lanczos_iter_; }

    //! Sets the number of Lanczos iterations executed by ChASE.
    /*! For the DoS algorithm to work effectively without overburdening
        the eigensolver, the number of Lanczos iteration should be not
        less than *10* but also no more than *100* . ChASE does not need
        very precise spectral estimates because at each iteration such
        estimates are automatically improved by the approximate spectrum
        computed. This is the reason why the default value of
        `lanczos_iter_` is *25*.
        \param lanczosIter Value of the total number of Lanczos iterations
       executed by ChASE.
     */
    void SetLanczosIter(std::size_t lanczosIter)
    {
        lanczos_iter_ = lanczosIter;
    }

    //! Returns the number of stochastic vectors used for the spectral
    //! estimates.
    /*! After having executed a number of Lanczos steps, ChASE uses a
        cheap and efficient estimator to calculate the value of the upper
        extremum of the search space. Such an estimator uses a small
        number of stochastic vectors.
        \return Number of stochastic vectors used by ChASE for the spectral
       estimate.
     */
    std::size_t GetNumLanczos() const { return num_lanczos_; }

    //! Sets the number of stochastic vectors used for the spectral estimates.
    /*! The stochastic estimator used by ChASE is based on a cheap and
        efficient DoS algorithm. Because ChASE does not need precise
        estimates of the upper extremum of the search space, the number
        of vectors used is quite small. The default value used is *4*.
        The expert user can change the value to a larger number (it
        is not suggested to use a smaller value) and pay a slightly
        higher computing cost. It is not suggested to exceed a value for
        `num_lanczos_` higher than *20*.
        \param lanczosIter Number of stochastic vectors used by ChASE.
     */
    void SetNumLanczos(std::size_t lanczosIter) { num_lanczos_ = lanczosIter; }

    //! Returns the size of the eigenproblem
    /*! This function returns the size of the matrix defining the
        standard or the generalized eigenvalue problem.
        \return Rank of the matrix.
     */
    std::size_t GetN() const { return N_; }

    //! Returns the number of desired eigenpairs.
    /*! The number of sought after eigenpairs as also specified in the
        constructor ChASEConfig.
        \return Number of desired eigenpairs.
     */
    std::size_t GetNev() const { return nev_; }

    //! Returns the number of extra eigenpairs that are used to augment the
    //! search subspace.
    /*! ChASE effectively uses an enlarged subspace to improve the
        convergence of the algorithm. With a very small value of `nex_`,
        the eigenpairs at the end of the desired interval may have hard
        time to converge. By including a small but substantial number of
        extra values (in most case no more than 20% of `nev_`), ensures
        that ChASE converges smoothly without stagnating.
        \return Number of extra eigenpairs augmenting the search space.
     */
    std::size_t GetNex() const { return nex_; }

    //! Sets the `cholqr_` flag to either `true` or `false`.
    /*! This function is used to change the value of `cholqr_` so
        either flexible CholQR (`true`) or Househoulder QR (`false`)
        will be used.
        \param flag A boolean parameter which admits either a `true` or `false`
       value.
     */
    void SetCholQR(bool flag) { cholqr_ = flag; }
    //! Return the value of `cholqr_`
    bool DoCholQR() { return cholqr_; }

    void EnableSymCheck(bool flag) { sym_check_ = flag; }
    bool DoSymCheck() { return sym_check_; }

    //! Returns the decaying rate for the polynomial lower bound
    /*! The lower bound of the chebyshev lower bound is set based
     *  on an approximation of the eigenvalues by few iterations of lanczos.
     *  It might be better to use under estimation of the lower bound in
     *  certain cases, for instance if the target eigenvalues are packed.
     */
    float GetDecayingRate() const { return decaying_rate_; }

    //! Sets the decaying rate for the polynomial lower bound
    /*! The lower bound of the chebyshev lower bound is set based
     *  on an approximation of the eigenvalues by few iterations of lanczos.
     *  It might be better to use under estimation of the lower bound in
     *  certain cases, for instance if the target eigenvalues are packed.
     */
    void SetDecayingRate(float decayingRate) { decaying_rate_ = decayingRate; }

    //! Returns whether cluster-aware degree optimization is enabled
    /*! When enabled, the algorithm detects clusters of eigenvalues and
     *  adjusts polynomial degrees accordingly to improve convergence for
     *  clustered eigenvalues.
     */
    bool UseClusterAwareDegrees() const { return cluster_aware_degrees_; }

    //! Sets whether to use cluster-aware degree optimization
    /*! This enables/disables the cluster detection algorithm that adjusts
     *  polynomial degrees based on eigenvalue clustering patterns.
     */
    void SetClusterAwareDegrees(bool flag) { cluster_aware_degrees_ = flag; }

    //! Returns the scale rate for upperb based on its sign
    /*!
        This variable controls how upperb is scaled based on its sign.
        For positive upperb, it's multiplied by this rate.
        For negative upperb, it's multiplied by (2 - rate).
        Default value is 1.2.
     */
    float GetUpperbScaleRate() const { return upperb_scale_rate_; }

    //! Sets the scale rate for upperb based on its sign
    /*!
        This variable controls how upperb is scaled based on its sign.
        For positive upperb, it's multiplied by this rate.
        For negative upperb, it's multiplied by (2 - rate).
        Default value is 1.2.
     */
    void SetUpperbScaleRate(float upperbScaleRate)
    {
        upperb_scale_rate_ = upperbScaleRate;
    }

private:
    ///////////////////////////////////////////////////
    // General parameters of the eigensolver
    //////////////////////////////////////////////////

    //! Rank of the matrix *A* defining the eigenproblem.
    /*!    This variable is initialized by the constructor using the value of
       the first of its input parameters `_N`. There is no default value.
     */
    std::size_t const N_;

    //! Number of desired extremal eigenpairs
    /*!
        This variable is initialized by the constructor using the value of the
       second of its input parameters `_nev`. There is no default value.
     */
    std::size_t const nev_;

    //! Increment of the search subspace so that its total size is `_nev +
    //! _nex`.
    /*!
        This variable is initialized by the constructor using the value of the
       third of its input parameters `_nex`. There is no default value.
     */
    std::size_t const nex_;

    //! An optional parameters indicating the minimal value of the threshold
    //! below which the desired eigenpairs are declared converged.
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
        When the flag `optimization_` is set to `true`, its value is
        used only the first first time the filter routine is
        called. Otherwise this value is used for each vector and all
        subsequent subspace iterations. This variable is initialized by
        the constructor. Its default value is set to *20* and *10* in double
        and single precision, respectively.
     */
    std::size_t deg_;

    //! An optional flag indicating if the filter uses a polynomial degree
    //! optimized for each single vector.
    /*!
        This variable is initialized by the constructor. Its default value is
       set to `true`.
     */
    bool optimization_;

    //! An optional flag indicating if approximate eigenvectors are provided by
    //! the user.
    /*!
        This variable is initialized by the constructor. Its default value is
       set to `false`.
     */
    bool approx_;

    //! An optional parameters limiting the total number of internal while loops
    //! ChASE executes.
    /*!
        This variable is initialized by the constructor. Its default value is
       *25*
     */
    std::size_t max_iter_;

    //! An optional parameter that limits from above the value of the allowed
    //! polynomial filter.
    /*!
        When the flag `optimization_` is set to `true`, it avoids that
        a vector is filtered with a too high of a degree which may
        introduce numerical instabilities and slow or even impede
        convergence. This variable is initialized by the
        constructor. Its default value is set to *36* and *18* in double and
        single precision, respectively.
     */

    std::size_t max_deg_;

    //! An optional parameter augmenting of few units the polynomial degree
    //! automatic computed by ChASE.
    /*!
        This parameter is exclusively used when the flag `optimization_`
        is set to `true` and should never be larger than a single
        digit. This variable is initialized by the constructor. Its
        default value is set to *2*.
     */
    std::size_t deg_extra_;

    ///////////////////////////////////////////////////
    // Lanczos DoS parameters
    //////////////////////////////////////////////////

    //! Optional parameter indicating the total number of steps executed by the
    //! Lanczos DoS.
    /*!
        This variable is initialized by the constructor. Its
        default value is set to *25*.
     */
    std::size_t lanczos_iter_;

    //! Optional parameter indicating the total number of vectors used for the
    //! vector estimate in the Lanczos DoS.
    /*!
        This variable is initialized by the constructor. Its
        default value is set to *4*.
     */
    std::size_t num_lanczos_;

    //! Optional parameter indicating the decaying rate of the lower bound
    //! of the Chebyshev polynomial
    /*!
        This variable is initialized by the constructor. Its
        default value is set to *1.0* (no decaying rate).
     */
    float decaying_rate_;

    //! Optional parameter indicating if CholeksyQR is disabled
    bool cholqr_ = true;

    bool sym_check_ = true;

    //! Optional parameter indicating the scale rate for upperb based on its
    //! sign
    /*!
        This variable is initialized by the constructor. Its
        default value is set to 1.2.
        For positive upperb, it's multiplied by this rate.
        For negative upperb, it's multiplied by (2 - rate).
     */
    float upperb_scale_rate_;

    //! Optional parameter enabling cluster-aware degree optimization
    /*!
        When enabled, the algorithm detects eigenvalue clusters and adjusts
        polynomial degrees to improve convergence for clustered eigenvalues.
        This variable is initialized by the constructor. Its default value
        is set to true.
     */
    bool cluster_aware_degrees_;
};

template <typename T>
std::ostream& operator<<(std::ostream& oss_, const ChaseConfig<T>& rhs)
{
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
    pretty_print(oss, "# of Lanczos Vectors:", rhs.GetNumLanczos());
    pretty_print(oss, "Decaying Rate:", rhs.GetDecayingRate());
    pretty_print(oss, "Upperb Scale Rate:", rhs.GetUpperbScaleRate());
    pretty_print(oss, "Cluster-Aware Degrees:", rhs.UseClusterAwareDegrees());
    oss << "\n";

    oss_ << oss.str();
    return oss_;
};

} // namespace chase
#endif
