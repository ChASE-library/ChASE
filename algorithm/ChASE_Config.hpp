#pragma once

class ChASE_Config {
public:
  ChASE_Config() :
    optimization(false),
    approx(false),
    tol(1e-10),
    deg(20)
  {}

  bool use_approx() {
    return approx;
  }

  bool do_optimization() {
    return optimization;
  }

  void setDeg( std::size_t _deg ) {
    deg = _deg;
  }

  void setTol( double _tol ) {
    tol = _tol;
  }

  double getTol() {
    return tol;
  }

  std::size_t getDeg() {
    return deg;
  }

  std::size_t getMaxDeg() {
    return 30;
  }

  std::size_t getDegExtra() {
    return 2;
  }

  std::size_t getMaxIter() {
    return 40;
  }

  std::size_t getLanczosIter() {
    return 20;
  }

private:
  bool optimization;
  bool approx;
  std::size_t deg;

  // not sure about this, would we ever need more?
  double tol;
};
