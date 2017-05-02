/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set ts=8 sts=2 et sw=2 tw=80: */
/* TODO License */
#pragma once

class ChASE_Config {
public:
    ChASE_Config(std::size_t _N, std::size_t _nev, std::size_t _nex)
        : N(_N)
        , nev(_nev)
        , nex(_nex)
        , optimization(false)
        , approx(false)
        , tol(1e-10)
        , deg(20)
    {
    }

    bool use_approx() { return approx; }

    bool do_optimization() { return optimization; }

    void setDeg(std::size_t _deg) { deg = _deg; }

    void setTol(double _tol) { tol = _tol; }

    void setOpt(bool flag) { optimization = flag; }

    double getTol() { return tol; }

    std::size_t getDeg() { return deg; }

    std::size_t getMaxDeg() { return 20; }

    std::size_t getDegExtra() { return 2; }

    std::size_t getMaxIter() { return 40; }

    std::size_t getLanczosIter() { return 30; }

    std::size_t getN() { return N; }

    std::size_t getNev() { return nev; }

    std::size_t getNex() { return nex; }

private:
    bool optimization;
    bool approx;
    std::size_t deg;

    std::size_t N, nev, nex;

    // not sure about this, would we ever need more?
    double tol;
};
