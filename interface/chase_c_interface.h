// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#ifndef CHASE_C_INTERFACE_H
#define CHASE_C_INTERFACE_H

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

void dchase_init_(int* N, int* nev, int* nex, double* H, int *ldh, double* V,
                    double* ritzv, int* init);
void schase_init_(int* N, int* nev, int* nex, float* H,  int *ldh, float* V,
                    float* ritzv, int* init);
void cchase_init_(int* N, int* nev, int* nex, float _Complex* H, int *ldh,
                    float _Complex* V, float* ritzv, int* init);
void zchase_init_(int* N, int* nev, int* nex, double _Complex* H, int *ldh,
                    double _Complex* V, double* ritzv, int* init);

void dchase_finalize_(int* flag);
void schase_finalize_(int* flag);
void cchase_finalize_(int* flag);
void zchase_finalize_(int* flag);
void dchase_(int* deg, double* tol, char* mode, char* opt, char *qr);    
void schase_(int* deg, float* tol, char* mode, char* opt, char *qr);     
void zchase_(int* deg, double* tol, char* mode, char* opt, char *qr);
void cchase_(int* deg, float* tol, char* mode, char* opt, char *qr);

// Sequential pseudo-Hermitian initialization functions
void cchase_init_pseudo_(int* N, int* nev, int* nex, float _Complex* H, int *ldh,
                    float _Complex* V, float* ritzv, int* init);
void zchase_init_pseudo_(int* N, int* nev, int* nex, double _Complex* H, int *ldh,
                    double _Complex* V, double* ritzv, int* init);

void cchase_pseudo_(int* deg, float* tol, char* mode, char* opt, char *qr);
void zchase_pseudo_(int* deg, double* tol, char* mode, char* opt, char *qr);

// BlockCyclic matrix initialization functions
void pdchase_init_blockcyclic_(int* N, int* nev, int* nex, int* mbsize,
                                int* nbsize, double* H, int* ldh, double* V,
                                double* ritzv, int* dim0, int* dim1,
                                char* grid_major, int* irsrc, int* icsrc,
                                MPI_Comm* comm, int* init);

void pschase_init_blockcyclic_(int* N, int* nev, int* nex, int* mbsize,
                                int* nbsize, float* H, int* ldh, float* V,
                                float* ritzv, int* dim0, int* dim1,
                                char* grid_major, int* irsrc, int* icsrc,
                                MPI_Comm* comm, int* init);

void pcchase_init_blockcyclic_(int* N, int* nev, int* nex, int* mbsize,
                                int* nbsize, float _Complex* H, int* ldh,
                                float _Complex* V, float* ritzv, int* dim0,
                                int* dim1, char* grid_major, int* irsrc,
                                int* icsrc, MPI_Comm* comm, int* init);

void pzchase_init_blockcyclic_(int* N, int* nev, int* nex, int* mbsize,
                                int* nbsize, double _Complex* H, int* ldh,
                                double _Complex* V, double* ritzv, int* dim0,
                                int* dim1, char* grid_major, int* irsrc,
                                int* icsrc, MPI_Comm* comm, int* init);

// PseudoHermitian BlockCyclic matrix initialization functions
void pcchase_init_pseudo_blockcyclic_(int* N, int* nev, int* nex, int* mbsize,
                                int* nbsize, float _Complex* H, int* ldh,
                                float _Complex* V, float* ritzv, int* dim0,
                                int* dim1, char* grid_major, int* irsrc,
                                int* icsrc, MPI_Comm* comm, int* init);

void pzchase_init_pseudo_blockcyclic_(int* N, int* nev, int* nex, int* mbsize,
                                int* nbsize, double _Complex* H, int* ldh,
                                double _Complex* V, double* ritzv, int* dim0,
                                int* dim1, char* grid_major, int* irsrc,
                                int* icsrc, MPI_Comm* comm, int* init);

// BlockBlock matrix initialization functions
void pdchase_init_(int* N, int* nev, int* nex, int* m, int* n, double* H,
                    int* ldh, double* V, double* ritzv, int* dim0, int* dim1,
                    char* grid_major, MPI_Comm* comm, int* init);

void pschase_init_(int* N, int* nev, int* nex, int* m, int* n, float* H,
                    int* ldh, float* V, float* ritzv, int* dim0, int* dim1,
                    char* grid_major, MPI_Comm* comm, int* init);

void pcchase_init_(int* N, int* nev, int* nex, int* m, int* n,
                    float _Complex* H, int* ldh, float _Complex* V,
                    float* ritzv, int* dim0, int* dim1, char* grid_major,
                    MPI_Comm* comm, int* init);

void pzchase_init_(int* N, int* nev, int* nex, int* m, int* n,
                    double _Complex* H, int* ldh, double _Complex* V,
                    double* ritzv, int* dim0, int* dim1, char* grid_major,
                    MPI_Comm* comm, int* init);

// PseudoHermitian BlockBlock matrix initialization functions
void pcchase_init_pseudo_(int* N, int* nev, int* nex, int* m, int* n,
                    float _Complex* H, int* ldh, float _Complex* V,
                    float* ritzv, int* dim0, int* dim1, char* grid_major,
                    MPI_Comm* comm, int* init);

void pzchase_init_pseudo_(int* N, int* nev, int* nex, int* m, int* n,
                    double _Complex* H, int* ldh, double _Complex* V,
                    double* ritzv, int* dim0, int* dim1, char* grid_major,
                    MPI_Comm* comm, int* init);
void pdchase_finalize_(int* flag);
void pschase_finalize_(int* flag);
void pcchase_finalize_(int* flag);
void pzchase_finalize_(int* flag);

void pdchase_(int* deg, double* tol, char* mode, char* opt, char *qr);
void pschase_(int* deg, float* tol, char* mode, char* opt, char *qr);
void pcchase_(int* deg, float* tol, char* mode, char* opt, char *qr);
void pzchase_(int* deg, double* tol, char* mode, char* opt, char *qr);

void pschase_wrtHam_(const char* filename);
void pdchase_wrtHam_(const char* filename);
void pcchase_wrtHam_(const char* filename);
void pzchase_wrtHam_(const char* filename);

void pschase_readHam_(const char* filename);
void pdchase_readHam_(const char* filename);
void pcchase_readHam_(const char* filename);
void pzchase_readHam_(const char* filename);

// Aliases without leading 'p' (convenience wrappers; forward to p*chase_readHam_)
void schase_readHam_(const char* filename);
void dchase_readHam_(const char* filename);
void cchase_readHam_(const char* filename);
void zchase_readHam_(const char* filename);

#ifdef __cplusplus
}
#endif

#endif