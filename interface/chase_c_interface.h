#ifndef CHASE_C_INTERFACE_H
#define CHASE_C_INTERFACE_H

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

#ifdef __cplusplus
}
#endif

#endif