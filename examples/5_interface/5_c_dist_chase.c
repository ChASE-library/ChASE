#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <complex.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void pzchase_init_(int *N, int *nev, int *nex, int *m, int *n, double _Complex *H,  int *ldh, double _Complex *V, double *ritzv, int *dim0, int *dim1, char *grid_major, MPI_Comm *comm, int *init);
void pzchase_finalize_(int *flag);
void pzchase_(int* deg, double* tol, char* mode, char* opt);

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size, init;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(size != 2){
      printf("This example is only desigend for MPI comm size = 2\n");
      return 1;
    }

    int N = 1001;
    int nev = 100;
    int nex = 40;
    int idx_max = 5;
    int m;
    int n;
    int xoff, yoff, xlen, ylen;
    MPI_Comm comm = MPI_COMM_WORLD;

    double perturb = 1e-4;
    int dims[2];
    dims[0] = 2;
    dims[1] = 1;

    //config
    int deg = 20;
    double tol = 1e-10;
    char mode = 'R';
    char opt = 'S';

    m = 501;
    n = N;
    if(rank == 0){
      xoff = 0;
    }else{
      xoff = 501;
    }

    yoff = 0;

    xlen = m;
    ylen = n;

    if (rank == 0)
        printf("ChASE C example driver\n");

    double _Complex *V = (double _Complex *)malloc(sizeof(double _Complex) * m * (nev+nex));
    double *Lambda = (double *)malloc(sizeof(double) * (nev+nex));
    double _Complex *H = (double _Complex *)malloc(sizeof(double _Complex) * m * n);
    double _Complex *Hh = (double _Complex *)malloc(sizeof(double _Complex) * N * N);

    pzchase_init_(&N, &nev, &nex, &m, &n, H, &m, V, Lambda, &dims[0], &dims[1], (char*)"C", &comm, &init);

    // Generate Clement matrix
    for (int i = 0; i < N; ++i)
    {
        Hh[i + N * i] = 0.0 + 0.0 * I;
        if (i != N - 1){
          double v = sqrt(i * (N + 1 - i));
            Hh[i + 1 + N * i] = v + 0.0 * I;
        }

        if (i != N - 1){
          double v = sqrt(i * (N + 1 - i));
            Hh[i + N * (i + 1)] = v + 0.0 * I;
        }
    }

    //distribute Hh to H
    
    for (int x = 0; x < xlen; x++)
    {
      for (int y = 0; y < ylen; y++)
      {
         H[x + xlen * y] = Hh[(xoff + x) * N + (yoff + y)];
      }
    }

    pzchase_(&deg, &tol, &mode, &opt);

    pzchase_finalize_(&init);

    MPI_Finalize();
}   
