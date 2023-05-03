// This file is a part of ChASE.
// Copyright (c) 2015-2023, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <complex.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

void pzchase_init_(int* N, int* nev, int* nex, int* m, int* n,
                   double _Complex* H, int* ldh, double _Complex* V,
                   double* ritzv, int* dim0, int* dim1, char* grid_major,
                   MPI_Comm* comm, int* init);
void pzchase_finalize_(int* flag);
void pzchase_(int* deg, double* tol, char* mode, char* opt, char *qr);

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size, init;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 1001;
    int nev = 100;
    int nex = 40;
    int idx_max = 5;
    int m;
    int n;
    int xoff, yoff, xlen, ylen;
    MPI_Comm comm = MPI_COMM_WORLD;

    int dims[2];
    dims[0] = dims[1] = 0;
    // Create a grid as square as possible
    // if not able to be square, dims[0] > dim[1]
    MPI_Dims_create(size, 2, dims);

    // config
    int deg = 20;
    double tol = 1e-10;
    char mode = 'R';
    char opt = 'S';
    char qr = 'C';

    if (N % dims[0] == 0)
    {
        m = N / dims[0];
    }
    else
    {
        m = MIN(N, N / dims[0] + 1);
    }
    if (N % dims[1] == 0)
    {
        n = N / dims[1];
    }
    else
    {
        n = MIN(N, N / dims[1] + 1);
    }

    xoff = (rank % dims[0]) * m;
    yoff = (rank / dims[0]) * n;

    xlen = m;
    ylen = n;

    if (rank == 0)
        printf("ChASE C example driver\n");

    double _Complex* V =
        (double _Complex*)malloc(sizeof(double _Complex) * m * (nev + nex));
    double* Lambda = (double*)malloc(sizeof(double) * (nev + nex));
    double _Complex* H =
        (double _Complex*)malloc(sizeof(double _Complex) * m * n);

    pzchase_init_(&N, &nev, &nex, &m, &n, H, &m, V, Lambda, &dims[0], &dims[1],
                  (char*)"C", &comm, &init);

    // Generate Clement matrix in distributed manner
    for (int x = 0; x < xlen; x++)
    {
        for (int y = 0; y < ylen; y++)
        {
            int x_global = xoff + x;
            int y_global = yoff + y;
            if (x_global == y_global + 1)
            {
                double v = sqrt(y_global * (N + 1 - y_global));
                H[x + xlen * y] = v + 0.0 * I;
            }
            if (y_global == x_global + 1)
            {
                double v = sqrt(x_global * (N + 1 - x_global));
                H[x + xlen * y] = v + 0.0 * I;
            }
        }
    }

    pzchase_(&deg, &tol, &mode, &opt, &qr);

    pzchase_finalize_(&init);

    MPI_Finalize();
}
