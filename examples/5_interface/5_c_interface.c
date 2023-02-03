#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <complex.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double drand()   /* uniform distribution, (0..1] */
{
  return (rand()+1.0)/(RAND_MAX+1.0);
}

double random_normal() 
 /* normal distribution, centered on 0, std dev 1 */
{
  return sqrt(-2*log(drand())) * cos(2*M_PI*drand());
}

int zchase_init_(int *N, int *nev, int *nex, double _Complex *H, double _Complex *V, double *ritzv);
void zchase_finalize_();
void zchase_(int* deg, double* tol, char* mode, char* opt);

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 1001;
    int nev = 100;
    int nex = 40;
    int idx_max = 5;
    double perturb = 1e-4;
    
    //config
    int deg = 20;
    double tol = 1e-10;
    char mode = 'R';
    char opt = 'S';

    if (rank == 0)
        printf("ChASE C example driver\n");

    double _Complex *V = (double _Complex *)malloc(sizeof(double _Complex) * N * (nev+nex));
    double *Lambda = (double *)malloc(sizeof(double) * (nev+nex));
    double _Complex *H = (double _Complex *)malloc(sizeof(double _Complex) * N * N);

    int init = 0;
    init = zchase_init_(&N, &nev, &nex, H, V, Lambda);

    // Generate Clement matrix
    for (int i = 0; i < N; ++i)
    {
        H[i + N * i] = 0.0 + 0.0 * I;
        if (i != N - 1){
        	double v = sqrt(i * (N + 1 - i));
            H[i + 1 + N * i] = v + 0.0 * I;
        }

        if (i != N - 1){
        	double v = sqrt(i * (N + 1 - i));
            H[i + N * (i + 1)] = v + 0.0 * I;
        }
    }

    for (int idx = 0; idx < idx_max; ++idx)
    {
        if (rank == 0)
        {
            printf("Starting Problem # %d\n", idx);
            if (idx != 0)
            {
                printf("Using approximate solution\n");
            }
        }

		zchase_(&deg, &tol, &mode, &opt);

        // Perturb Full Clement matrix
        for (int i = 1; i < N; ++i)
        {
            for (int j = 1; j < i; ++j)
            {
            	double randm = random_normal();
                double _Complex element_perturbation = randm * perturb + randm * perturb * I;
                H[j + N * i] += element_perturbation;
                H[i + N * j] += conj(element_perturbation);
            }
        }

        mode = 'A';
    }

    zchase_finalize_();
    MPI_Finalize();
}   