#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

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


int chaseSeqInit(int *N, int *nev, int *nex, double *H, double *V, double *ritzv);
void chaseSeqFinalize();
void chaseSeqSolve(int* deg, double* tol, char* mode, char* opt);

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("%d, %d\n", rank, size );

    int N = 1001;
    int nev = 40;
    int nex = 20;
    int idx_max = 2;
    double perturb = 1e-4;
    
    //config
    int deg = 20;
    double tol = 1e-10;
    char mode = 'R';
    char opt = 'S';

    if (rank == 0)
        printf("ChASE C example driver\n");

    double *V = (double *)malloc(sizeof(double) * N * (nev+nex));
    double *Lambda = (double *)malloc(sizeof(double) * (nev+nex));
    double *H = (double *)malloc(sizeof(double) * N * N);

    int init = 0;
    init = chaseSeqInit(&N, &nev, &nex, H, V, Lambda);

    // Generate Clement matrix
    for (int i = 0; i < N; ++i)
    {
        H[i + N * i] = 0;
        if (i != N - 1)
            H[i + 1 + N * i] = sqrt(i * (N + 1 - i));
        if (i != N - 1)
            H[i + N * (i + 1)] = sqrt(i * (N + 1 - i));
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

		chaseSeqSolve(&deg, &tol, &mode, &opt);

        // Perturb Full Clement matrix
/*        for (int i = 1; i < N; ++i)
        {
            for (int j = 1; j < i; ++j)
            {
                double element_perturbation = random_normal() * perturb;
                H[j + N * i] += element_perturbation;
            }
        }
*/
        mode = 'A';
    }

    chaseSeqFinalize();
    MPI_Finalize();
}   