#include <stdio.h>
#include <stdlib.h>
#include <mpi.h> // MPI header file
#include <immintrin.h>

#define scale 15

typedef double v4df __attribute__((vector_size(32)));

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);               // starts MPI, called by every processor
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get current process id
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get number of processes

    int N = 1 << scale;
    int columns = N / size;

    // Construct local part of M
    double *M = (double *)aligned_alloc(32, sizeof(double) * columns * N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < columns; j++)
            M[i * columns + j] = rank * columns + i + j;

    double *V;
    if (rank == 0)
    {
        V = (double *)aligned_alloc(32, sizeof(double) * N);
        for (int i = 0; i < N; i++)
            V[i] = i;
    }
    else
    {
        V = (double *)aligned_alloc(32, sizeof(double) * columns);
    }

    double *C = (double *)aligned_alloc(32, sizeof(double) * N);

    MPI_Barrier(MPI_COMM_WORLD);

    double t0 = MPI_Wtime();
    MPI_Scatter(V, columns, MPI_DOUBLE, V, columns, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    // for (int i = 0; i < N; i++)
    // {
    //     C[i] = 0.0;
    //     for (int j = 0; j < columns; j++)
    //     {
    //         C[i] += M[i * columns + j] * V[j];
    //     }
    // }

    for (int i = 0; i < N; i++)
    {
        v4df c = {0.0, 0.0, 0.0, 0.0};
        for (int j = 0; j < columns; j += 4)
        {
            v4df m = _mm256_load_pd(M + i * columns + j);
            v4df v = _mm256_load_pd(V + j);

            c = m * v + c;
        }
        C[i] = c[0] + c[1] + c[2] + c[3];
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double t2 = MPI_Wtime();
    MPI_Reduce(C, V, N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double t3 = MPI_Wtime();

    if (rank == 0) // Validate results
    {
        printf("%lf %lf %lf\n", t1 - t0, t2 - t1, t3 - t2);

        double error = 0.0;
        for (int i = 0; i < N; i++)
        {
            double target = 0.0;
            for (int j = 0; j < N; j++)
                target += j * (i + j);
            error += (target - V[i]) * (target - V[i]);
        }

        printf("%lf\n", error);
    }

    free(C);
    free(V);
    free(M);

    MPI_Finalize(); // End MPI, called by every processor
    return 0;
}