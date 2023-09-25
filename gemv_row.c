#include <stdio.h>
#include <stdlib.h>
#include <mpi.h> // MPI header file
#include <immintrin.h>

typedef double v4df __attribute__((vector_size(32)));

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);               // starts MPI, called by every processor
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get current process id
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get number of processes

    if (argc != 2)
    {
        printf("Give one argument to use as scale\n");
        return 1;
    }

    int scale = atoi(argv[1]);

    size_t N = 1 << scale;
    size_t rows = N / size;

    // Construct local part of A
    double *A = (double *)aligned_alloc(32, sizeof(double) * rows * N);
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < N; j++)
            A[i * N + j] = rank * rows + i + j;

    double *x = (double *)aligned_alloc(32, sizeof(double) * N);
    if (rank == 0)
    {
        for (size_t i = 0; i < N; i++)
            x[i] = i;
    }

    double *b;
    if (rank == 0)
        b = (double *)aligned_alloc(32, sizeof(double) * N);
    else
        b = (double *)aligned_alloc(32, sizeof(double) * rows);

    MPI_Barrier(MPI_COMM_WORLD);

    double t0 = MPI_Wtime();
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

#ifndef AVX
    for (size_t i = 0; i < rows; i++)
    {
        b[i] = 0.0;
        for (size_t j = 0; j < N; j++)
        {
            b[i] += A[i * N + j] * x[j];
        }
    }
#else
    for (size_t i = 0; i < rows; i++)
    {
        v4df c = {0.0, 0.0, 0.0, 0.0};
        for (size_t j = 0; j < N; j += 4)
        {
            v4df m = _mm256_load_pd(A + i * N + j);
            v4df v = _mm256_load_pd(x + j);

            c = m * v + c;
        }
        b[i] = c[0] + c[1] + c[2] + c[3];
    }
#endif

    MPI_Barrier(MPI_COMM_WORLD);

    double t2 = MPI_Wtime();
    MPI_Gather(b, rows, MPI_DOUBLE, b, rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t3 = MPI_Wtime();

    if (rank == 0) // Validate results
    {
        double error = 0.0;
        for (size_t i = 0; i < N; i++)
        {
            double target = 0.0;
            for (size_t j = 0; j < N; j++)
                target += j * (i + j);
            error += (target - b[i]) * (target - b[i]);
        }

        printf("%lf %lf %lf %lf\n", t1 - t0, t2 - t1, t3 - t2, error);
    }

    free(b);
    free(x);
    free(A);

    MPI_Finalize(); // End MPI, called by every processor
    return 0;
}