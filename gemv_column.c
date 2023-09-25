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

    size_t N = 1 << scale;
    size_t columns = N / size;

    // Construct local part of A
    double *A = (double *)aligned_alloc(32, sizeof(double) * columns * N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < columns; j++)
            A[i * columns + j] = i + columns * rank + j;

    double *x;
    if (rank == 0)
    {
        x = (double *)aligned_alloc(32, sizeof(double) * N);
        for (int i = 0; i < N; i++)
            x[i] = i;
    }
    else
    {
        x = (double *)aligned_alloc(32, sizeof(double) * columns);
    }

    double *b = (double *)aligned_alloc(32, sizeof(double) * N);

    MPI_Barrier(MPI_COMM_WORLD);

    double t0 = MPI_Wtime();
    MPI_Scatter(x, columns, MPI_DOUBLE, x, columns, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    for (int i = 0; i < N; i++)
    {
        b[i] = 0.0;
        for (int j = 0; j < columns; j++)
        {
            b[i] += A[i * columns + j] * x[j];
        }
    }

    // for (int i = 0; i < rows; i++)
    // {
    //     v4df c = {0.0, 0.0, 0.0, 0.0};
    //     for (int j = 0; j < N; j += 4)
    //     {
    //         v4df m = _mm256_load_pd(A + i * N + j);
    //         v4df v = _mm256_load_pd(x + j);

    //         c = m * v + c;
    //     }
    //     b[i] = c[0] + c[1] + c[2] + c[3];
    // }

    MPI_Barrier(MPI_COMM_WORLD);

    double t2 = MPI_Wtime();
    if (rank == 0)
        MPI_Reduce(MPI_IN_PLACE, b, N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    else
        MPI_Reduce(b, b, N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
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
            error += (target - b[i]) * (target - b[i]);
        }

        printf("%lf\n", error);
    }

    free(b);
    free(x);
    free(A);

    MPI_Finalize(); // End MPI, called by every processor
    return 0;
}