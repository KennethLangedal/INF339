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

    double bt = 1e9, btc1 = 1e9, btf = 1e9, btc2 = 1e9;

    MPI_Barrier(MPI_COMM_WORLD);
    double ts0 = MPI_Wtime();

    if (argc != 3)
    {
        printf("Give two arguments, scale and iterations\n");
        return 1;
    }

    int scale = atoi(argv[1]);
    int n_it = atoi(argv[2]);

    size_t N = 1 << scale;
    size_t columns = N / size;

    // Construct local part of A
    double *A = (double *)aligned_alloc(32, sizeof(double) * columns * N);
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < columns; j++)
            A[i * columns + j] = i + columns * rank + j;

    double *x;
    if (rank == 0)
    {
        x = (double *)aligned_alloc(32, sizeof(double) * N);
        for (size_t i = 0; i < N; i++)
            x[i] = i;
    }
    else
    {
        x = (double *)aligned_alloc(32, sizeof(double) * columns);
    }

    double *b = (double *)aligned_alloc(32, sizeof(double) * N);
    double *res;
    if (rank == 0)
        res = (double *)aligned_alloc(32, sizeof(double) * N);

    MPI_Barrier(MPI_COMM_WORLD);
    double ts1 = MPI_Wtime();

    for (size_t it = 0; it < n_it; it++)
    {

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        MPI_Scatter(x, columns, MPI_DOUBLE, x, columns, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        if (rank == 0 && t1 - t0 < btc1)
            btc1 = t1 - t0;
    }
    for (size_t it = 0; it < n_it; it++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
#ifdef AVX
        for (size_t i = 0; i < N; i++)
        {
            v4df c = {0.0, 0.0, 0.0, 0.0};
            for (size_t j = 0; j < columns; j += 4)
            {
                v4df m = _mm256_load_pd(A + i * columns + j);
                v4df v = _mm256_load_pd(x + j);

                c = m * v + c;
            }
            b[i] = c[0] + c[1] + c[2] + c[3];
        }
#else
        for (size_t i = 0; i < N; i++)
        {
            b[i] = 0.0;
            for (size_t j = 0; j < columns; j++)
            {
                b[i] += A[i * columns + j] * x[j];
            }
        }
#endif

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        if (rank == 0 && t1 - t0 < btf)
            btf = t1 - t0;
    }
    for (size_t it = 0; it < n_it; it++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        MPI_Reduce(b, res, N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        if (rank == 0 && t1 - t0 < btc2)
            btc2 = t1 - t0;
    }

    if (rank == 0) // Validate results
    {
        double error = 0.0;
        for (size_t i = 0; i < N; i++)
        {
            double target = 0.0;
            for (size_t j = 0; j < N; j++)
                target += j * (i + j);
            error += (target - res[i]) * (target - res[i]);
        }

        bt = btc1 + btf + btc2;
        printf("%lf %lf %lf %lf %lf %lf\n", bt, btc1, btf, btc2, ts1 - ts0, error);
    }

    free(b);
    free(x);
    free(A);

    MPI_Finalize(); // End MPI, called by every processor
    return 0;
}