#include <stdio.h>
#include <stdlib.h>
#include <mpi.h> // MPI header file
#include <immintrin.h>
#include <assert.h>

#define AVX
// #define TIME_INDIVIDUAL

typedef double v4df __attribute__((vector_size(32)));

typedef struct
{
    double min, max, total, count;
} time_info;

void update_time_info(time_info *ti, double t)
{
    if (t < ti->min)
        ti->min = t;
    if (t > ti->max)
        ti->max = t;
    ti->total += t;
    ti->count += 1.0;
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);               // starts MPI, called by every processor
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get current process id
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get number of processes

#ifdef TIME_INDIVIDUAL
    time_info t_com1 = {.min = 1e9, .max = 0.0, .total = 0.0, .count = 0.0};
    time_info t_com2 = {.min = 1e9, .max = 0.0, .total = 0.0, .count = 0.0};
    time_info t_mm = {.min = 1e9, .max = 0.0, .total = 0.0, .count = 0.0};
#else
    time_info t = {.min = 1e9, .max = 0.0, .total = 0.0, .count = 0.0};
#endif

    // Parse command-line arguments
    if (argc != 4)
    {
        printf("Give three arguments, scale, vectors, and iterations\n");
        return 1;
    }

    int scale = atoi(argv[1]);
    int vectors = atoi(argv[2]);
    int n_it = atoi(argv[3]);

    size_t N = 1 << scale;
    size_t rows = N / size;

    // -----Initialization start-----
    MPI_Barrier(MPI_COMM_WORLD);
    double ts0 = MPI_Wtime();

    // Construct local part of A
    double *A = (double *)aligned_alloc(32, sizeof(double) * rows * N);
    for (size_t i = 0; i < rows; i += 4)
        for (size_t j = 0; j < N; j++)
            for (size_t k = 0; k < 4; k++)
                A[j * 4 + i * N + k] = rank * rows + i + j + k;

    // Allocate data for x, rank 0 initialize
    double *x = (double *)aligned_alloc(32, sizeof(double) * N * vectors);
    if (rank == 0)
    {
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < vectors; j++)
                x[i * vectors + j] = i + j;
    }

    // Rank 0 allocates for the whole b array, everyone else just their share
    double *b;
    if (rank == 0)
        b = (double *)aligned_alloc(32, sizeof(double) * N * vectors);
    else
        b = (double *)aligned_alloc(32, sizeof(double) * rows * vectors);

    MPI_Barrier(MPI_COMM_WORLD);
    double ts1 = MPI_Wtime();
    // -----Initialization end-----

    // -----Main program start-----
    for (size_t it = 0; it < n_it; it++)
    {
        // Distribute data
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        MPI_Bcast(x, N * vectors, MPI_DOUBLE, 0, MPI_COMM_WORLD);

#ifdef TIME_INDIVIDUAL
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        update_time_info(&t_com1, t1 - t0);
    }
    for (size_t it = 0; it < n_it; it++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        double t2 = MPI_Wtime();
#endif

        // Compute matrix vector product
#ifdef AVX

        assert(vectors == 8);

        for (size_t i = 0; i < rows * vectors; i++)
            b[i] = 0.0;

        for (size_t j = 0; j < N; j += 256)
        {
            for (size_t i = 0; i < rows; i += 4)
            {
                v4df b00 = _mm256_load_pd(b + i * vectors);
                v4df b01 = _mm256_load_pd(b + i * vectors + 4);
                v4df b10 = _mm256_load_pd(b + (i + 1) * vectors);
                v4df b11 = _mm256_load_pd(b + (i + 1) * vectors + 4);
                v4df b20 = _mm256_load_pd(b + (i + 2) * vectors);
                v4df b21 = _mm256_load_pd(b + (i + 2) * vectors + 4);
                v4df b30 = _mm256_load_pd(b + (i + 3) * vectors);
                v4df b31 = _mm256_load_pd(b + (i + 3) * vectors + 4);

                for (size_t k = 0; k < 256; k++)
                {
                    v4df x0 = _mm256_load_pd(x + (j + k) * vectors);
                    v4df x1 = _mm256_load_pd(x + (j + k) * vectors + 4);

                    v4df a0 = _mm256_broadcast_sd(A + i * N + (j + k) * 4);
                    b00 = _mm256_fmadd_pd(a0, x0, b00);
                    b01 = _mm256_fmadd_pd(a0, x1, b01);

                    v4df a1 = _mm256_broadcast_sd(A + i * N + (j + k) * 4 + 1);
                    b10 = _mm256_fmadd_pd(a1, x0, b10);
                    b11 = _mm256_fmadd_pd(a1, x1, b11);

                    v4df a2 = _mm256_broadcast_sd(A + i * N + (j + k) * 4 + 2);
                    b20 = _mm256_fmadd_pd(a2, x0, b20);
                    b21 = _mm256_fmadd_pd(a2, x1, b21);

                    v4df a3 = _mm256_broadcast_sd(A + i * N + (j + k) * 4 + 3);
                    b30 = _mm256_fmadd_pd(a3, x0, b30);
                    b31 = _mm256_fmadd_pd(a3, x1, b31);
                }

                _mm256_store_pd(b + i * vectors, b00);
                _mm256_store_pd(b + i * vectors + 4, b01);
                _mm256_store_pd(b + (i + 1) * vectors, b10);
                _mm256_store_pd(b + (i + 1) * vectors + 4, b11);
                _mm256_store_pd(b + (i + 2) * vectors, b20);
                _mm256_store_pd(b + (i + 2) * vectors + 4, b21);
                _mm256_store_pd(b + (i + 3) * vectors, b30);
                _mm256_store_pd(b + (i + 3) * vectors + 4, b31);
            }
        }
#else
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < vectors; j++)
                b[i * vectors + j] = 0.0;

        for (size_t i = 0; i < rows; i += 4)
        {
            for (size_t k = 0; k < N; k++)
            {
                for (size_t j = 0; j < vectors; j++)
                {
                    for (size_t l = 0; l < 4; l++)
                    {
                        b[(i + l) * vectors + j] += A[i * N + k * 4 + l] * x[k * vectors + j];
                    }
                }
            }
        }
#endif

#ifdef TIME_INDIVIDUAL
        MPI_Barrier(MPI_COMM_WORLD);
        double t3 = MPI_Wtime();

        update_time_info(&t_mm, t3 - t2);
    }
    for (size_t it = 0; it < n_it; it++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        double t4 = MPI_Wtime();
#endif
        // Gather results
        MPI_Gather(b, rows * vectors, MPI_DOUBLE, b, rows * vectors, MPI_DOUBLE, 0, MPI_COMM_WORLD);

#ifdef TIME_INDIVIDUAL
        MPI_Barrier(MPI_COMM_WORLD);
        double t5 = MPI_Wtime();
        update_time_info(&t_com2, t5 - t4);
#else
        double t1 = MPI_Wtime();
        update_time_info(&t, t1 - t0);
#endif
    }

    if (rank == 0) // Validate results
    {
        double error = 0.0;
        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < vectors; j++)
            {
                double target = 0.0;
                for (size_t k = 0; k < N; k++)
                    target += (i + k) * (j + k);

                // printf("%lf %lf\n", target, b[i * vectors + j]);
                error += (target - b[i * vectors + j]) * (target - b[i * vectors + j]);
            }
        }

#ifdef TIME_INDIVIDUAL
        printf("Init and error: %lf %lf\n", ts1 - ts0, error);
        printf("Comm 1: %lf %lf %lf\n", t_com1.min, t_com1.max, t_com1.total / t_com1.count);
        printf("Compute: %lf %lf %lf\n", t_mm.min, t_mm.max, t_mm.total / t_mm.count);
        printf("Comm 2: %lf %lf %lf\n", t_com2.min, t_com2.max, t_com2.total / t_com2.count);
#else
        printf("%lf %lf %lf %lf %lf %lf\n", t.min, t.max, t.total / t.count, ts1 - ts0, error, (double)(N * N * vectors * 2) / t.min / 1e9);
#endif
    }

    free(b);
    free(x);
    free(A);

    MPI_Finalize(); // End MPI, called by every processor
    return 0;
}