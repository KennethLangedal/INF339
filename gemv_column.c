#include <stdio.h>
#include <stdlib.h>
#include <mpi.h> // MPI header file
#include <immintrin.h>

// #define AVX
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
    if (argc != 3)
    {
        printf("Give two arguments, scale and iterations\n");
        return 1;
    }

    int scale = atoi(argv[1]);
    int n_it = atoi(argv[2]);

    size_t N = 1 << scale;
    size_t columns = N / size;

    // -----Initialization start-----
    MPI_Barrier(MPI_COMM_WORLD);
    double ts0 = MPI_Wtime();

    // Construct local part of A
    double *A = (double *)aligned_alloc(32, sizeof(double) * columns * N);
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < columns; j++)
            A[i * columns + j] = i + columns * rank + j;

    // Allocate data for x, rank 0 initialize
    // Rank 0 allocates for the whole x array, everyone else just their share
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

    // Allocate data for b, and additional array for the final results
    double *b = (double *)aligned_alloc(32, sizeof(double) * N);
    double *res = NULL;
    if (rank == 0)
        res = (double *)aligned_alloc(32, sizeof(double) * N);

    MPI_Barrier(MPI_COMM_WORLD);
    double ts1 = MPI_Wtime();
    // -----Initialization end-----

    // -----Main program start-----
    for (size_t it = 0; it < n_it; it++)
    {
        // Distribute data
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        MPI_Scatter(x, columns, MPI_DOUBLE, x, columns, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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
        MPI_Reduce(b, res, N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

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
            double target = 0.0;
            for (size_t j = 0; j < N; j++)
                target += j * (i + j);
            error += (target - res[i]) * (target - res[i]);
        }

#ifdef TIME_INDIVIDUAL
        printf("Init and error: %lf %lf\n", ts1 - ts0, error);
        printf("Comm 1: %lf %lf %lf\n", t_com1.min, t_com1.max, t_com1.total / t_com1.count);
        printf("Compute: %lf %lf %lf\n", t_mm.min, t_mm.max, t_mm.total / t_mm.count);
        printf("Comm 2: %lf %lf %lf\n", t_com2.min, t_com2.max, t_com2.total / t_com2.count);
#else
        printf("%lf %lf %lf %lf %lf\n", t.min, t.max, t.total / t.count, ts1 - ts0, error);
#endif
    }

    free(b);
    free(x);
    free(A);

    MPI_Finalize(); // End MPI, called by every processor
    return 0;
}