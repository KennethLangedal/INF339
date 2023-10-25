#include <stdio.h>
#include <stdlib.h>
#include <mpi.h> // MPI header file
#include <math.h>
#include <immintrin.h>
#include "p2.h"

#define alpha 0.7
#define beta 0.1

typedef double v4df __attribute__((vector_size(32)));

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);               // starts MPI, called by every processor
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get current process id
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get number of processes

    // Parse command-line arguments
    if (argc != 3)
    {
        printf("Give two arguments, scale and iterations\n");
        return 1;
    }

    int scale = atoi(argv[1]);
    int n_it = atoi(argv[2]);

    size_t N = 1 << (scale * 2);
    size_t rows = N / size;

    // -----Initialization start-----
    MPI_Barrier(MPI_COMM_WORLD);
    double ts0 = MPI_Wtime();

    // Allocate local part of A and I
    double *A = aligned_alloc(32, sizeof(double) * rows * nonzero);
    int *I = aligned_alloc(16, sizeof(int) * rows * nonzero);

    int *sep = malloc(sizeof(int) * size);

    mesh m;
    int start_id = 0;
    if (rank == 0)
    {
        m = init_mesh_4(scale, alpha, beta);
        int *old_id = malloc(sizeof(int) * m.N);

        reorder_separators(m, size, rows, sep, old_id);
        for (int i = 0; i < m.N; i++)
        {
            if (old_id[i] == 0)
            {
                start_id = i;
                break;
            }
        }

        free(old_id);
    }

    MPI_Bcast(sep, size, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scatter(m.A, rows * nonzero, MPI_DOUBLE, A, rows * nonzero, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(m.I, rows * nonzero, MPI_INT, I, rows * nonzero, MPI_INT, 0, MPI_COMM_WORLD);

    double *Vo = aligned_alloc(32, sizeof(double) * N);
    double *Vn = aligned_alloc(32, sizeof(double) * N);

    MPI_Barrier(MPI_COMM_WORLD);
    double ts1 = MPI_Wtime();

    if (rank == 0)
        printf("%lfs for initialization\n", ts1 - ts0);

    // -----Initialization end-----

    // -----Main program start-----
    for (size_t it = 0; it < n_it; it++)
    {
        for (int i = 0; i < N; i++)
            Vo[i] = 0.0;
        for (int i = 0; i < N; i++)
            Vn[i] = 0.0;

        if (rank == 0)
            Vo[start_id] = 0xffffff;

        double tcomm = 0.0, tcomp = 0.0;

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

#pragma omp parallel
        {
            for (int i = 0; i < 100; i++)
            {
                double tc1 = MPI_Wtime();

#pragma omp for
                for (size_t j = 0; j < rows; j += 4)
                {
                    v4df z = {0.0, 0.0, 0.0, 0.0};

                    for (size_t k = 0; k < 4; k++)
                        z[k] = A[(j + k) * nonzero + 0] * Vo[I[(j + k) * nonzero + 0]] +
                               A[(j + k) * nonzero + 1] * Vo[I[(j + k) * nonzero + 1]] +
                               A[(j + k) * nonzero + 2] * Vo[I[(j + k) * nonzero + 2]] +
                               A[(j + k) * nonzero + 3] * Vo[I[(j + k) * nonzero + 3]];

                    _mm256_stream_pd(Vn + rank * rows + j, z);
                }

#pragma omp master
                {
                    MPI_Barrier(MPI_COMM_WORLD);
                    double tc2 = MPI_Wtime();

                    for (int j = 0; j < size; j++)
                        MPI_Bcast(Vn + rows * j, sep[j], MPI_DOUBLE, j, MPI_COMM_WORLD);

                    double *t = Vn;
                    Vn = Vo;
                    Vo = t;

                    MPI_Barrier(MPI_COMM_WORLD);
                    double tc3 = MPI_Wtime();

                    tcomm += tc3 - tc2;
                    tcomp += tc2 - tc1;
                }
#pragma omp barrier
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        MPI_Allgather(Vo + rank * rows, rows, MPI_DOUBLE, Vn, rows, MPI_DOUBLE, MPI_COMM_WORLD);

        // Compute L2 and GLOPS

        double l2 = 0.0;
        for (int j = 0; j < N; j++)
            l2 += Vn[j] * Vn[j];

        l2 = sqrt(l2);

        double ops = (long long)m.N * 8ll * 100ll; // 4 multiplications and 4 additions
        double time = t1 - t0;

        if (rank == 0)
        {
            int comm_size = 0;
            for (int j = 0; j < size; j++)
                comm_size += sep[j];

            printf("%lfs (%lfs, %lfs), %lf GFLOPS, %lf GBs mem, %lf GBs comm, L2 = %lf\n",
                   time, tcomp, tcomm, (ops / time) / 1e9, (N * 64.0 * 100.0 / tcomp) / 1e9, (comm_size * size * 8.0 * 100.0 / tcomm) / 1e9, l2);
        }
    }

    if (rank == 0)
        free_mesh(&m);

    free(A);
    free(I);
    free(Vn);
    free(Vo);

    MPI_Finalize(); // End MPI, called by every processor
    return 0;
}