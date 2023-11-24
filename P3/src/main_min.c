#include "mtx.h"
#include "spmv.h"

#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h> // MPI header file

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);               // starts MPI, called by every processor
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get current process id
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get number of processes

    graph g;
    int *p = malloc(sizeof(int) * (size + 1));
    double *input;
    comm_lists c = init_comm_lists(size);

    if (rank == 0)
    {
        g = parse_and_validate_mtx(argv[1]);

        input = malloc(sizeof(double) * g.N);
        for (int i = 0; i < g.N; i++)
            input[i] = ((double)rand() / (double)RAND_MAX) - 0.5;

        // partition_graph_naive(g, 0, g.N, size, p);
        partition_graph(g, size, p, input);
    }

    distribute_graph(&g, rank);
    MPI_Bcast(p, size + 1, MPI_INT, 0, MPI_COMM_WORLD);

    double *V = malloc(sizeof(double) * g.N);

    find_sendlists(g, p, rank, size, c);
    find_receivelists(g, p, rank, size, c);

    double *Y = malloc(sizeof(double) * g.N);

    int *p_local;
#pragma omp parallel shared(p_local)
    {
        int nt = omp_get_num_threads();
        int tid = omp_get_thread_num();
        if (tid == 0)
        {
            p_local = malloc(sizeof(int) * (nt + 1));
            partition_graph_naive(g, p[rank], p[rank + 1], nt, p_local);
        }
    }

    // Main program start - Full version

    if (rank == 0)
        printf("Time GFLOPs GBs CS\n");

    for (int t = 0; t < 3; t++)
    {
        if (rank == 0)
            for (int i = 0; i < g.N; i++)
                V[i] = input[i];
        MPI_Bcast(V, g.N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);

        for (int i = 0; i < 100; i++)
        {
#pragma omp parallel default(shared)
            {
                int tid = omp_get_thread_num();
                spmv_part(g, p_local[tid], p_local[tid + 1], V, Y);
            }

            exchange_separators(c, Y, rank, size);

            double *tmp = Y;
            Y = V;
            V = tmp;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        // Gather results in rank 0
        if (rank == 0)
            for (int i = 1; i < size; i++)
                MPI_Recv(V + p[i], p[i + 1] - p[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        else
            MPI_Send(V + p[rank], p[rank + 1] - p[rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            double a = 1.0;
            for (int i = 0; i < g.N; i++)
                a = a < fabs(V[i]) ? fabs(V[i]) : a;

            double L2 = 0.0;
            for (int i = 0; i < g.N; i++)
                L2 += (V[i] / a) * (V[i] / a);

            double ops = (double)g.M * 2.0 * 100.0;
            double GFLOPS = (ops / (t1 - t0)) / 1e9;

            //                            g.E,  g.A                   g.V,   V,    Y
            double data = ((double)g.M * (4.0 + 8.0) + (double)g.N * (4.0 + 8.0 + 8.0)) * 100.0;
            double GBs = (data / (t1 - t0)) / 1e9;

            printf("%.2lf %.2lf %.2lf %.10lf\n", t1 - t0, GFLOPS, GBs, L2);
        }
    }

    // Only compute

    if (rank == 0)
        printf("Time GFLOPs GBs\n");

    for (int t = 0; t < 3; t++)
    {
        if (rank == 0)
            for (int i = 0; i < g.N; i++)
                V[i] = input[i];
        MPI_Bcast(V, g.N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);

        // Main program start - Full version

        for (int i = 0; i < 100; i++)
        {
#pragma omp parallel default(shared)
            {
                int tid = omp_get_thread_num();
                spmv_part(g, p_local[tid], p_local[tid + 1], V, Y);
            }

            double *tmp = Y;
            Y = V;
            V = tmp;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        if (rank == 0)
        {
            double ops = (double)g.M * 2.0 * 100.0;
            double GFLOPS = (ops / (t1 - t0)) / 1e9;

            //                            g.E,  g.A                   g.V,   V,    Y
            double data = ((double)g.M * (4.0 + 8.0) + (double)g.N * (4.0 + 8.0 + 8.0)) * 100.0;
            double GBs = (data / (t1 - t0)) / 1e9;

            printf("%.2lf %.2lf %.2lf\n", t1 - t0, GFLOPS, GBs);
        }
    }

    // Only communication

    if (rank == 0)
        printf("Time GBs MB\n");

    for (int t = 0; t < 3; t++)
    {
        if (rank == 0)
            for (int i = 0; i < g.N; i++)
                V[i] = input[i];
        MPI_Bcast(V, g.N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);

        // Main program start - Full version

        for (int i = 0; i < 100; i++)
        {
            exchange_separators(c, Y, rank, size);

            double *tmp = Y;
            Y = V;
            V = tmp;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        // Communication volume
        int commc = 0;
        for (int i = 0; i < size; i++)
            commc += c.send_count[i];

        int commc_total = 0;
        MPI_Reduce(&commc, &commc_total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            double data = (double)commc_total * 8.0 * 100.0;
            double GBs = (data / (t1 - t0)) / 1e9;

            printf("%.2lf %.2lf %.2lf\n", t1 - t0, GBs, (double)commc_total / 1e6);
        }
    }

    // Cleanup

    free(p);
    free(p_local);
    free_graph(&g);
    free(V);
    free(Y);

    MPI_Finalize(); // End MPI, called by every processor
}