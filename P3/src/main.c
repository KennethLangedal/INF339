#include "mtx.h"
#include "spmv.h"
#include <stdlib.h>
#include <math.h>
#include <mpi.h> // MPI header file

// #define COMP_ONLY
// #define COMM_ONLY

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);               // starts MPI, called by every processor
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get current process id
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get number of processes

    int *pV = malloc(sizeof(int) * (size + 1)); // Vertex range per rank

    graph g;

    double *input;

    // Parse input, partition and renumber graph, distribute data

    if (rank == 0)
    {
        FILE *f = fopen(argv[1], "r");
        g = parse_mtx(f);
        fclose(f);

        printf("%d %d\n", g.N, g.M);

        input = malloc(sizeof(double) * g.N);

        // Generate random input
        for (int i = 0; i < g.N; i++)
        {
            input[i] = ((double)rand() / (double)RAND_MAX) - 0.5;
        }

        partition_graph(g, size, pV, input);

        sort_edges(g);
        normalize_graph(g);
        if (!validate_graph(g))
            printf("Error in graph\n");
    }

    MPI_Bcast(&g.N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&g.M, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        input = malloc(sizeof(double) * g.N);
        g.V = malloc(sizeof(int) * (g.N + 1));
        g.E = malloc(sizeof(int) * g.M);
        g.A = malloc(sizeof(double) * g.M);
    }

    MPI_Bcast(pV, size + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(input, g.N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(g.V, g.N + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g.E, g.M, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g.A, g.M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double *x = malloc(sizeof(double) * g.N);
    double *y = malloc(sizeof(double) * g.N);

    // Compute send and recieve lists (super bad solution, but small number of nodes and not timed)

    int *recv_mark = malloc(sizeof(int) * g.N);
    int *recv_count = malloc(sizeof(int) * size);

    for (int i = 0; i < g.N; i++)
        recv_mark[i] = -1;
    for (int i = 0; i < size; i++)
        recv_count[i] = 0;

    for (int i = pV[rank]; i < pV[rank + 1]; i++)
    {
        for (int j = g.V[i]; j < g.V[i + 1]; j++)
        {
            if (g.E[j] >= pV[rank] && g.E[j] < pV[rank + 1])
                continue;
            int from = 0;
            while (!(g.E[j] >= pV[from] && g.E[j] < pV[from + 1]))
                from++;

            if (recv_mark[g.E[j]] != from)
                recv_count[from]++;
            recv_mark[g.E[j]] = from;
        }
    }

    int **recv_list = malloc(sizeof(int *) * size);
    for (int i = 0; i < size; i++)
    {
        if (recv_count[i] == 0)
            continue;
        recv_list[i] = malloc(sizeof(int) * recv_count[i]);
        int id = 0;
        for (int j = 0; j < g.N; j++)
        {
            if (recv_mark[j] == i)
                recv_list[i][id++] = j;
        }
    }

    int *send_count = malloc(sizeof(int) * size);
    int **send_list = malloc(sizeof(int *) * size);
    for (int i = 0; i < size; i++)
    {
        if (i == rank)
        {
            send_count[i] = 0;
            for (int j = 0; j < size; j++)
            {
                if (j != rank)
                {
                    MPI_Send(&recv_count[j], 1, MPI_INT, j, 0, MPI_COMM_WORLD);
                    if (recv_count[j] > 0)
                        MPI_Send(recv_list[j], recv_count[j], MPI_INT, j, 0, MPI_COMM_WORLD);
                }
            }
        }
        else
        {
            MPI_Recv(&send_count[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (send_count[i] > 0)
            {
                send_list[i] = malloc(sizeof(int) * send_count[i]);
                MPI_Recv(send_list[i], send_count[i], MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    double *send_buffer = malloc(sizeof(double) * g.N);
    double *recv_buffer = malloc(sizeof(double) * g.N);

    // Finished computing send/recieve lists

    if (rank == 0)
        printf("GFLOPS GFLOP GBs_mem GBs_net GB_net t L2\n");

    // Main loop start
    for (int e = 0; e < 3; e++)
    {

        for (int i = 0; i < g.N; i++)
        {
            x[i] = input[i];
        }

        double comm_count = 0.0;
        MPI_Request send_requests[size];
        MPI_Request recv_requests[size];

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        for (int t = 0; t < 100; t++)
        {
#ifndef COMM_ONLY
            spmv_part(g, pV[rank], pV[rank + 1], x, y);
#endif

#ifndef COMP_ONLY
            for (int i = 0; i < size; i++)
            {
                if (send_count[i] == 0)
                    continue;

                // Construct message
                for (int j = 0; j < send_count[i]; j++)
                    send_buffer[pV[i] + j] = y[send_list[i][j]];

                comm_count += send_count[i];
                MPI_Isend(send_buffer + pV[i], send_count[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &send_requests[i]);
            }

            for (int i = 0; i < size; i++)
            {
                if (recv_count[i] == 0)
                    continue;

                MPI_Irecv(recv_buffer + pV[i], recv_count[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &recv_requests[i]);
            }

            for (int i = 0; i < size; i++)
            {
                if (recv_count[i] == 0)
                    continue;
                MPI_Wait(&recv_requests[i], MPI_STATUS_IGNORE);

                // Unpack message
                for (int j = 0; j < recv_count[i]; j++)
                    y[recv_list[i][j]] = recv_buffer[pV[i] + j];
            }

            for (int i = 0; i < size; i++)
            {
                if (send_count[i] == 0)
                    continue;
                MPI_Wait(&send_requests[i], MPI_STATUS_IGNORE);
            }
#endif

            double *tmp = x;
            x = y;
            y = tmp;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        double total_comm;
        MPI_Reduce(&comm_count, &total_comm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            for (int i = 1; i < size; i++)
            {
                MPI_Recv(x + pV[i], pV[i + 1] - pV[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            double a = 0.0;
            for (int i = 0; i < g.N; i++)
            {
                a = a < fabs(x[i]) ? fabs(x[i]) : a;
            }

            double L2 = 0.0;
            for (int i = 0; i < g.N; i++)
            {
                L2 += (x[i] / a) * (x[i] / a);
            }
            // L2 = a * sqrt(L2);

            // # nonzeroes multiplications and additions
            double ops = g.M * 2.0 * 100.0;
            // The size of G (|E| * (8 + 4) + |N| * 4)
            // And the size of the 2 vectors x and y, both doubles (8)
            double data = (g.M * 12.0 + g.N * 20.0) * 100.0;

            double comm = total_comm * 8;

            printf("%.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf\n",
                   (ops / (t1 - t0)) / 1e9,
                   ops / 1e9,
                   (data / (t1 - t0)) / 1e9,
                   (comm / (t1 - t0)) / 1e9,
                   comm / 1e9,
                   (t1 - t0),
                   L2);
        }
        else
        {
            MPI_Send(x + pV[rank], pV[rank + 1] - pV[rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

    free(pV);
    free(input);
    free(x);
    free(y);

    MPI_Finalize(); // End MPI, called by every processor
    return 0;
}