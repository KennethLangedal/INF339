#include "spmv.h"
#include <stdlib.h>
#include <metis.h>
#include <string.h>
#include <mpi.h>

void spmv(graph g, double *x, double *y)
{
#pragma omp parallel for
    for (int u = 0; u < g.N; u++)
    {
        double z = 0.0;
        for (int i = g.V[u]; i < g.V[u + 1]; i++)
        {
            int v = g.E[i];
            z += x[v] * g.A[i];
        }
        y[u] = z;
    }
}

void spmv_part(graph g, int s, int t, double *x, double *y)
{
    for (int u = s; u < t; u++)
    {
        double z = 0.0;
        for (int i = g.V[u]; i < g.V[u + 1]; i++)
        {
            int v = g.E[i];
            z += x[v] * g.A[i];
        }
        y[u] = z;
    }
}

void partition_graph(graph g, int k, int *p, double *x)
{
    if (k == 1)
    {
        p[0] = 0;
        p[1] = g.N;
        return;
    }

    int ncon = 1;
    int objval;
    real_t ubvec = 1.01;
    int *part = malloc(sizeof(int) * g.N);
    int rc = METIS_PartGraphKway(&g.N, &ncon, g.V, g.E, NULL, NULL, NULL, &k, NULL, &ubvec, NULL, &objval, part);

    int *new_id = malloc(sizeof(int) * g.N);
    int *old_id = malloc(sizeof(int) * g.N);
    int id = 0;
    p[0] = 0;
    for (int r = 0; r < k; r++)
    {
        for (int i = 0; i < g.N; i++)
        {
            if (part[i] == r)
            {
                old_id[id] = i;
                new_id[i] = id++;
            }
        }
        p[r + 1] = id;
    }

    int *new_V = malloc(sizeof(int) * (g.N + 1));
    int *new_E = malloc(sizeof(int) * g.M);
    double *new_A = malloc(sizeof(double) * g.M);

    new_V[0] = 0;
    for (int i = 0; i < g.N; i++)
    {
        int d = g.V[old_id[i] + 1] - g.V[old_id[i]];
        new_V[i + 1] = new_V[i] + d;
        memcpy(new_E + new_V[i], g.E + g.V[old_id[i]], sizeof(int) * d);
        memcpy(new_A + new_V[i], g.A + g.V[old_id[i]], sizeof(double) * d);

        for (int j = new_V[i]; j < new_V[i + 1]; j++)
        {
            new_E[j] = new_id[new_E[j]];
        }
    }

    double *new_X = malloc(sizeof(double) * g.N);
    for (int i = 0; i < g.N; i++)
    {
        new_X[i] = x[old_id[i]];
    }

    memcpy(x, new_X, sizeof(double) * g.N);

    memcpy(g.V, new_V, sizeof(int) * (g.N + 1));
    memcpy(g.E, new_E, sizeof(int) * g.M);
    memcpy(g.A, new_A, sizeof(double) * g.M);

    free(new_V);
    free(new_E);
    free(new_A);
    free(new_X);

    free(new_id);
    free(old_id);
    free(part);
}

void partition_graph_naive(graph g, int s, int t, int k, int *p)
{
    int edges_per = (g.V[t] - g.V[s]) / k;
    p[0] = s;
    int id = 1;
    for (int u = s; u < t; u++)
    {
        if ((g.V[u] - g.V[s]) >= edges_per * id)
            p[id++] = u;
    }
    while (id <= k)
        p[id++] = t;
    p[k] = t;
}

void distribute_graph(graph *g, int rank)
{
    MPI_Bcast(&g->N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&g->M, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        g->V = malloc(sizeof(int) * (g->N + 1));
        g->E = malloc(sizeof(int) * g->M);
        g->A = malloc(sizeof(double) * g->M);
    }

    MPI_Bcast(g->V, g->N + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g->E, g->M, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g->A, g->M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

comm_lists init_comm_lists(int size)
{
    comm_lists c = {
        .send_count = malloc(sizeof(int) * size),
        .receive_count = malloc(sizeof(int) * size),
        .send_items = malloc(sizeof(int *) * size),
        .receive_items = malloc(sizeof(int *) * size),
        .send_lists = malloc(sizeof(double *) * size),
        .receive_lists = malloc(sizeof(double *) * size)};
    return c;
}

void free_comm_lists(comm_lists *c, int size)
{
    for (int i = 0; i < size; i++)
    {
        free(c->send_items[i]);
        free(c->send_lists[i]);
        free(c->receive_items[i]);
        free(c->receive_lists[i]);
    }

    free(c->send_count);
    free(c->receive_count);

    free(c->send_items);
    free(c->receive_items);

    free(c->send_lists);
    free(c->receive_lists);
}

void find_sendlists(graph g, int *p, int rank, int size, comm_lists c)
{
    int *send_mark = malloc(sizeof(int) * g.N);
    for (int r = 0; r < size; r++)
    {
        c.send_count[r] = 0;
        c.send_items[r] = NULL;
        c.send_lists[r] = NULL;
        if (r == rank)
            continue;

        // Set marks to zero
        for (int i = p[rank]; i < p[rank + 1]; i++)
            send_mark[i] = 0;

        // Find separators
        for (int u = p[r]; u < p[r + 1]; u++)
        {
            for (int i = g.V[u]; i < g.V[u + 1]; i++)
            {
                int v = g.E[i];
                if (v >= p[rank] && v < p[rank + 1])
                    send_mark[v] = 1;
            }
        }

        // Count separators
        for (int i = p[rank]; i < p[rank + 1]; i++)
            c.send_count[r] += send_mark[i];

        if (c.send_count[r] == 0)
            continue;

        // Store list of separators
        c.send_items[r] = malloc(sizeof(int) * c.send_count[r]);
        c.send_lists[r] = malloc(sizeof(double) * c.send_count[r]);

        int j = 0;
        for (int i = p[rank]; i < p[rank + 1]; i++)
            if (send_mark[i])
                c.send_items[r][j++] = i;
    }

    free(send_mark);
}

void find_receivelists(graph g, int *p, int rank, int size, comm_lists c)
{
    int *receive_mark = malloc(sizeof(int) * g.N);
    for (int r = 0; r < size; r++)
    {
        c.receive_count[r] = 0;
        c.receive_items[r] = NULL;
        c.receive_lists[r] = NULL;
        if (r == rank)
            continue;

        // Set marks to zero
        for (int i = p[r]; i < p[r + 1]; i++)
            receive_mark[i] = 0;

        // Find separators
        for (int u = p[rank]; u < p[rank + 1]; u++)
        {
            for (int i = g.V[u]; i < g.V[u + 1]; i++)
            {
                int v = g.E[i];
                if (v >= p[r] && v < p[r + 1])
                    receive_mark[v] = 1;
            }
        }

        // Count separators
        for (int i = p[r]; i < p[r + 1]; i++)
            c.receive_count[r] += receive_mark[i];

        if (c.receive_count[r] == 0)
            continue;

        // Store list of separators
        c.receive_items[r] = malloc(sizeof(int) * c.receive_count[r]);
        c.receive_lists[r] = malloc(sizeof(double) * c.receive_count[r]);

        int j = 0;
        for (int i = p[r]; i < p[r + 1]; i++)
            if (receive_mark[i])
                c.receive_items[r][j++] = i;
    }

    free(receive_mark);
}

void exchange_separators(comm_lists c, double *y, int rank, int size)
{
    MPI_Request sends[size], receives[size];
    // Start sends
    for (int r = 0; r < size; r++)
    {
        if (c.send_count[r] == 0)
            continue;
        for (int j = 0; j < c.send_count[r]; j++)
            c.send_lists[r][j] = y[c.send_items[r][j]];

        MPI_Isend(c.send_lists[r], c.send_count[r], MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &sends[r]);
    }
    // Start receives
    for (int r = 0; r < size; r++)
        if (c.receive_count[r] > 0)
            MPI_Irecv(c.receive_lists[r], c.receive_count[r], MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &receives[r]);

    // Wait for receives and unpack
    for (int r = 0; r < size; r++)
    {
        if (c.receive_count[r] > 0)
            MPI_Wait(&receives[r], MPI_STATUS_IGNORE);

        for (int j = 0; j < c.receive_count[r]; j++)
            y[c.receive_items[r][j]] = c.receive_lists[r][j];
    }

    // Wait for sends (could go after spmv)
    for (int r = 0; r < size; r++)
        if (c.send_count[r] > 0)
            MPI_Wait(&sends[r], MPI_STATUS_IGNORE);
}