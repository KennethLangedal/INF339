#include "spmv.h"
#include <stdlib.h>
#include <metis.h>
#include <string.h>

void spmv(graph g, double *x, double *y) {
#pragma omp parallel for schedule(dynamic, 1024)
    for (int u = 0; u < g.N; u++) {
        double z = 0.0;
        for (int i = g.V[u]; i < g.V[u + 1]; i++) {
            int v = g.E[i];
            z += x[v] * g.A[i];
        }
        y[u] = z;
    }
}

void partition_graph(graph g, int k, int *p, double *x) {
    int ncon = 1;
    int objval;
    real_t ubvec = 1.01;
    int *part = malloc(sizeof(int) * g.N);
    int rc = METIS_PartGraphKway(&g.N, &ncon, g.V, g.E, NULL, NULL, NULL, &k, NULL, &ubvec, NULL, &objval, part);

    int *new_id = malloc(sizeof(int) * g.N);
    int *old_id = malloc(sizeof(int) * g.N);
    int id = 0;
    p[0] = 0;
    for (int r = 0; r < k; r++) {
        for (int i = 0; i < g.N; i++) {
            if (part[i] == r) {
                old_id[id] = i;
                new_id[i] = id++;
            }
        }
        p[r + 1] = id;
    }

    int *new_V = malloc(sizeof(int) * (g.N + 1));
    int *new_E = malloc(sizeof(int) * g.M);
    int *new_A = malloc(sizeof(double) * g.M);

    new_id[0] = 0;
    for (int i = 0; i < g.N; i++) {
        new_V[i + 1] = new_V[i] + (g.V[old_id[i] + 1] - g.V[old_id[i]]);
        memcpy(new_E + new_V[i], g.E + g.V[old_id[i]], sizeof(int) * (new_V[i + 1] - new_V[i]));
        memcpy(new_A + new_V[i], g.A + g.V[old_id[i]], sizeof(double) * (new_V[i + 1] - new_V[i]));

        for (int j = new_V[i]; j < new_V[i + 1]; j++) {
            new_E[j] = new_id[new_E[j]];
        }
    }

    double *new_X = malloc(sizeof(double) * g.N);
    for (int i = 0; i < g.N; i++) {
        new_X[i] = x[old_id[i]];
    }

    memcpy(x, new_X, sizeof(double) * g.N);

    memcpy(g.V, new_V, sizeof(int) * (g.N + 1));
    memcpy(g.E, new_E, sizeof(int) * g.M);
    memcpy(g.A, new_A, sizeof(double) * g.M);

    free(new_V);
    free(new_E);
    free(new_A);

    free(new_id);
    free(old_id);
    free(part);
}