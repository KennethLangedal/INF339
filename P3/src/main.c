#include "mtx.h"
#include "spmv.h"
#include <stdlib.h>
#include <math.h>
#include <mpi.h> // MPI header file

// TODO, fix possible e-16 in float parsing

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);               // starts MPI, called by every processor
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get current process id
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get number of processes

    if (rank != 0) {
        MPI_Finalize();
        return 0;
    }

    FILE *f = fopen(argv[1], "r");
    graph g = parse_mtx(f);
    fclose(f);

    sort_edges(g);
    normalize_graph(g);

    printf("%d %d\n", g.N, g.M);

    if (!validate_graph(g))
        printf("Error in graph\n");

    double *input = malloc(sizeof(double) * g.N);
    double *x = malloc(sizeof(double) * g.N);
    double *y = malloc(sizeof(double) * g.N);

    // Generate random input
    for (int i = 0; i < g.N; i++) {
        input[i] = ((double)rand() / (double)RAND_MAX) - 0.5;
    }

    int *p = malloc(sizeof(int) * (size + 1));
    partition_graph(g, 2, NULL, input);

    // Main loop start
    for (int e = 0; e < 3; e++) {

        for (int i = 0; i < g.N; i++) {
            x[i] = input[i];
        }

        double t0 = MPI_Wtime();

        for (int t = 0; t < 100; t++) {
            spmv(g, x, y);
            double *tmp = x;
            x = y;
            y = tmp;
        }

        double t1 = MPI_Wtime();

        double a = 0.0;
        for (int i = 0; i < g.N; i++) {
            a = a < fabs(x[i]) ? fabs(x[i]) : a;
        }

        double L2 = 0.0;
        for (int i = 0; i < g.N; i++) {
            L2 += (x[i] / a) * (x[i] / a);
        }
        // L2 = a * sqrt(L2);

        // # nonzeroes multiplications and additions
        double ops = g.M * 2.0 * 100.0;
        // The size of G (|E| * (8 + 4) + |N| * 4)
        // And the size of the 2 vectors x and y, both doubles (8)
        double data = (g.M * 12.0 + g.N * 20.0) * 100.0;

        printf("%.3lf %.3lf %.3lf\n", (ops / (t1 - t0)) / 1e9, (data / (t1 - t0)) / 1e9, L2);
    }

    free(p);
    free(input);
    free(x);
    free(y);
    free_graph(&g);

    MPI_Finalize(); // End MPI, called by every processor
    return 0;
}