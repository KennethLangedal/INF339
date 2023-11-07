#include "mtx.h"
#include <stdlib.h>
#include <metis.h>
#include <mpi.h> // MPI header file

// TODO, fix possible e-16 in float parsing

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);               // starts MPI, called by every processor
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get current process id
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get number of processes

    FILE *f = fopen(argv[1], "r");
    graph g = parse_mtx(f);
    fclose(f);

    sort_edges(g);

    printf("%d %d\n", g.N, g.M);

    if (!validate_graph(g))
        printf("Error in graph\n");

    int ncon = 1;
    int k = 4;
    int objval;
    int *part = malloc(sizeof(int) * g.N);
    METIS_PartGraphKway(&g.N, &ncon, g.V, g.E, NULL, NULL, NULL, &k, NULL, NULL, NULL, &objval, part);

    printf("%d\n", objval);
    free(part);

    free_graph(&g);

    MPI_Finalize(); // End MPI, called by every processor
    return 0;
}