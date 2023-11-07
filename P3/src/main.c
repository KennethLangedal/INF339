#include "mtx.h"
#include <mpi.h> // MPI header file

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

    if (!validate_graph(g))
        printf("Error in graph\n");

    

    free_graph(&g);

    MPI_Finalize(); // End MPI, called by every processor
    return 0;
}