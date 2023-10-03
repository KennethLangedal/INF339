#include <stdio.h>
#include <mpi.h> // MPI header file

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);               // starts MPI, called by every processor
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get current process id
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get number of processes

    printf("Hello World %d %d\n", rank, size);

    MPI_Finalize(); // End MPI, called by every processor
    return 0;
}