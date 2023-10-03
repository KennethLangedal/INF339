#include "mpi.h"
#include <stdio.h>

int main(int argc, char **argv)
{

    MPI_Init(NULL, NULL);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int color = world_rank / 2;

    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &row_comm);

    int row_rank, row_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);

    int colcolor = world_rank % 2;
    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, colcolor, world_rank, &col_comm);

    int col_rank, col_size;
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);

    printf("WORLD RANK/SIZE: %d/%d \t ROW RANK/SIZE: %d/%d \t COL RANK/SIZE: %d/%d\n",
           world_rank, world_size, row_rank, row_size, col_rank, col_size);

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    MPI_Finalize(); // End MPI, called by every processor
    return 0;
}