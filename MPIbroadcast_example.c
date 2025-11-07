#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int rank, size;
    int number;  // variable to be broadcasted

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        number = 42;  // Root initializes the value
        printf("Root process (rank %d) broadcasting number = %d\n", rank, number);
    }

    // Broadcast the variable "number" from root (0) to all processes
    MPI_Bcast(&number, 1, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Process %d received number = %d\n", rank, number);

    MPI_Finalize();
    return 0;
}

/*
Root (rank 0): number = 42
   | 
   +--> rank 1: receives 42
   +--> rank 2: receives 42
   +--> rank 3: receives 42


*/