#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 1000000;
    int *array = malloc(N * sizeof(int));

    // Rank 0 initializes the data
    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            array[i] = i + 1;
        }
    }

    // Broadcast the full array from rank 0 to all ranks
    MPI_Bcast(array, N, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process computes a local sum over its assigned segment
    int worksize = N / size;
    int start = rank * worksize;
    int local_sum = 0;

    for (int i = start; i < start + worksize; i++) {
        local_sum += array[i];
    }

    // 🔥 ALLREDUCE VERSION - EVERY PROCESS GETS THE TOTAL SUM 🔥
    int total_sum;
    MPI_Allreduce(&local_sum, &total_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Now ALL processes have the total_sum!
    printf("Process %d: Total sum = %d\n", rank, total_sum);

    free(array);
    MPI_Finalize();
    return 0;
}