#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    const int N = 12;
    int *array = malloc(N * sizeof(int));
    
    // Initialize data on all processes
    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            array[i] = i + 1;  // [1, 2, 3, ..., 12]
        }
    }
    
    // Broadcast data to all processes
    MPI_Bcast(array, N, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Each process computes local sum of its chunk
    int worksize = N / size;
    int start = rank * worksize;
    int local_sum = 0;
    
    for (int i = start; i < start + worksize; i++) {
        local_sum += array[i];
    }
    
    printf("Process %d: local_sum = %d\n", rank, local_sum);
    
    // 🔥 MPI_Scan: Compute prefix sums
    int prefix_sum;
    MPI_Scan(&local_sum, &prefix_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    printf("Process %d: prefix_sum = %d\n", rank, prefix_sum);
    
    // 🔥 BONUS: Calculate starting offsets for each process
    int start_offset = prefix_sum - local_sum;
    printf("Process %d: start_offset = %d\n", rank, start_offset);
    
    free(array);
    MPI_Finalize();
    return 0;
}