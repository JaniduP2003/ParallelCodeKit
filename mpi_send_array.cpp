#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);  // Initialize MPI environment

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get current process ID (rank)
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get total number of processes

    // Ensure we are running with exactly 2 processes
    if (size != 2) {
        if (rank == 0)
            printf("Please run this program with 2 processes only.\n");
        MPI_Finalize();
        return 0;
    }

    const int N = 5;       // Size of the array
    int data[N];           

    if (rank == 0) {
        // Initialize array with values
        for (int i = 0; i < N; i++)
            data[i] = i + 1;

        printf("Process %d sending array: ", rank);
        for (int i = 0; i < N; i++) printf("%d ", data[i]);
        printf("\n");

        // Send array to process 1
        MPI_Send(data, N, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } 
    else if (rank == 1) {
        // Receive array from process 0
        MPI_Recv(data, N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Process %d received array: ", rank);
        for (int i = 0; i < N; i++) printf("%d ", data[i]);
        printf("\n");
    }

    MPI_Finalize();  // Clean up MPI environment
    return 0;
}
