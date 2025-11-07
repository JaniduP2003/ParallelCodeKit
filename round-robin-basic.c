#include <stdio.h>
#include <mpi.h>

// Function prototype
void round_robin(int rank, int num_procs);

int main(int argc, char **argv) {
    int num_procs; // total number of processes
    int rank;      // rank (ID) of each process

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Get the rank of this process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Print a starting message
    //printf("%d: hello (total processes = %d)\n", rank, num_procs);

    // Call our round-robin task
    round_robin(rank, num_procs);

    // Print a finishing message
    //printf("%d: goodbye\n", rank);

    // Clean up and exit MPI
    MPI_Finalize();

    return 0;
}

// -------------------------------------------------------
// Example round-robin logic:
// Each process will "pass a token" to the next process.
// -------------------------------------------------------
void round_robin(int rank, int num_procs) {
    int token;

    if (rank == 0) {
        token = 100; // start the token with process 0
        printf("Process %d sending token %d to process %d\n", rank, token, (rank + 1) % num_procs);
        MPI_Send(&token, 1, MPI_INT, (rank + 1) % num_procs, 0, MPI_COMM_WORLD); //send to process 2
        MPI_Recv(&token, 1, MPI_INT, num_procs - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //resive forn 3 process
        printf("Process %d received token %d back from process %d\n", rank, token, num_procs - 1);
    } 
    else {
        MPI_Recv(&token, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // rank not 0
        // so we must first resive the data form previous process
        printf("Process %d received token %d from process %d\n", rank, token, rank - 1);
        // and need to send to the next note to 
        token++; // modify the token before passing it on
        MPI_Send(&token, 1, MPI_INT, (rank + 1) % num_procs, 0, MPI_COMM_WORLD);
    }
}
