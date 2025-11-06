// mpi_hello.cpp
// A simple MPI program where multiple processes each print their rank and node name.

#include "mpi.h"      // Main MPI library header
#include <cstdio>     // For printf()
#include <cstdlib>    // For general C functions (optional but common)

// Main function — all MPI programs start from here
int main(int argc, char *argv[]) {

    // 1️⃣ Initialize the MPI environment.
    // This must be the first MPI call in any program.
    MPI_Init(&argc, &argv);

    // 2️⃣ Declare variables.
    int rank;         // Rank = unique ID for this process (0, 1, 2, ...)
    int size;         // Size = total number of processes
    int namelen;      // Length of the processor name
    char name[MPI_MAX_PROCESSOR_NAME];  // Name of the processor/node

    // 3️⃣ Get the rank (process ID) of this process.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 4️⃣ Get the total number of processes running.
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 5️⃣ Get the name of the processor (host name).
    MPI_Get_processor_name(name, &namelen);

    // 6️⃣ Print a message from each process.
    // Each process will execute this line independently.
    printf("Hello World from rank %d running on %s!\n", rank, name);

    // 7️⃣ Let only the root process (rank 0) print the total number of processes.
    if (rank == 0) {
        printf("MPI World size = %d processes\n", size);
    }

    // 8️⃣ Finalize the MPI environment.
    // This cleans up all MPI-related resources before exiting.
    MPI_Finalize();

    // 9️⃣ Return success.
    return 0;
}
