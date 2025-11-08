#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 8;  // total elements to scatter
    int sendbuf[N];
    int recvcount = N / size;
    int recvbuf[recvcount];

    if (rank == 0) {
        // Initialize send buffer
        for (int i = 0; i < N; i++) sendbuf[i] = i + 1;

        // Root sends chunks
        for (int dest = 0; dest < size; dest++) {
            if (dest == 0) {
                // Copy its own portion locally
                for (int j = 0; j < recvcount; j++)
                    recvbuf[j] = sendbuf[j];
            } else {
                // Send appropriate chunk synchronously
                MPI_Ssend(&sendbuf[dest * recvcount], recvcount, MPI_INT,
                          dest, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        // Non-root processes receive their chunk
        MPI_Recv(recvbuf, recvcount, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Print received data
    printf("Rank %d received:", rank);
    for (int i = 0; i < recvcount; i++) printf(" %d", recvbuf[i]);
    printf("\n");

    MPI_Finalize();
    return 0;
}

/*

Time | Process 0 (Root)           | Process 1           | Process 2           | Process 3
-----|----------------------------|---------------------|---------------------|-------------------
T0   | Initialize sendbuf[1..8]   | MPI_Init complete   | MPI_Init complete   | MPI_Init complete
T1   | Copy [1,2] to recvbuf      | MPI_Recv (blocking) | MPI_Recv (blocking) | MPI_Recv (blocking)
T2   | Send [3,4] to Process 1    | ← Receives [3,4]    | Waiting...          | Waiting...
T3   | Send [5,6] to Process 2    | Processing [3,4]    | ← Receives [5,6]    | Waiting...
T4   | Send [7,8] to Process 3    | Processing [3,4]    | Processing [5,6]    | ← Receives [7,8]
T5   | Print: "Rank 0: 1 2"       | Print: "Rank 1: 3 4"| Print: "Rank 2: 5 6"| Print: "Rank 3: 7 8"

*/