#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 1000000;
    int *array = malloc(N * sizeof(int));  // Allocate full array on all processes

    // Rank 0 initializes the data
    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            array[i] = i + 1;  // Simple data initialization
        }
    }

    // Broadcast the full array from rank 0 to all ranks
    MPI_Bcast(array, N, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process computes a local sum over its assigned segment
    int worksize = N / size;
    int start = rank * worksize;
    int local_sum = 0;

    for (int i = start; i < start + worksize; i++) {    //END IS CACULATED HERE 
        local_sum += array[i];
    }

    MPI_Status status;
    int total_sum = 0;

    if (rank != 0) {
        // Non-root processes send their local sum to rank 0
        MPI_Send(&local_sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
        // Rank 0 receives all partial sums and combines them
        total_sum = local_sum;

        for (int r = 1; r < size; r++) {
            int recv_sum;
            MPI_Recv(&recv_sum, 1, MPI_INT, r, 0, MPI_COMM_WORLD, &status);
            total_sum += recv_sum;
        }

        printf("Total sum = %d\n", total_sum);
    }

    free(array);
    MPI_Finalize();
    return 0;
}

/*

N = 12, Size = 4 processes
Array should contain: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

ALL PROCESSES: malloc(12 * sizeof(int)) → Each gets empty array
               [?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?]

PROCESS 0 ONLY: Fills array with values
               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

PROCESS 0: ────📦 BROADCAST ────→ ALL PROCESSES
           Sends: [1,2,3,4,5,6,7,8,9,10,11,12]

NOW ALL PROCESSES HAVE:
Process 0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
Process 1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  
Process 2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
Process 3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

worksize = N/size = 12/4 = 3 elements per process

Process 0: start = 0*3 = 0, end = 0+3 = 3 → elements [0,1,2]
Process 1: start = 1*3 = 3, end = 3+3 = 6 → elements [3,4,5]  
Process 2: start = 2*3 = 6, end = 6+3 = 9 → elements [6,7,8]
Process 3: start = 3*3 = 9, end = 9+3 = 12 → elements [9,10,11]

Process 0: sums [1, 2, 3]     → local_sum = 6
Process 1: sums [4, 5, 6]     → local_sum = 15  
Process 2: sums [7, 8, 9]     → local_sum = 24
Process 3: sums [10, 11, 12]  → local_sum = 33

WORKERS SEND TO MASTER:
Process 1: ────📤 15 ────→ Process 0
Process 2: ────📤 24 ────→ Process 0  
Process 3: ────📤 33 ────→ Process 0

MASTER COMBINES:
Process 0: total_sum = 6 (own) + 15 + 24 + 33 = 78

TIME    PROCESS 0 (MASTER)      PROCESS 1 (WORKER)      PROCESS 2 (WORKER)      PROCESS 3 (WORKER)
---------------------------------------------------------------------------------------------------
T=0     Allocates array         Allocates array         Allocates array         Allocates array
        Fills: [1,2,3,...,12]   Waits...                Waits...                Waits...

T=1     ╔═══════════════════════ MPI_BCAST ═══════════════════════════════════════════════════════╗
        ║     Sends [1..12]     →  Receives [1..12]   → Receives [1..12]   → Receives [1..12]     ║
        ╚═════════════════════════════════════════════════════════════════════════════════════════╝

T=2     Computes: 1+2+3=6       Computes: 4+5+6=15     Computes: 7+8+9=24     Computes: 10+11+12=33

T=3     Waits for results...    ────SENDS 15────→       ────SENDS 24────→       ────SENDS 33────→
        ╔══════════════════════════════════════════════════════════════════════════════════════╗
        ══COLLECTS══→ 6 + 15 + 24 + 33 = 78 ═══════════════════════════════════════════════════════╣
        ╚══════════════════════════════════════════════════════════════════════════════════════════╝

T=4     total_sum = 78 ✅       Done!                 Done!                 Done!



*/