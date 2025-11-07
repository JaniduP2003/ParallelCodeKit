#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    printf("Process %d: Started my work...\n", rank);
    
    // Each process does different amounts of work
    int work_time = rank + 1;  // Process 0: 1s, Process 1: 2s, etc.
    sleep(work_time);
    
    printf("Process %d: Finished my work in %d seconds\n", rank, work_time);
    
    // 🔥 BARRIER SYNCHRONIZATION 🔥
    printf("Process %d: Waiting at barrier...\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Process %d: Barrier crossed! All processes synchronized!\n", rank);
    
    // Now all processes continue together
    printf("Process %d: Continuing after synchronization\n", rank);
    
    MPI_Finalize();
    return 0;
}

/*

TIMELINE:    PROCESS 0     PROCESS 1     PROCESS 2     PROCESS 3
------------------------------------------------------------------
T=0s:       "Started..."  "Started..."  "Started..."  "Started..."
            ⏳ work 1s    ⏳ work 2s    ⏳ work 3s    ⏳ work 4s

T=1s:       ✅ Finished!  ⏳ working...  ⏳ working...  ⏳ working...
            "Waiting..."  

T=2s:       ⏸️ WAITING    ✅ Finished!   ⏳ working...  ⏳ working...
            ⏸️ WAITING    "Waiting..."  

T=3s:       ⏸️ WAITING    ⏸️ WAITING    ✅ Finished!   ⏳ working...
            ⏸️ WAITING    ⏸️ WAITING    "Waiting..."  

T=4s:       ⏸️ WAITING    ⏸️ WAITING    ⏸️ WAITING    ✅ Finished!
            ⏸️ WAITING    ⏸️ WAITING    ⏸️ WAITING    "Waiting..."

T=4.1s:     🎉 ALL CROSS! 🎉 ALL CROSS! 🎉 ALL CROSS! 🎉 ALL CROSS!
            "Continuing"  "Continuing"  "Continuing"  "Continuing"

*/