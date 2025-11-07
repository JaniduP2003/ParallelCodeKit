#include <mpi.h>
#include <stdio.h>

#define DATA_SIZE 8  // total elements to distribute

int main(int argc, char **argv) {
    int rank, size;
    int data[DATA_SIZE];       // only root will fill this
    int recv_buf[2];           // each process receives 2 elements

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Initialize data only on root
        for (int i = 0; i < DATA_SIZE; i++)
            data[i] = i + 1; // [1,2,3,4,5,6,7,8]
        printf("Root data before scatter: ");
        for (int i = 0; i < DATA_SIZE; i++)
            printf("%d ", data[i]);
        printf("\n");
    }

    // Scatter: root sends 2 elements to each process
    MPI_Scatter(data, 2, MPI_INT, recv_buf, 2, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Process %d received: %d %d\n", rank, recv_buf[0], recv_buf[1]);

    MPI_Finalize();
    return 0;
}

/* 

Root (rank 0)
data: [1, 2, 3, 4, 5, 6, 7, 8]
 |        |        |        |
 v        v        v        v
P0<-{1,2} P1<-{3,4} P2<-{5,6} P3<-{7,8}
    {oth ,first}

    thats why 
        printf("Process %d received: %d %d\n", rank, recv_buf[0], recv_buf[1]);



*/