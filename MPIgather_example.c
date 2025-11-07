#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int rank, size;
    int send_buf[2];       // each process sends 2 numbers
    int recv_buf[8];       // root will collect from everyone

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Each process prepares its data
    send_buf[0] = rank * 2 + 1;
    send_buf[1] = rank * 2 + 2;

    printf("Process %d sending: %d %d\n", rank, send_buf[0], send_buf[1]);

    // Gather: root collects from everyone
    MPI_Gather(send_buf, 2, MPI_INT,
               recv_buf, 2, MPI_INT,
               1, MPI_COMM_WORLD);

    if (rank == 1) {
        printf("\nRoot gathered data: ");
        for (int i = 0; i < 8; i++)
            printf("%d ", recv_buf[i]);
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
