#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size, n = 1000000;
    double h, local_sum = 0.0, total_sum = 0.0;
    double start, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    h = 1.0 / (double)n;

    // Each process handles a subset of the range
    int start_index = rank * (n / size);
    int end_index   = (rank + 1) * (n / size);

    start = MPI_Wtime();
    ///CALCULATIONS----------------------------------------------------
    for (int i = start_index; i < end_index; i++) {
        double x = (i + 0.5) * h;
        local_sum += 4.0 / (1.0 + x * x);
    }
    ///CALCULATIONS----------------------------------------------------
    end = MPI_Wtime();

    // Combine all partial sums into total_sum at rank 0
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double pi = h * total_sum;
        printf("Estimated π = %.12f\n", pi);
        printf("Time taken = %.6f seconds\n", end - start);
    }

    MPI_Finalize();
    return 0;
}

/*
Serial (1 process):
[████████████████████████████████████████] 1,000,000 rectangles

Parallel (4 processes):
[█████████████] [█████████████] [█████████████] [█████████████]
   Process 0       Process 1       Process 2       Process 3
   250,000 rect    250,000 rect    250,000 rect    250,000 rect

   the size variable is 
   in mpirun -np 8 ./MPIpi_mpi size is 8
   in mpirun -np 4 ./MPIpi_mpi size is 4
*/