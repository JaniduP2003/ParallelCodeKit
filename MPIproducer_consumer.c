#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>  // for sleep()

int main(int argc, char *argv[]) {
    int rank, size;
    int data[5];
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            printf("This program requires at least 2 processes!\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Producer (rank 0) - Consumer (rank 1) pattern
    if (rank == 0) {
        // PRODUCER: Generates data and sends it
        printf("Producer (rank 0): Starting to generate data...\n");
        
        for (int i = 0; i < 3; i++) {
            // Generate some data
            for (int j = 0; j < 5; j++) {
                data[j] = i * 10 + j;  // Example data
            }
            
            printf("Producer: Generated data batch %d: [%d, %d, %d, %d, %d]\n", 
                   i, data[0], data[1], data[2], data[3], data[4]);
            
            printf("Producer: Sending data to consumer (using MPI_Ssend)...\n");
            
            // 🔥 KEY: Using MPI_Ssend - waits until consumer is ready to receive
            MPI_Ssend(data, 5, MPI_INT, 1, 0, MPI_COMM_WORLD);
            
            printf("Producer: Data batch %d successfully delivered to consumer!\n", i);
            printf("Producer: Consumer confirmed receipt before I continue.\n\n");
            
            sleep(1);  // Simulate some work time
        }
        
        // Send termination signal
        data[0] = -1;  // Termination signal
        MPI_Ssend(data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD); ////////////////////////////////////
        printf("Producer: Sent termination signal. Done!\n");
        
    } else if (rank == 1) {
        // CONSUMER: Receives and processes data
        int batch_count = 0;
        
        printf("Consumer (rank 1): Ready to receive data...\n\n");
        
        while (1) {
            // Simulate variable processing time
            int process_time = 2 + (rand() % 3);  // 2-4 seconds
            printf("Consumer: Will take %d seconds to process next batch...\n", process_time);
            sleep(process_time);
            
            printf("Consumer: Ready to receive now...\n");
            
            // Receive data
            MPI_Recv(data, 5, MPI_INT, 0, 0, MPI_COMM_WORLD, &status); 
            
            // Check for termination
            if (data[0] == -1) {
                printf("Consumer: Received termination signal. Processed %d batches total.\n", batch_count);
                break;
            }
            
            batch_count++;
            printf("Consumer: Received batch %d: [%d, %d, %d, %d, %d]\n", 
                   batch_count, data[0], data[1], data[2], data[3], data[4]);
            printf("Consumer: Processing complete! Ready for next batch.\n\n");
        }
    } else {
        // Other processes just wait
        printf("Process %d: I'm not involved in this producer-consumer setup.\n", rank);
    }

    MPI_Finalize();
    return 0;
}