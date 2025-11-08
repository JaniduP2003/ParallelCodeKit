#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define IMAGE_COUNT 1000
#define WIDTH 800
#define HEIGHT 600
#define PIXEL_COUNT (WIDTH * HEIGHT)

// Function to merge a set of images (e.g., pixel-wise average)
void merge_images(float images[][PIXEL_COUNT], float result[PIXEL_COUNT], int start, int end) {
    int count = end - start;
    for (int i = 0; i < PIXEL_COUNT; i++) {
        float sum = 0.0f;
        for (int j = start; j < end; j++) {
            sum += images[j][i];
        }
        result[i] = sum / count;  // Average pixel value
    }
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate how many images each process handles
    int images_per_proc = IMAGE_COUNT / size;

    // Allocate memory dynamically (to avoid stack overflow)
    float *images = NULL;
    float *local_images = (float *)malloc(images_per_proc * PIXEL_COUNT * sizeof(float));
    float *merged = NULL;
    float *local_merged = (float *)calloc(PIXEL_COUNT, sizeof(float));

    // Root process allocates and initializes all images
    if (rank == 0) {
        images = (float *)malloc(IMAGE_COUNT * PIXEL_COUNT * sizeof(float));

        // Initialize all images with dummy data
        for (int i = 0; i < IMAGE_COUNT * PIXEL_COUNT; i++) {
            images[i] = (float)(i % 256); // Example: grayscale pattern
        }
    }

    // Scatter the images from root to all processes
    MPI_Scatter(
        images, images_per_proc * PIXEL_COUNT, MPI_FLOAT,
        local_images, images_per_proc * PIXEL_COUNT, MPI_FLOAT,
        0, MPI_COMM_WORLD
    );

    // Each process merges its subset of images
    // local_images is treated as 2D: [images_per_proc][PIXEL_COUNT]
    merge_images((float (*)[PIXEL_COUNT])local_images, local_merged, 0, images_per_proc);

    // Root prepares final merged image buffer
    if (rank == 0) {
        merged = (float *)calloc(PIXEL_COUNT, sizeof(float));
    }

    // Combine all local merged images at the root using SUM reduction
    MPI_Reduce(local_merged, merged, PIXEL_COUNT, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Root averages across processes (optional, to normalize)
    if (rank == 0) {
        for (int i = 0; i < PIXEL_COUNT; i++) {
            merged[i] /= size;  // average of all process results
        }

        // Display confirmation
        printf("Final merged image computed successfully.\n");
        printf("Example pixels: %.2f %.2f %.2f ...\n", merged[0], merged[1], merged[2]);
    }

    // Free memory
    free(local_images);
    free(local_merged);
    if (rank == 0) {
        free(images);
        free(merged);
    }

    MPI_Finalize();
    return 0;
}


/*

                ┌─────────────────────────────┐
                │ Root (Rank 0): 1000 images   │
                └────────────┬────────────────┘
                             │ Scatter
         ┌───────────────────┼───────────────────┐
         v                   v                   v
    Rank 1              Rank 2              Rank 3
 [250 imgs]          [250 imgs]          [250 imgs]
      │                   │                   │
 merge_images()      merge_images()      merge_images()
      │                   │                   │
      └─────────── MPI_Reduce (SUM) ───────────┘
                             │
                             v
              Root (Rank 0): merged final image


*/