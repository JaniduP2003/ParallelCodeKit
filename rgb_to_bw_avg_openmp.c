// rgb_to_bw_avg_openmp.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define WIDTH 800
#define HEIGHT 600

int main(void) {
    int x, y;
    double t_start, t_end;

    // Allocate 2D arrays dynamically to avoid stack overflow
    float (*Red)[HEIGHT]  = malloc(sizeof(float[WIDTH][HEIGHT]));
    float (*Green)[HEIGHT] = malloc(sizeof(float[WIDTH][HEIGHT]));
    float (*Blue)[HEIGHT]  = malloc(sizeof(float[WIDTH][HEIGHT]));
    float (*BW)[HEIGHT]    = malloc(sizeof(float[WIDTH][HEIGHT]));

    if (!Red || !Green || !Blue || !BW) {
        perror("Memory allocation failed");
        return 1;
    }

    // Initialize RGB arrays with random values [0,255]
    unsigned int seed = 1234;
    for (x = 0; x < WIDTH; ++x) {
        for (y = 0; y < HEIGHT; ++y) {
            Red[x][y]   = rand_r(&seed) % 256;
            Green[x][y] = rand_r(&seed) % 256;
            Blue[x][y]  = rand_r(&seed) % 256;
        }
    }

    double sum = 0.0;

    // --- Parallel region: convert and sum using reduction ---
    t_start = omp_get_wtime();

    #pragma omp parallel for collapse(2) reduction(+:sum)
    for (x = 0; x < WIDTH; ++x) {
        for (y = 0; y < HEIGHT; ++y) {
            BW[x][y] = 0.21f * Red[x][y] + 0.72f * Green[x][y] + 0.07f * Blue[x][y];
            sum += BW[x][y];
        }
    }

    t_end = omp_get_wtime();

    double average = sum / (WIDTH * HEIGHT);

    printf("Parallel conversion completed in %.6f seconds\n", t_end - t_start);
    printf("Average BW pixel value = %.2f\n", average);

    // Optional: verify a few pixels
    printf("Sample pixels:\n");
    for (int i = 0; i < 3; ++i)
        printf("BW[%d][%d] = %.2f\n", i, i, BW[i][i]);

    free(Red);
    free(Green);
    free(Blue);
    free(BW);

    return 0;
}

