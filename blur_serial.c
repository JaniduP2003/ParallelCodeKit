#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WIDTH 800
#define HEIGHT 600

int main() {
    int x, y, i, j;

    // Allocate 2D arrays dynamically
    float (*Red)[HEIGHT]   = malloc(sizeof(float[WIDTH][HEIGHT]));
    float (*Green)[HEIGHT] = malloc(sizeof(float[WIDTH][HEIGHT]));
    float (*Blue)[HEIGHT]  = malloc(sizeof(float[WIDTH][HEIGHT]));

    float (*RedBlur)[HEIGHT]   = malloc(sizeof(float[WIDTH][HEIGHT]));
    float (*GreenBlur)[HEIGHT] = malloc(sizeof(float[WIDTH][HEIGHT]));
    float (*BlueBlur)[HEIGHT]  = malloc(sizeof(float[WIDTH][HEIGHT]));

    if (!Red || !Green || !Blue || !RedBlur || !GreenBlur || !BlueBlur) {
        perror("Memory allocation failed");
        return 1;
    }

    // Initialize RGB arrays with random intensity values [0, 255]
    srand(1234);
    for (x = 0; x < WIDTH; ++x) {
        for (y = 0; y < HEIGHT; ++y) {
            Red[x][y]   = rand() % 256;
            Green[x][y] = rand() % 256;
            Blue[x][y]  = rand() % 256;
        }
    }

    // 3x3 averaging kernel
    const float kernel[3][3] = {
        {1/9.0f, 1/9.0f, 1/9.0f},
        {1/9.0f, 1/9.0f, 1/9.0f},
        {1/9.0f, 1/9.0f, 1/9.0f}
    };

    clock_t start = clock();

    // --- Sequential blur computation ---
    for (x = 1; x < WIDTH - 1; ++x) {
        for (y = 1; y < HEIGHT - 1; ++y) {
            float sumR = 0, sumG = 0, sumB = 0;

            // Apply 3×3 kernel
            for (i = -1; i <= 1; ++i) {
                for (j = -1; j <= 1; ++j) {
                    sumR += Red[x + i][y + j] * kernel[i + 1][j + 1];
                    sumG += Green[x + i][y + j] * kernel[i + 1][j + 1];
                    sumB += Blue[x + i][y + j] * kernel[i + 1][j + 1];
                }
            }

            RedBlur[x][y]   = sumR;
            GreenBlur[x][y] = sumG;
            BlueBlur[x][y]  = sumB;
        }
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Image blur completed (Sequential)\n");
    printf("Execution time: %.4f seconds\n", elapsed);

    // Optional: print sample blurred pixels
    for (int k = 0; k < 3; ++k)
        printf("Blur[%d][%d] = (%.2f, %.2f, %.2f)\n",
               k+1, k+1, RedBlur[k+1][k+1], GreenBlur[k+1][k+1], BlueBlur[k+1][k+1]);

    free(Red); free(Green); free(Blue);
    free(RedBlur); free(GreenBlur); free(BlueBlur);

    return 0;
}
