#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel: doubles each input value
__global__ void scale2D(float *input, float *output, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int idx = row * width + col;
        output[idx] = 2.0f * input[idx];
        //The code initializes the input array with values starting at 0 up to 63 
        //(because width * height = 8 * 8 = 64).
    }
}

// Print each block’s threads (like before)
void printGridLayout(int width, int height, dim3 grid, dim3 block) {
    printf("\nGrid(%d,%d) with Block(%d,%d)\n", grid.x, grid.y, block.x, block.y);
    printf("=========================================\n");

    for (int by = 0; by < grid.y; by++) {
        for (int bx = 0; bx < grid.x; bx++) {
            printf(" Block(%d,%d):\n", bx, by);

            for (int ty = 0; ty < block.y; ty++) {
                for (int tx = 0; tx < block.x; tx++) {
                    int gx = bx * block.x + tx;
                    int gy = by * block.y + ty;

                    if (gx < width && gy < height)
                        printf("(%d,%d) ", gx, gy);
                    else
                        printf("   -   ");
                }
                printf("\n");
            }
            printf("\n");
        }
    }
}

// 🧩 Print the combined 2D view of all blocks
void printFullImageView(int width, int height, dim3 grid, dim3 block) {
    printf("=========================================\n");
    printf(" Combined Full Image View (All Threads Mapped)\n");
    printf("=========================================\n");

    for (int gy = 0; gy < height; gy++) {
        for (int gx = 0; gx < width; gx++) {
            printf("(%d,%d)", gx, gy);

            // separate each column of blocks
            if ((gx + 1) % block.x == 0 && gx != width - 1)
                printf(" | ");
            else
                printf(" ");
        }
        printf("\n");

        // horizontal separator between block rows
        if ((gy + 1) % block.y == 0 && gy != height - 1) {
            for (int i = 0; i < width; i++) {
                printf("------");
                if ((i + 1) % block.x == 0 && i != width - 1)
                    printf("+");
            }
            printf("\n");
        }
    }
    printf("=========================================\n");
}

int main() {
    const int width = 10;
    const int height = 10;
    const int size = width * height * sizeof(float);

    float h_input[width * height];
    float h_output[width * height];

    for (int i = 0; i < width * height; i++)
        h_input[i] = (float)i;

    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 block(4, 4);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    // Step 1: Print per-block breakdown
    printGridLayout(width, height, grid, block);

    // Step 2: Print combined visual map
    printFullImageView(width, height, grid, block);

    // Run kernel
    scale2D<<<grid, block>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Step 3: Show results
    printf("Computation Results (Input → Output):\n");
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int i = r * width + c;
            printf("%4.0f→%4.0f ", h_input[i], h_output[i]);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
