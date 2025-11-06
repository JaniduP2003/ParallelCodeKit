#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel: each thread processes one element in a 2D array
__global__ void scale2D(float *input, float *output, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Y- Calculates the thread’s row index in the 2D array
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // X- Calculates the thread’s column index in the 2D array

    // Check if within bounds
    if (row < height && col < width) {
        int idx = row * width + col;   // Convert 2D index to 1D
        //If width = 8 and you want the element at (row=2, col=3), the linear index is:
        // idx = 2 * 8 + 3 = 19
        output[idx] = 2.0f * input[idx];
    }
}

int main() {
    const int width = 8;
    const int height = 8;
    const int size = width * height * sizeof(float);

    float h_input[width * height];
    float h_output[width * height];

    // Initialize input data
    for (int i = 0; i < width * height; i++)
        h_input[i] = (float)i;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);

    // Copy data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    // Launch kernel
    scale2D<<<grid, block>>>(d_input, d_output, width, height);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print results
    printf("Input → Output:\n");
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int i = r * width + c;
            printf("%4.1f→%4.1f  ", h_input[i], h_output[i]);
        }
        printf("\n");
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
