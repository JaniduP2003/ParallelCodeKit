#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024        // total elements
#define R 3           // stencil radius (neighborhood size)
#define BLOCK_SIZE 256  // threads per block

// -----------------------------
// CUDA Kernel: 1D Stencil
// -----------------------------
__global__ void stencil(int *in, int *out) {
    // Shared memory for data + halo
    __shared__ int tmp[BLOCK_SIZE + 2 * R];

    // Compute global and local indices
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int l = threadIdx.x + R;

    // Load the main data element
    tmp[l] = in[g];

    // Load left and right halo elements
    if (threadIdx.x < R) {
        tmp[l - R] = in[g - R];
        tmp[l + blockDim.x] = in[g + blockDim.x];
    }

    // Synchronize threads to ensure shared memory is ready
    __syncthreads();

    // Compute stencil sum
    int sum = 0;
    for (int o = -R; o <= R; o++) {
        sum += tmp[l + o];
    }

    // Store result
    out[g] = sum;
}

// -----------------------------
// Main Function
// -----------------------------
int main() {
    int *h_in, *h_out;       // Host memory
    int *d_in, *d_out;       // Device memory
    size_t bytes = N * sizeof(int);

    // Allocate host memory
    h_in = (int *)malloc(bytes);
    h_out = (int *)malloc(bytes);

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_in[i] = i + 1;
    }

    // Allocate device memory
    cudaMalloc((void **)&d_in, bytes);
    cudaMalloc((void **)&d_out, bytes);

    // Copy input data from host to device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // Define execution configuration
    int gridSize = N / BLOCK_SIZE;

    // Launch kernel
    stencil<<<gridSize, BLOCK_SIZE>>>(d_in, d_out);

    // Copy results back to host
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Display sample output
    printf("1D Stencil Output (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("out[%d] = %d\n", i, h_out[i]);
    }

    // Cleanup
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
