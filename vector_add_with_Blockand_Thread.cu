#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (2048*2048)        // Number of elements (~4.2 million)
#define THREADS_PER_BLOCK 512 // Threads per block

// GPU kernel function — each thread adds one element
__global__ void add(int *a, int *b, int *c) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {  // Make sure we don't go out of bounds • Avoid accessing beyond the end of the arrays:
        c[index] = a[index] + b[index];
    }
}

// Helper function to fill an array with random integers 0-99
void random_ints(int *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = rand() % 100;
    }
}

int main(void) {
    int *a, *b, *c;           // Host copies of vectors
    int *d_a, *d_b, *d_c;     // Device copies of vectors
    int size = N * sizeof(int);

    // 1. Allocate space on the GPU
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // 2. Allocate and initialize host data
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // Seed random number generator
    srand(time(NULL));

    random_ints(a, N);
    random_ints(b, N);

    // 3. Copy input vectors from CPU to GPU
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // 4. Launch the kernel on the GPU
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; //Avoid accessing beyond the end 
    //of the arrays:
    add<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

    // 5. Copy the result vector back to CPU memory
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // 6. Display a few results to verify
    printf("Sample results:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // 7. Cleanup
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
