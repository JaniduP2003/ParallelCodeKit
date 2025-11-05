#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 5  // number of elements in each vector

// GPU kernel function — runs in parallel on the GPU
__global__ void add(int *a, int *b, int *c) {
    // Each thread adds one element
    int index = blockIdx.x;   // each block handles one element
    c[index] = a[index] + b[index];
}

// Helper function to fill an array with random integers
void random_ints(int *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = rand() % 100;   //The modulus operator (%) is a simple way to "wrap" a 
        //large number into a smaller range:IF rand(12345) output THEN  12345 % 100 = 45
    }
}

int main(void) {
    int *a, *b, *c;           // host (CPU) copies of vectors
    int *d_a, *d_b, *d_c;     // device (GPU) copies of vectors
    int size = N * sizeof(int);

    // 1. Allocate space on the GPU
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // 2. Allocate and initialize host data
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);
    random_ints(a, N);
    random_ints(b, N);

    // 3. Copy input vectors from CPU to GPU
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // 4. Launch the kernel on the GPU
    //    <<<N, 1>>> means: N blocks, 1 thread per block
    add<<<N, 1>>>(d_a, d_b, d_c);

    // 5. Copy the result vector back to CPU memory
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // 6. Display a few results to verify
    printf("Sample results:\n");
    for (int i = 0; i < 4; i++) {
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
