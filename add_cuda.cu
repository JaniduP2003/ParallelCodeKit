#include <stdio.h>
#include <cuda_runtime.h>

// GPU kernel function
__global__ void add(int *a, int *b, int *c) {
    // Each thread does one addition
    *c = *a + *b;
}

int main(void) {
    int a, b, c;               // host (CPU) copies of a, b, c
    int *d_a, *d_b, *d_c;      // device (GPU) copies of a, b, c
    int size = sizeof(int);    // size of each integer

    // 1. Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // 2. Setup input values on the host (CPU)
    a = 2;
    b = 7;

    // 3. Copy inputs from host to device
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    // 4. Launch the kernel on the GPU with 1 block and 1 thread
    add<<<1, 1>>>(d_a, d_b, d_c);

    // 5. Copy the result back from device to host
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

    // 6. Print the result
    printf("Result: %d + %d = %d\n", a, b, c);

    // 7. Cleanup GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
