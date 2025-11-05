#include <stdio.h>

// A simple CUDA kernel that prints from the GPU
__global__ void hello_from_gpu() {
    printf("Hello World from GPU! (thread %d, block %d)\n", threadIdx.x, blockIdx.x);
}

int main() {
    printf("Hello World from CPU!\n");

    // Launch 5 blocks with 10 threads each
    hello_from_gpu<<<5, 10>>>();
    cudaDeviceSynchronize();  // wait for GPU to finish

    return 0;
}
