#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define WIDTH 1920
#define HEIGHT 1080
#define SIZE (WIDTH * HEIGHT)

__global__ void mergeImagesKernel(unsigned char *ImageA, unsigned char *ImageB, unsigned char *ImageC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < SIZE) {
        ImageC[idx] = static_cast<unsigned char>(0.7f * ImageA[idx] + 0.3f * ImageB[idx]);
    }
}

// Simple function to write a grayscale PGM file
void savePGM(const char *filename, unsigned char *image, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    fwrite(image, sizeof(unsigned char), width * height, fp);
    fclose(fp);
}

int main() {
    unsigned char *ImageA, *ImageB, *ImageC;          // Host arrays
    unsigned char *d_ImageA, *d_ImageB, *d_ImageC;    // Device arrays

    // Allocate host memory
    ImageA = (unsigned char *)malloc(SIZE);
    ImageB = (unsigned char *)malloc(SIZE);
    ImageC = (unsigned char *)malloc(SIZE);

    // Initialize ImageA and ImageB with simple patterns (for demo)
    for (int i = 0; i < SIZE; i++) {
        ImageA[i] = (i % WIDTH) * 255 / WIDTH;   // Horizontal gradient
        ImageB[i] = (i / WIDTH) * 255 / HEIGHT;  // Vertical gradient
    }

    // Allocate device memory
    cudaMalloc((void **)&d_ImageA, SIZE);
    cudaMalloc((void **)&d_ImageB, SIZE);
    cudaMalloc((void **)&d_ImageC, SIZE);

    // Copy data from host to device
    cudaMemcpy(d_ImageA, ImageA, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ImageB, ImageB, SIZE, cudaMemcpyHostToDevice);

    // Define number of blocks and threads per block
    int threadsPerBlock = 256;
    int blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    mergeImagesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_ImageA, d_ImageB, d_ImageC);

    // Copy the results back to the host
    cudaMemcpy(ImageC, d_ImageC, SIZE, cudaMemcpyDeviceToHost);

    // Save the merged image to disk
    savePGM("merged_output.pgm", ImageC, WIDTH, HEIGHT);

    // Cleanup
    free(ImageA);
    free(ImageB);
    free(ImageC);
    cudaFree(d_ImageA);
    cudaFree(d_ImageB);
    cudaFree(d_ImageC);

    printf("Merged image saved as merged_output.pgm\n");
    return 0;
}
