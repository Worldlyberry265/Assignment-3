#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int n) {
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;

    float Cvalue = 0.0;

    for (int i = 0; i < n/TILE_DIM; ++i) {
        sA[ty][tx] = A[Row*n + i*TILE_DIM + tx];
        sB[ty][tx] = B[(i*TILE_DIM + ty)*n + Col];

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            Cvalue += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    C[Row*n+Col] = Cvalue;
}

int main() {
    const int N = 512;
    size_t size = N * N * sizeof(float);
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    // Allocate memory on host
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_A[i*N+j] = (float)i;
            h_B[i*N+j] = (float)j;
            h_C[i*N+j] = 0.0;
        }
    }

    // Allocate memory on device
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid(N/TILE_DIM, N/TILE_DIM);

    // Start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    matrixMultiplyTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy output matrix from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print output matrix
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%f ", h_C[i*N+j]);
        }
        printf("\n");
    }

    // Print execution time
    printf("Execution time: %f ms\n", milliseconds);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}