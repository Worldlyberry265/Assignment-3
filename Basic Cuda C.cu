#include <stdio.h>
#include <stdlib.h>
//#include <cuda_runtime.h>
#include <time.h>

#define BLOCK_SIZE 16

__global__ void matrix_multiplication(float* M, float* N, float* P, int Width) {
// Calculate the row index of the P element and M
int Row = blockIdx.y*blockDim.y+threadIdx.y;
// Calculate the column index of P and N
int Col = blockIdx.x*blockDim.x+threadIdx.x;
if ((Row < Width) && (Col < Width)) {
float Pvalue = 0;
// each thread computes one element of the block sub-matrix
for (int k = 0; k < Width; ++k) {
Pvalue += M[Row*Width+k]*N[k*Width+Col];
}
P[Row*Width+Col] = Pvalue;
}
}
int main() {
    int N = 512;
    float *h_matrix1, *h_matrix2, *h_result;
    float *d_matrix1, *d_matrix2, *d_result;
    size_t bytes = N * N * sizeof(float);

    // Allocate memory on host
    h_matrix1 = (float*) malloc(bytes);
    h_matrix2 = (float*) malloc(bytes);
    h_result = (float*) malloc(bytes);

    // Initialize matrices on host
    for (int i = 0; i < N * N; i++) {
        h_matrix1[i] = rand() % 10;
        h_matrix2[i] = rand() % 10;
        h_result[i] = 0;
    }

    // Allocate memory on device
    cudaMalloc(&d_matrix1, bytes);
    cudaMalloc(&d_matrix2, bytes);
    cudaMalloc(&d_result, bytes);

    // Copy matrices from host to device
    cudaMemcpy(d_matrix1, h_matrix1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, h_matrix2, bytes, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Start timer
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);

    // Launch kernel





 // Start timer
    clock_t start = clock();
    
    matrix_multiplication<<<grid, block>>>(d_matrix1, d_matrix2, d_result, N);

        // Stop timer
    clock_t stop = clock();
    double milliseconds = (double) (stop - start) * 1000 / CLOCKS_PER_SEC;

    // Stop timer
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result from device to host
    cudaMemcpy(h_result, d_result, bytes, cudaMemcpyDeviceToHost);

    

    // Print resulting matrix
    printf("Resulting matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", h_result[i * N + j]);
        }
        printf("\n");
    }
       // Print execution time
    printf("Execution time: %f ms\n", milliseconds);

    // Free memory on host and device
    free(h_matrix1);
    free(h_matrix2);
    free(h_result);
    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_result);

    return 0;
}