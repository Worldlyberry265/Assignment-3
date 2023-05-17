#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matrix_multiplication(int M, int N, int P, int matrix1[M][N], int matrix2[N][P], int result[M][P]) {
    // Perform matrix multiplication
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            for (int k = 0; k < N; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

int main() {
    int N = 512;
    int matrix1[N][N], matrix2[N][N], result[N][N];

    // Initialize matrices with random values
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix1[i][j] = rand() % 10;
            matrix2[i][j] = rand() % 10;
            result[i][j] = 0;
        }
    }

    // Start timer
    clock_t start = clock();

    // Perform matrix multiplication
    matrix_multiplication(N, N, N, matrix1, matrix2, result);

    // Stop timer
    clock_t stop = clock();
    double milliseconds = (double) (stop - start) * 1000 / CLOCKS_PER_SEC;

 

    // Print resulting matrix
    printf("Resulting matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", result[i][j]);
        }
        printf("\n");
    }
       // Print execution time
    printf("Execution time: %f ms\n", milliseconds);

    return 0;
}