#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cublas_v2.h>

#define BLOCK_SIZE 16
#define BASE_TYPE float

int toMultiple(int a, int b) {
    int mod = a % b;
    if (mod != 0) {
        mod = b - mod;
        return a + mod;
    }
    return a;
}

int main()
{
    // Start, stop - for kernel time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Matrix dimensions
    int Arows = 1000;
    int Acols = 2000;
    int Brows = Acols;
    int Bcols = 1500;
    

    Arows = toMultiple(Arows, BLOCK_SIZE);
    printf("Arows = %d\n", Arows);

    Acols = toMultiple(Acols, BLOCK_SIZE);
    printf("Acols = %d\n", Acols);

    Brows = toMultiple(Brows, BLOCK_SIZE);
    printf("Brows = %d\n", Brows);

    Bcols = toMultiple(Bcols, BLOCK_SIZE);
    printf("Bcols = %d\n", Bcols);

    int Crows = Arows;
    int Ccols = Bcols;

    // Allocate host memory
    BASE_TYPE* h_A = (BASE_TYPE*)malloc(Arows * Acols * sizeof(BASE_TYPE));
    BASE_TYPE* h_B = (BASE_TYPE*)malloc(Brows * Bcols * sizeof(BASE_TYPE));
    BASE_TYPE* h_C = (BASE_TYPE*)malloc(Crows * Ccols * sizeof(BASE_TYPE));

    for (int i = 0; i < Arows * Acols; ++i) {
        h_A[i] = rand() / (BASE_TYPE)RAND_MAX;
    }
    for (int i = 0; i < Brows * Bcols; ++i) {
        h_B[i] = rand() / (BASE_TYPE)RAND_MAX;
    }

    BASE_TYPE* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, Arows * Acols * sizeof(BASE_TYPE));
    cudaMalloc((void**)&d_B, Brows * Bcols * sizeof(BASE_TYPE));
    cudaMalloc((void**)&d_C, Crows * Ccols * sizeof(BASE_TYPE));

    // Copy host data to device
    cudaMemcpy(d_A, h_A, Arows * Acols * sizeof(BASE_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, Brows * Bcols * sizeof(BASE_TYPE), cudaMemcpyHostToDevice);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    const BASE_TYPE alpha = 1.0;
    const BASE_TYPE beta = 0.0;

    cudaEventRecord(start, 0);
    cublasStatus_t stat = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        Ccols, Crows, Acols,
        &alpha,
        d_B, Bcols,
        d_A, Acols,
        &beta,
        d_C, Ccols
    );
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS matrix multiplication failed!\n");
        return -1;
    }

    cudaMemcpy(h_C, d_C, Crows * Ccols * sizeof(BASE_TYPE), cudaMemcpyDeviceToHost);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Matrix multiplication completed in %.2f ms.\n", elapsedTime);

    printf("Test STARTED\n");
    for (int i = 0; i < Arows; i++) {
        for (int j = 0; j < Bcols; j++) {
            BASE_TYPE sum = 0;
            for (int k = 0; k < Acols; k++)
                sum += h_A[i * Acols + k] * h_B[k *
                Bcols + j];

            if (fabs(h_C[i * Bcols + j] - sum) > 1e-2) //!точность у умножения ниже, чем при использовании прямых вычислений
            {
                fprintf(stderr, "Result verification \
					failed at element[% d, % d]!\n", i, j);
                printf("sum = %f, h_C[i * Bcols + j] = \
						% f\n", sum, h_C[i * Bcols + j]);
                exit(EXIT_FAILURE);
            }
        }
    }



    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
