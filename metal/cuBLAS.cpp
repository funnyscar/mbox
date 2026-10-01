// CLAUDE A*B
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    const int M = 2, N = 2, K = 2;
    // cuBLAS is COLUMN-MAJOR
    float hA[] = {1, 3, 2, 4};   // A = [1 2; 3 4]
    float hB[] = {5, 7, 6, 8};   // B = [5 6; 7 8]
    float hC[4];

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof hA);
    cudaMalloc(&dB, sizeof hB);
    cudaMalloc(&dC, sizeof hC);
    cudaMemcpy(dA, hA, sizeof hA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof hB, cudaMemcpyHostToDevice);

    cublasHandle_t h;
    cublasCreate(&h);

    float alpha = 1.0f, beta = 0.0f;
    // C = alpha*A*B + beta*C ; lda=M, ldb=K, ldc=M
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                &alpha, dA, M, dB, K, &beta, dC, M);

    cudaMemcpy(hC, dC, sizeof hC, cudaMemcpyDeviceToHost);
    printf("%g %g\n%g %g\n", hC[0], hC[2], hC[1], hC[3]); // 19 22 / 43 50

    cublasDestroy(h);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}