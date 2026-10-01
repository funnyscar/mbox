// CLAUDE A*B
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    const int M = 2, N = 2, K = 2;
    float hA[] = {1, 3, 2, 4}, hB[] = {5, 7, 6, 8}, hC[4];  // column-major

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof hA); cudaMalloc(&dB, sizeof hB); cudaMalloc(&dC, sizeof hC);
    cudaMemcpy(dA, hA, sizeof hA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof hB, cudaMemcpyHostToDevice);

    cublasLtHandle_t lt;
    cublasLtCreate(&lt);

    // Operation descriptor: compute type + scale type
    cublasLtMatmulDesc_t op;
    cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    // Matrix layouts: type, rows, cols, leading dim
    cublasLtMatrixLayout_t Ad, Bd, Cd;
    cublasLtMatrixLayoutCreate(&Ad, CUDA_R_32F, M, K, M);
    cublasLtMatrixLayoutCreate(&Bd, CUDA_R_32F, K, N, K);
    cublasLtMatrixLayoutCreate(&Cd, CUDA_R_32F, M, N, M);

    float alpha = 1.0f, beta = 0.0f;
    // D = alpha*A*B + beta*C  (here C and D are the same buffer)
    cublasLtMatmul(lt, op, &alpha, dA, Ad, dB, Bd, &beta,
                   dC, Cd, dC, Cd,
                   NULL,      // algo: NULL = let cuBLASLt pick
                   NULL, 0,   // workspace, size
                   0);        // stream

    cudaMemcpy(hC, dC, sizeof hC, cudaMemcpyDeviceToHost);
    printf("%g %g\n%g %g\n", hC[0], hC[2], hC[1], hC[3]);  // 19 22 / 43 50

    cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd);
    cublasLtMatrixLayoutDestroy(Cd); cublasLtMatmulDescDestroy(op);
    cublasLtDestroy(lt);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}