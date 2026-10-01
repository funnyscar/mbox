// CLAUDE WROTE THIS CODE
#include <cstdio>
#include <cuda_runtime.h>

// Runs on the GPU: each thread adds one element
__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 20;               // ~1M elements
    size_t bytes = n * sizeof(float);

    // Host (CPU) memory
    float *h_a = new float[n], *h_b = new float[n], *h_c = new float[n];
    for (int i = 0; i < n; i++) { h_a[i] = i; h_b[i] = 2.0f * i; }

    // Device (GPU) memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // CPU -> GPU
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Launch: enough 256-thread blocks to cover n
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add<<<blocks, threads>>>(d_a, d_b, d_c, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("Launch error: %s\n", cudaGetErrorString(err)); return 1; }

    // GPU -> CPU (waits for kernel to finish)
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Check
    for (int i = 0; i < n; i++)
        if (h_c[i] != 3.0f * i) { printf("Mismatch at %d\n", i); return 1; }
    printf("OK: c[100] = %.0f\n", h_c[100]);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;
}