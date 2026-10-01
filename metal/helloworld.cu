#include <cstdio>

// flagged for gpu (kernel)
__global__ void hello() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

// runs on the CPU
int main() {
	// launch 1 block of 4 threads
    // calls kernel<<<blocks, threads>>>(args) to run a kernel across thousands of threads.
    hello<<<1, 4>>>();        
    cudaDeviceSynchronize();  // wait for the GPU to finish
}