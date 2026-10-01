// CNN kernel
// what underlies pytorch, tensorflow, etc
#include <cudnn.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    // Input: N=1, C=1, H=4, W=4 ; Filter: K=1, C=1, 3x3 ; pad 0, stride 1
    int N=1, C=1, H=4, W=4, K=1, R=3, S=3;
    float hX[16], hW[9], hY[4];
    for (int i = 0; i < 16; i++) hX[i] = i;
    for (int i = 0; i < 9;  i++) hW[i] = 1;      // box filter

    cudnnHandle_t h;
    cudnnCreate(&h);

    // Descriptors
    cudnnTensorDescriptor_t xD, yD;
    cudnnFilterDescriptor_t wD;
    cudnnConvolutionDescriptor_t cD;
    cudnnCreateTensorDescriptor(&xD);
    cudnnCreateTensorDescriptor(&yD);
    cudnnCreateFilterDescriptor(&wD);
    cudnnCreateConvolutionDescriptor(&cD);

    cudnnSetTensor4dDescriptor(xD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
    cudnnSetFilter4dDescriptor(wD, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, R, S);
    cudnnSetConvolution2dDescriptor(cD, 0, 0, 1, 1, 1, 1,      // pad, stride, dilation
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    // Output shape
    int on, oc, oh, ow;
    cudnnGetConvolution2dForwardOutputDim(cD, xD, wD, &on, &oc, &oh, &ow);  // 1,1,2,2
    cudnnSetTensor4dDescriptor(yD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);

    // Device memory
    float *dX, *dW, *dY;
    cudaMalloc(&dX, sizeof hX); cudaMalloc(&dW, sizeof hW); cudaMalloc(&dY, sizeof hY);
    cudaMemcpy(dX, hX, sizeof hX, cudaMemcpyHostToDevice);
    cudaMemcpy(dW, hW, sizeof hW, cudaMemcpyHostToDevice);

    // Algorithm + workspace
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    size_t wsSize;
    cudnnGetConvolutionForwardWorkspaceSize(h, xD, wD, cD, yD, algo, &wsSize);
    void *ws = NULL;
    if (wsSize) cudaMalloc(&ws, wsSize);

    float alpha = 1.f, beta = 0.f;
    cudnnConvolutionForward(h, &alpha, xD, dX, wD, dW, cD, algo,
                            ws, wsSize, &beta, yD, dY);

    cudaMemcpy(hY, dY, sizeof hY, cudaMemcpyDeviceToHost);
    printf("%g %g\n%g %g\n", hY[0], hY[1], hY[2], hY[3]);  // 45 54 / 81 90

    cudaFree(ws); cudaFree(dX); cudaFree(dW); cudaFree(dY);
    cudnnDestroyConvolutionDescriptor(cD);
    cudnnDestroyFilterDescriptor(wD);
    cudnnDestroyTensorDescriptor(xD); cudnnDestroyTensorDescriptor(yD);
    cudnnDestroy(h);
}