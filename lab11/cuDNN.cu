#include <cuda_runtime.h>
#include <cudnn.h> 

#include <iostream>

/**
 * 检查CUDNN运行错误
 */
#define checkCUDNN(expression)                               \
{                                                            \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
        std::cerr << "Error on line " << __LINE__ << ": "    \
                << cudnnGetErrorString(status) << std::endl; \
        std::exit(EXIT_FAILURE);                             \
    }                                                        \
}


void cuDNN(
    float& time,
    const float* d_in, const float* d_k, float* d_out, 
    int in_size, int k_size, int out_size, 
    int channel, int stride, int padding
) {
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    // 创建描述符
    cudnnTensorDescriptor_t in_desc, out_desc;
    cudnnFilterDescriptor_t kernel_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    checkCUDNN(cudnnCreateTensorDescriptor(&in_desc));
    checkCUDNN(cudnnCreateTensorDescriptor(&out_desc));
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_desc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    // 设置描述符
    checkCUDNN(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, channel, in_size, in_size));
    checkCUDNN(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, channel, out_size, out_size));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, channel, k_size, k_size));
    checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    
    cudnnConvolutionForward(
        cudnn,
        &alpha,
        in_desc, d_in, 
        kernel_desc, d_k,
        conv_desc,
        algo,
        nullptr, 0,
        &beta,
        out_desc, d_out
    );
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);

    // 清理资源
    checkCUDNN(cudnnDestroyTensorDescriptor(in_desc));
    checkCUDNN(cudnnDestroyTensorDescriptor(out_desc));
    checkCUDNN(cudnnDestroyFilterDescriptor(kernel_desc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    checkCUDNN(cudnnDestroy(cudnn));
}