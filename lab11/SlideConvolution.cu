#include <cuda_runtime.h>

/**
 * 每个线程计算一个output位置的值
 */
__global__ void slideConvolution(
    const float* d_in, const float* d_k, float* d_out, 
    int in_size, int k_size, int out_size, 
    int channel, int stride, int padding
) {
    // 本线程负责计算output的位置(row, col)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= out_size || col >= out_size) {
        return;
    }

    float sum = 0.0f;

    // 每个channel计算的值累加
    for(int c = 0; c < channel; ++c) {
        // 计算 c 通道下 kernel 每个位置与对应input位置的值的结果
        for(int k_row = 0; k_row < k_size; ++k_row) {
            for(int k_col = 0; k_col < k_size; ++k_col) {
                // kernel 在(k_row, k_col) 对应 input 的位置(in_row, in_col)计算
                int in_row = row * stride - padding + k_row;
                int in_col = col * stride - padding + k_col;

                if(in_row < 0 || in_row >= in_size || in_col < 0 || in_col >= in_size) {
                    continue;
                }
                // 由于使用一维数组, 将 (c, in_row, in_col) 转换为一维下标
                int in_idx = (c * in_size + in_row) * in_size + in_col;
                int k_idx = (c * k_size + k_row) * k_size + k_col;
                sum += d_in[in_idx] * d_k[k_idx];
            }
        }
    }

    // 输出(row,col)转换为一维下标
    int out_idx = row * out_size + col;
    d_out[out_idx] = sum;
}

void slide(float& time, const float* d_in, const float* d_k, float* d_out, 
    int in_size, int k_size, int out_size, 
    int channel, int stride, int padding, dim3 blockDim) {

    dim3 gridDim((out_size + blockDim.x - 1) / blockDim.x,
                 (out_size + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    
    slideConvolution<<<gridDim, blockDim>>>(
                d_in, d_k, d_out, 
                in_size, k_size, out_size, 
                channel, stride, padding
            );
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    cudaEventElapsedTime(&time, start, end);
}