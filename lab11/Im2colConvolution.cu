#include <cuda_runtime.h>

/**
 * im2col 核函数
 * 每个线程计算col一个位置的值
 */
__global__ void im2colKernel(float* d_col,const float* d_in, int col_x, int col_y, int in_size, int out_size, int k_size, int stride, int padding) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;

    if(x >= col_x || y >= col_y) {
        return;
    }

    int k_col = k_size * k_size;

    // col (x,y) 位置对应 im (in_c, in_x, in_y) 位置的值
    int in_c = x / k_col;
    // x % k_col / k_size: 第一个 kernel 的 x 位置; y / out_size * stride: kernel 向下移动偏移, padding: padding偏移
    int in_x = x % k_col / k_size + y / out_size * stride - padding;
    // x % k_col % k_size: 第一个 kernel 的 y 位置; y % out_size * stride: kernel 向右移动偏移: padding: padding偏移
    int in_y = x % k_col % k_size + y % out_size * stride - padding;
    
    // 处于padding部分, 赋值为0
    if(in_x < 0 || in_x >= in_size || in_y < 0 || in_y >= in_size) {
        d_col[x * col_y + y] = 0.0f;
    } else {
        // (in_c, in_x, in_y) 转换为一维下标
        d_col[x * col_y + y] = d_in[(in_c * in_size + in_x) * in_size + in_y];
    }
}

/**
 * 通用矩阵乘法核函数
 * 共享内存优化+循环展开优化
 */
__global__ void GEMM(const float* A, const float* B, float* C, int m, int n, int k) {
    __shared__ float sharedTileA[32][32]; // 共享A矩阵分块
    __shared__ float sharedTileB[32][32]; // 共享B矩阵分块

    // 该线程负责及计算C矩阵row行col列的结果
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // 分块数量
    int tile_count = (n + blockDim.x - 1) / blockDim.x;

    // 循环处理分块
    for(int tile = 0; tile < tile_count; ++tile) {
        // 当前分块线程负责加载A的位置 (row, tile * blockDim.x + tx)
        int tile_idx_A = row * n + tile * blockDim.x + tx;
        // 当前分块线程负责加载B的位置 (tile * blockDim.y + ty, col)
        int tile_idx_B = (tile * blockDim.y + ty) * k + col;

        // 加载A,B矩阵对应分块的值到共享内存中
        if(row < m && tile * blockDim.x + tx < n) {
            sharedTileA[ty][tx] = A[tile_idx_A];
        } else {
            sharedTileA[ty][tx] = 0.0f;
        }

        if(col < k && tile * blockDim.y + ty < n) {
            sharedTileB[ty][tx] = B[tile_idx_B];
        } else {
            sharedTileB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // 循环展开 计算子块乘法结果
        int floor4 = (blockDim.x & (~3));
        for(int i = 0; i < floor4; i += 4) {
            sum.x += sharedTileA[ty][i] * sharedTileB[i][tx];
            sum.y += sharedTileA[ty][i + 1] * sharedTileB[i + 1][tx];
            sum.z += sharedTileA[ty][i + 2] * sharedTileB[i + 2][tx];
            sum.w += sharedTileA[ty][i + 3] * sharedTileB[i + 3][tx];
        }
        // 不足4列的部分
        for(int i = floor4; i < blockDim.x; ++i) {
            sum.x += sharedTileA[ty][i] * sharedTileB[i][tx];
        }

        __syncthreads();
    }

    if(row < m && col < k) {
        C[row * k + col] = sum.x + sum.y + sum.z + sum.w;
    }
}

/**
 * 将输入转换为col
 * 然后执行GEMM
 */
float* im2colGEMM(
    float& time, int col_x, int col_y, 
    const dim3 blockDim, const float* d_in, const float* d_k, float* d_out,
    int in_size, int out_size, int k_size, int channel, int stride, int padding
) {

    float* d_col;
    cudaMalloc(&d_col, col_x * col_y * sizeof(float));

    dim3 gridDim((col_x + blockDim.x - 1) / blockDim.x,
                 (col_y + blockDim.y - 1) / blockDim.y);
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    
    im2colKernel<<<gridDim, blockDim>>>(d_col, d_in, col_x, col_y, in_size, out_size, k_size, stride, padding);
    cudaDeviceSynchronize();

    // 任务变化, GEMM结果维度为(1, out_size * out_size), 重新计算gridDim
    gridDim.x = (1 + blockDim.x - 1) / blockDim.x;
    gridDim.y = (out_size * out_size + blockDim.y - 1) / blockDim.y;

    GEMM<<<gridDim, blockDim>>>(d_k, d_col, d_out, 1, channel * k_size * k_size, col_y);
    cudaDeviceSynchronize();
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    
    return d_col;
}