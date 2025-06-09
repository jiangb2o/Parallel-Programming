#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <random>
#include <stdio.h>

std::string mode_name[3] = {
    "Global Memory",
    "Shared Memory",
    "Loop Unrolling"
};

/**
 * 结果验证核函数
 * 线程计算自己负责位置的diff值
 */
__global__ void verifyResultKernel(float* diff, const float* cpu_C, const float* cuda_C, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    diff[row * n + col] = fabs(cpu_C[row * n + col] - cuda_C[row * n + col]);
}

/**
 * verify result
 * result: m x n
 */
void verifyMatrixMul(const dim3 blockDim, const dim3 gridDim, const float* cpu_C, const float* d_C, const float* cuda_C, int m, int n) {
    // 根据矩阵规模动态调整误差阈值
    float epsilon = m * n * 1e-8;
    int error_count = 0;

    // cpu计算结果复制到cuda中
    float* d_cpu_C;
    cudaMalloc(&d_cpu_C, m * n * sizeof(float));
    cudaMemcpy(d_cpu_C, cpu_C, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // diff矩阵
    float* h_diff = new float[m * n];
    float* d_diff;
    cudaMalloc(&d_diff, m * n * sizeof(float));

    verifyResultKernel<<<gridDim, blockDim>>>(d_diff, d_cpu_C, d_C, m, n);
    cudaMemcpy(h_diff, d_diff, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < m * n; ++i) {
        if(h_diff[i] > epsilon) {
            error_count++;
            printf("Error at (%3d,%3d): cpu = %.4f, cuda = %.4f\n", i / m, i % m, cpu_C[i], cuda_C[i]);
        }
    }

    if(error_count == 0) {
        std::cout << "verify successfully!" << std::endl;
    } else {
        std::cout << "verify failed! error count: " << error_count << std::endl;
    }

    cudaFree(d_cpu_C);
    cudaFree(d_diff);
    delete[] h_diff;
}

/**
 * CPU matrix mul
 * A: hA x wA
 * B: wA x wB
 * C: hA x wB
 */
void matrixMul(const float* A,const float* B,float* C, int wA, int hA, int wB) {
    for(int i = 0; i < hA; ++i) {
        for(int j = 0; j < wB; ++j) {
            float sum = 0;
            for(int k = 0; k < wA; ++k) {
                sum += A[i * wA + k] * B[k * wB + j];
            }
            C[i * wB + j] = sum;
        }
    }
}

/**
 * random initilize matrix
 * size: elements count(height x width)
 */
void randomInitMatrix(float* mat, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for(int i = 0; i < size; ++i) mat[i] = dis(gen);
}

/**
 * 基础矩阵乘法核函数
 * 线程计算自己位置的结果
 */
__global__ void matrixMulKernel(const float* A, const float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < k) {
        float sum = 0;
        // A row all elements mul B col all elements
        for(int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

/**
 * 共享内存矩阵乘法核函数
 */
__global__ void matrixMulSharedKernel(const float* A, const float* B, float* C, int m, int n, int k) {
    __shared__ float sharedTileA[32][32]; // 共享A矩阵分块
    __shared__ float sharedTileB[32][32]; // 共享B矩阵分块

    // 该线程负责及计算C矩阵row行col列的结果
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;

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

        // 计算子块乘法结果
        for(int i = 0; i < blockDim.x; ++i) {
            sum += sharedTileA[ty][i] * sharedTileB[i][tx];
        }

        __syncthreads();
    }

    if(row < m && col < k) {
        C[row * k + col] = sum;
    }
}

/**
 * 循环展开优化矩阵乘法核函数
 */
__global__ void matrixMulUnrollingKernel(const float* A, const float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < k) {
        // CUDA 内置向量类型
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        // n & (~3) 向下对齐4的倍数
        int floor4 = (n & (~3));
        for(int i = 0; i < floor4; i += 4) {
            float4 a = reinterpret_cast<const float4*>(&A[row * n + i])[0];
            float4 b = make_float4(B[i * k + col], B[(i + 1) * k + col],
                                   B[(i + 2) * k + col], B[(i + 3) * k + col]);

            sum.x += a.x * b.x;
            sum.y += a.y * b.y;
            sum.z += a.z * b.z;
            sum.w += a.w * b.w;
        }
        C[row * k + col] = sum.x + sum.y + sum.z + sum.w;

        // 不足4列的部分
        for(int i = floor4; i < n; ++i) {
            C[row * k + col] += A[row * n + i] * B[i * k + col];
        }
    }
}

/**
 * 运行不同模式的矩阵乘法核函数
 */
void runCudaMatrixMul(int mode, const dim3 blockDim, const dim3 gridDim, int m, int n, int k,
    const float* d_A, const float* d_B, float* d_C, float* cuda_C, const float* cpu_C) {
    cudaEvent_t start, end;

    std::cout << "Mode: " << mode_name[mode] << std::endl;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    // 矩阵乘法核函数
    switch (mode) {
        case 0: matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k); break;
        case 1: matrixMulSharedKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k); break;
        case 2: matrixMulUnrollingKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k); break;
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float time = 0;
    cudaEventElapsedTime(&time, start, end);

    // 将答案拷贝到主机
    cudaMemcpy(cuda_C, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost);
    verifyMatrixMul(gridDim, blockDim, cpu_C, d_C, cuda_C, m, k);
    std::cout << "CUDA Running time: " << time << " ms" << std::endl << std::endl;
}

int main(int argc, char* argv[]) {
    int n;
    int block_size = 32;
    
    if(argc > 1) {
        n = atoi(argv[1]);
    } else {
        std::cout << "Please input matrix size n(128-2048): " << std::endl;
        std::cin >> n;
    }
    if(argc > 2) {
        block_size = atoi(argv[2]);
    }
    
    freopen("result.log", "a+", stdout);
    std::cout << "========== matrix size:" << std::setw(4) << n << " x " << std::setw(4) << n;
    std::cout << ", block size: " << block_size << " x " << block_size << " ==========" << std::endl;

    // initialize host A, B
    float* h_A = new float[n * n];
    float* h_B = new float[n * n];
    randomInitMatrix(h_A, n * n);
    randomInitMatrix(h_B, n * n);
    
    // cup result and cuda result
    float* cpu_C = new float[n * n];
    float* cuda_C = new float[n * n];

    // cuda malloc
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_B, n * n * sizeof(float));
    cudaMalloc(&d_C, n * n * sizeof(float));
    cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // CPU 矩阵乘法
    matrixMul(h_A, h_B, cpu_C, n, n, n);
    
    // 设置线程块大小与grid大小, 每个线程计算一个位置的矩阵乘法结果
    dim3 blockDim(block_size, block_size);
    dim3 gridDim(n / block_size, n / block_size);
    
    runCudaMatrixMul(0, blockDim, gridDim, n, n, n, d_A, d_B, d_C, cuda_C, cpu_C);
    runCudaMatrixMul(1, blockDim, gridDim, n, n, n, d_A, d_B, d_C, cuda_C, cpu_C);
    runCudaMatrixMul(2, blockDim, gridDim, n, n, n, d_A, d_B, d_C, cuda_C, cpu_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] cpu_C;
    delete[] cuda_C;
    
    return 0;
}
