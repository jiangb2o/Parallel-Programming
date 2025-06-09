#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

__global__ void transposeKernel(float* mat, float* ans, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < n && col < n) {
        ans[row * n + col] = mat[col * n + row];
    }
}

__global__ void transpose_shared(float* mat, float* ans, int n) {
    __shared__ float tile[32][32];
    
    int row_t = blockIdx.x * blockDim.x + threadIdx.x;
    int col_t = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row_t < n && col_t < n) {
        tile[threadIdx.y][threadIdx.x] = mat[col_t * n + row_t];
    }
    // 等待所有的线程第一步执行完毕
    __syncthreads();
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        ans[row * n + col] = tile[threadIdx.y][threadIdx.x];
    }
}

bool checkTranspose(float* mat, float* ans, int n) {
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            if(abs(ans[i * n + j] - mat[j * n + i]) > 1e-6) {
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    int n;
    int block_size = 8;
    int mode = 0;
    
    if(argc > 1) {
        n = atoi(argv[1]);
    } else {
        std::cout << "Please input matrix size n(512-2048): " << std::endl;
        std::cin >> n;
    }
    if(argc > 2) {
        block_size = atoi(argv[2]);
    }
    if(argc > 3) {
        mode = atoi(argv[3]);
    }
    
    freopen("result.txt", "a+", stdout);
    std::cout << "==========matrix size:" << std::setw(4) << n << " x " << std::setw(4) << n;
    std::cout << ", block size: " << block_size << " x " << block_size;
    std::cout << ", mode: " << (mode == 1 ? "shared" : "normal") << "==========" << std::endl;

    // initialize matrix
    float* mat = new float[n * n];
    float* ans = new float[n * n];
    for(int i = 0; i < n * n; ++i) mat[i] = rand() % 100;

    // cuda malloc
    float* cuda_mat;
    float* cuda_ans;
    cudaMalloc(&cuda_mat, n * n * sizeof(float));
    cudaMalloc(&cuda_ans, n * n * sizeof(float));
    cudaMemcpy(cuda_mat, mat, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块大小与grid大小, 每个线程执行一个转置操作
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    // 转置核函数
    if(mode == 0) {
        transposeKernel<<<gridDim, blockDim>>>(cuda_mat, cuda_ans, n);
    } else {
        transpose_shared<<<gridDim, blockDim>>>(cuda_mat, cuda_ans, n);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float time = 0;
    cudaEventElapsedTime(&time, start, end);

    // 将答案拷贝到主机
    cudaMemcpy(ans, cuda_ans, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << (checkTranspose(mat, ans, n) ? "Correct" : "Wrong") << std::endl;
    std::cout << "Running time: " << time << " ms" << std::endl << std::endl;


    cudaFree(cuda_mat);
    cudaFree(cuda_ans);
    delete[] mat;
    delete[] ans;
    
    return 0;
}
