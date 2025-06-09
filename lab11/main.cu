#include<iostream>
#include<stdio.h>
#include<random>
#include<string>
#include<sstream>

#include <cuda_runtime.h>

// const parameters
const int channel = 3;
const int k_size = 3;
const int strides[3] = {1, 2, 3};
const std::string method_name[3] = {"Slide", "im2col", "cuDNN"};
const int show_size = 8;

void usage(const char* program_name) {
    printf("useage:./%s <thread block size> <input matrix size> \n", program_name);
}

void randomInit(float* data, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for(int i = 0; i < size; ++i) data[i] = dis(gen);
}

void oneInit(float* data, int size) {
    for(int i = 0; i < size; ++i) data[i] = 1.0f;
}

void showMatrixCorner(float* mat, int m, int n) {
    printf("[\n");
    for(int i = 0; i < m; ++i) {
        printf("[");
        for(int j = 0; j < n; ++j) {
            printf("%6.3f ", mat[i*m + j]);
            if(j != n - 1) {
                printf(",");
            }
        }
        i != m - 1 ? printf("],\n") : printf("]");
    }
    printf("]\n");
}

/**
 * print 2D matrix
 */
void printMat2D(const float* mat, int row, int col) {
    printf("[");
    for(int i = 0; i < row; ++i) {
        printf("[");
        for(int j = 0; j < col; ++j) {
            printf("%6.3f", mat[i * col + j]);
            if(j != col - 1) {
                printf(",");
            }
        }
        i != row - 1 ? printf("],\n") : printf("]");
    }
    printf("]\n");
}

/**
 * print 3D matrix
 */
void printMat3D(const float* mat, int depth, int row, int col) {
    printf("[\n");
    for(int i = 0; i < depth; ++i) {
        printMat2D(&mat[i * row * col], row, col);
        if(i != depth - 1) {
            printf(",\n");
        }
    }
    printf("]\n");
}

extern __global__ void slideConvolution(const float* d_in, const float* d_k, float* d_out, 
    int in_size, int k_size, int out_size, 
    int channel, int stride, int padding);

extern __global__ void im2colKernel(float* d_col, const float* d_in, int col_x, int col_y, int in_size, int out_size, int k_size, int stride, int padding);

extern __global__ void GEMM(const float* A, const float* B, float* C, int m, int n, int k);

extern void cuDNN(float& time, const float* d_in, const float* d_k, float* d_out, 
    int in_size, int k_size, int out_size, 
    int channel, int stride, int padding);

extern float* im2colGEMM(float& time, int col_x, int col_y, 
    const dim3 blockDim, const float* d_in, const float* d_k, float* d_out,
    int in_size, int out_size, int k_size, int channel, int stride, int padding);

extern void slide(float& time, const float* d_in, const float* d_k, float* d_out, 
    int in_size, int k_size, int out_size, 
    int channel, int stride, int padding, dim3 blockDim);

/**
 * run different convolution methods
 */
void runConv(
    const float* d_in, const float* d_k, float* d_out, 
    int in_size, int k_size, int out_size, 
    int channel, int stride, int padding,
    float* h_out,
    int block_size
) {
    dim3 blockDim(block_size, block_size);
    std::cout << "thread block: " << block_size << " x " << block_size << std::endl;

    float* d_col;
    // im2col 的col矩阵规模
    int col_x = k_size * k_size * channel;
    int col_y = out_size * out_size;

    for(int method = 0; method < 3; ++method) {
        std::cout << "Convolution method: " << method_name[method] << std::endl;

        float time = 0;
        switch (method)
        {
        case 0: slide(
                time,
                d_in, d_k, d_out, 
                in_size, k_size, out_size, 
                channel, stride, padding,
                blockDim
            ); break;
        case 1: d_col = im2colGEMM(
            time, col_x, col_y, 
            blockDim, d_in, d_k, d_out, 
            in_size, out_size, k_size,
            channel, stride, padding); break;
        case 2: cuDNN(
                time,
                d_in, d_k, d_out, 
                in_size, k_size, out_size, 
                channel, stride, padding
            ); break;
        }
        
        // 将答案拷贝到主机
        cudaMemcpy(h_out, d_out, out_size * out_size * sizeof(float), cudaMemcpyDeviceToHost);

        if(method == 1) {
            std::cout << "im2col shape: (" << col_x << ", " << col_y << ")" << std::endl;

            float* h_col = new float [col_x * col_y];
            cudaMemcpy(h_col, d_col, col_x * col_y * sizeof(float), cudaMemcpyDeviceToHost);
            
            // 打印 im2col 结果
            // std::cout << "im2col matrix:" << std::endl;
            // printMat2D(h_col, col_x, col_y);

            delete[] h_col;
            cudaFree(d_col);
        }

        // 打印结果
        // std::cout << "output:" << std::endl;
        // printMat2D(h_out, out_size, out_size);

        // 打印部分结果
        std::cout << "output corner:" << std::endl;
        showMatrixCorner(h_out, show_size, show_size);
        
        std::cout << "Convolution Running time: " << time << " ms" << std::endl << std::endl;
    }
}

int main(int argc, char* argv[]) {
    int block_size = 16;
    int in_size = 32;

    if(argc > 1) {
        try {
            block_size = atoi(argv[1]);
        } catch (std::invalid_argument const& ex) {
            usage(argv[0]);
        }
    }
    if(argc > 2) {
        try {
            in_size = atoi(argv[2]);
        } catch (std::invalid_argument const& ex) {
            usage(argv[0]);
        }
    }

    std::stringstream result_name;
    result_name << "./result/result_input_" << std::to_string(in_size) << ".log";
    freopen(result_name.str().c_str(), "a+", stdout);
    std::cout << "======input size: " << in_size << " x " << in_size << "======" << std::endl;
    
    int in_elements = channel * in_size * in_size;
    int k_elements = channel * k_size * k_size;
    // host分配内存
    float* h_in = new float[in_elements];
    float* h_k = new float[k_elements];

    randomInit(h_in, in_elements);
    // oneInit(h_in, in_elements);

    // 初始化卷积核为1, 便于验证答案
    oneInit(h_k, k_elements);

    // cuda分配内存
    float* d_in;
    float* d_k;
    cudaMalloc(&d_in, in_elements * sizeof(float));
    cudaMalloc(&d_k, k_elements * sizeof(float));
    cudaMemcpy(d_in, h_in, in_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, k_elements * sizeof(float), cudaMemcpyHostToDevice);

    // 打印input
    // std::cout << "input: " << std::endl;
    // printMat3D(h_in, k_size, in_size, in_size);

    // 不同步长
    for(int stride : strides) {
        // 采用SAME Padding策略(输出大小约等于输入大小除以步长), 根据input_size和stride计算padding
        int out_size = ceil(in_size / (float)stride);
        int padding = ((out_size - 1) * stride + k_size - in_size) / 2;
        // 计算出的结果可能无法被2整除, 使用对称padding, 所以还需要根据padding重新计算输出大小
        out_size = (in_size - k_size + 2 * padding) / stride + 1;
        
        std::cout << "stride: " << stride << ", padding: " << padding << ", output size: " << out_size << std::endl;

        int out_elements = out_size * out_size;
        float* h_out = new float[out_elements];

        float* d_out;
        cudaMalloc(&d_out, out_elements * sizeof(float));

        runConv(
            d_in, d_k, d_out, 
            in_size, k_size, out_size, 
            channel, stride, padding,
            h_out, 
            block_size
        );
        
        delete[] h_out;
        cudaFree(d_out);
    }

    // 释放内存
    delete[] h_in;
    delete[] h_k;
    cudaFree(d_in);
    cudaFree(d_k);

    return 0;
}