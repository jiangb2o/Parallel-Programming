#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel function
__global__ void hello_world_kernel() {
    printf("Hello World from thread (%2d,%2d) in Block %2d!\n", threadIdx.x, threadIdx.y, blockIdx.x);
}

int main() {
    int n, m, k;

    std::cout << "Please input three integers n,m,k(Range from 1 to 32): " << std::endl;
    std::cin >> n >> m >> k;

    int blockNum = n;
    dim3 blockDim(m, k);

    // Launch the kernel
    hello_world_kernel<<<blockNum, blockDim>>>();

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    std::cout << "Hello World from the host!" << std::endl;

    return 0;
}
