#include<iostream>
#include<random>
#include<chrono>
#include"parallel_for.h"

using namespace std;

const int show_size = 8;

void getRandomMat(float* mat, int m, int n) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(1, 10);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mat[i*n+j] = distrib(gen);
        }
    }
}

void showMatCorner(FILE* fp, float* mat, int m, int n) {
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            fprintf(fp, "%12.3f ", mat[i*m + j]);
        }
        fprintf(fp, "\n");
    }
}

struct MatMulArgs {
    float* A;
    float* B;
    float* C;
    int n;
    int k;
};

void* matMul(int i, void* arg) {
    MatMulArgs* args = static_cast<MatMulArgs*>(arg);
    int n = args->n;
    int k = args->k;

    for(int j = 0; j < n; ++j) {
        for(int l = 0; l < k; ++l) {
            args->C[i*n + j] += args->A[i*k + l] * args->B[l*n + j];
        }
    }
    return nullptr;
}

int main(void) {
    int m = 1024, n = 1024, k = 1024;
    int thread_num = 4;
    FILE *fp = fopen("matrixMul_result.txt", "a+");
    float* A = new float[1024 * 1024];
    float* B = new float[1024 * 1024];
    float* C = new float[1024 * 1024]();
    getRandomMat(A, m, n);
    getRandomMat(B, n, k);
    fprintf(fp, "A:\n");
    showMatCorner(fp, A, m, show_size);
    fprintf(fp, "B:\n");
    showMatCorner(fp, B, n, show_size);

    MatMulArgs* arg = new MatMulArgs{A, B, C, n, k};
    auto begin = chrono::high_resolution_clock::now();
    parallel_for(0, m, 1, matMul, arg, thread_num);
    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);
    fprintf(fp, "C:\n");
    showMatCorner(fp, C, m, show_size);
    fprintf(fp,"running time: %.4f s\n", elapsed * 1e-9);
    fclose(fp);
}
