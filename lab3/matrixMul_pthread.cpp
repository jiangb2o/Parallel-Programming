// g++ -pthread filename.cpp -o filename
#include<pthread.h>
#include<iostream>
#include<stdio.h>
#include<string>
#include<random>
#include<chrono>

using namespace std;
const int show_size = 8;

double *mat1, *mat2, *ans;

void usage(const char* program_name) {
    printf("useage:./%s <num of threads> <matrix size>\n", program_name);
}

void getRandomMat(double* mat, int m, int n) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(1, 10);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mat[i*n+j] = distrib(gen);
        }
    }
}

void showMatCorner(FILE* fp, double* mat, int m, int n) {
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            fprintf(fp, "%12.3f ", mat[i*m + j]);
        }
        fprintf(fp, "\n");
    }
}

struct matMulPara{
    int rows;
    int n;
    int thread_no;
    matMulPara(){rows = 0; n = 0; thread_no = 0;}
    matMulPara(int _rows, int _n, int _thread_no){
        rows = _rows;
        n = _n;
        thread_no = _thread_no;
    }
};

void* matMul(void* args) {
    matMulPara* para = static_cast<matMulPara*>(args);
    int rows = para->rows;
    int n = para->n;
    int begin = para->thread_no * rows;
    int end = (para->thread_no + 1) * rows;
    for(int i = begin; i < end; ++i) {
        for(int j = 0; j < n; ++j) {
            for(int k = 0; k < n; ++k)
                ans[i*n + j] += mat1[i*n + k] * mat2[k*n + j];
        }
    }
}

int main(int argc, char* argv[]) {
    int thread_num = 1;
    int matrix_size = 128;
    if(argc >= 2) {
        try {
            thread_num = std::stoi(argv[1]);
        } catch (std::invalid_argument const& ex) {
            usage(argv[0]);
        }
    }
    if(argc >= 3) {
        try {
            matrix_size = std::stoi(argv[2]);
        } catch (std::invalid_argument const& ex) {
            usage(argv[0]);
        }
    }
    FILE *fp = fopen("matrixMul_result.txt", "a+");
    fprintf(fp, "\n\n=====number of threads: %d, size of matrix: %d x %d=====\n", thread_num, matrix_size, matrix_size);

    mat1 = new double[matrix_size * matrix_size];
    mat2 = new double[matrix_size * matrix_size];
    ans = new double[matrix_size * matrix_size]();
    getRandomMat(mat1, matrix_size, matrix_size);
    getRandomMat(mat2, matrix_size, matrix_size);
    fprintf(fp, "mat1:\n");
    showMatCorner(fp, mat1, matrix_size, show_size);
    fprintf(fp, "mat2:\n");
    showMatCorner(fp, mat2, matrix_size, show_size);

    int rows_each_thread = matrix_size / thread_num;
    pthread_t* threads = new pthread_t[thread_num];

    // 创建参数
    matMulPara* threads_para = new matMulPara[thread_num];
    for(int i = 0; i < thread_num; ++i) {
        threads_para[i] = matMulPara(rows_each_thread, matrix_size, i);
    }

    // 创建线程
    auto begin = chrono::high_resolution_clock::now();
    for(int i = 0; i < thread_num; ++i) {
        pthread_create(&threads[i], NULL, matMul, static_cast<void*>(&threads_para[i]));
    }
    // 回收线程
    for(int i = 0; i < thread_num; ++i) {
        pthread_join(threads[i], NULL);
    }
    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);

    fprintf(fp, "ans:\n");
    showMatCorner(fp, ans, matrix_size, show_size);
    fprintf(fp,"running time: %.4f s\n", elapsed * 1e-9);
    fclose(fp);
    delete [] mat1;
    delete [] mat2;
    delete [] ans;
    delete [] threads;
    delete [] threads_para;

    return 0;
}

