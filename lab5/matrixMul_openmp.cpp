#include<iostream>
#include<random>
#include<chrono>
#include<vector>
#include<omp.h>


using namespace std;

const int show_size = 8;

void usage(const char* program_name) {
    printf("useage:./%s <num of threads> <matrix size>\n", program_name);
}

void getRandomMat(vector<double>& mat, int m, int n) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(1, 10);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mat[i*n+j] = distrib(gen);
        }
    }
}

void showMatCorner(FILE* fp, const vector<double>& mat, int m, int n) {
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            fprintf(fp, "%12.3f ", mat[i*m + j]);
        }
        fprintf(fp, "\n");
    }
}

void matMulDefault(vector<double>& ans, const vector<double>& mat1, const vector<double>& mat2, int n) {
    #pragma omp parallel for collapse(2) shared(ans, mat1, mat2)
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            for(int k = 0; k < n; ++k) {
                ans[i*n + j] += mat1[i*n + k] * mat2[k*n + j];
            }
        }
    }
}

void matMulStatic(vector<double>& ans, const vector<double>& mat1, const vector<double>& mat2, int n) {
    #pragma omp parallel for collapse(2) schedule(static) shared(ans, mat1, mat2)
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            for(int k = 0; k < n; ++k) {
                ans[i*n + j] += mat1[i*n + k] * mat2[k*n + j];
            }
        }
    }
}

void matMulDynamic(vector<double>& ans, const vector<double>& mat1, const vector<double>& mat2, int n) {
    #pragma omp parallel for collapse(2) schedule(dynamic) shared(ans, mat1, mat2)
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            for(int k = 0; k < n; ++k) {
                ans[i*n + j] += mat1[i*n + k] * mat2[k*n + j];
            }
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
    FILE *fp = fopen("matrixMul_resultv2.txt", "a+");
    fprintf(fp, "\n\n=====number of threads: %d, size of matrix: %d x %d=====\n", thread_num, matrix_size, matrix_size);

    // initialize matrices
    vector<double> mat1(matrix_size * matrix_size);
    vector<double> mat2(matrix_size * matrix_size);
    vector<double> ans(matrix_size * matrix_size, 0);
    getRandomMat(mat1, matrix_size, matrix_size);
    getRandomMat(mat2, matrix_size, matrix_size);
    fprintf(fp, "mat1:\n");
    showMatCorner(fp, mat1, matrix_size, show_size);
    fprintf(fp, "mat2:\n");
    showMatCorner(fp, mat2, matrix_size, show_size);

    // set number of threads
    omp_set_num_threads(thread_num);

    /* ======default schedule======= */
    fill(ans.begin(), ans.end(), 0);
    auto begin = chrono::high_resolution_clock::now();
    matMulDefault(ans, mat1, mat2, matrix_size);
    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);
    fprintf(fp, "==========default schedule ans:==========\n");
    showMatCorner(fp, ans, matrix_size, show_size);
    fprintf(fp,"running time: %.4f s\n\n", elapsed * 1e-9);

    /* ======static schedule======= */
    fill(ans.begin(), ans.end(), 0);
    begin = chrono::high_resolution_clock::now();
    matMulStatic(ans, mat1, mat2, matrix_size);
    end = chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);
    fprintf(fp, "==========static schedule ans:==========\n");
    showMatCorner(fp, ans, matrix_size, show_size);
    fprintf(fp,"running time: %.4f s\n\n", elapsed * 1e-9);


    /* ======dynamic schedule======= */
    fill(ans.begin(), ans.end(), 0);
    begin = chrono::high_resolution_clock::now();
    // matrix multiplication
    matMulDynamic(ans, mat1, mat2, matrix_size);
    end = chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);
    fprintf(fp, "==========dynamic schedule ans:==========\n");
    showMatCorner(fp, ans, matrix_size, show_size);
    fprintf(fp,"running time: %.4f s\n\n", elapsed * 1e-9);


    fclose(fp);
    return 0;
}