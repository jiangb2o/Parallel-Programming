#include<iostream>
#include <random>
#include<stdio.h>
#include<cmath>
#include<mpi.h>

using namespace std;
int show_size = 8;

void getRandomMat(double* mat, int m, int n) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(1, 100);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mat[i*n+j] = distrib(gen);
        }
    }
}

void matMul(double* ans, double* mat1, double* mat2, int rows, int n) {
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < n; ++j) {
            for(int k = 0; k < n; ++k)
                ans[i*n + j] += mat1[i*n + k] * mat2[k*n + j];
        }
    }
}

void showMatCorner(double* mat, int m, int n) {
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            printf("%12.3f ", mat[i*m + j]);
        }
        cout << endl;
    }
}

int main(int argc, char* argv[]) {
    int rank = 0, size = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;
    
    // 主进程
    if(rank == 0) {
        int m = 0;
        cout << "input matrix size m: ";
        cin >> m;
        double* mat1 = new double[m*m];
        double* mat2 = new double[m*m];
        double* ans = new double [m*m](); // 初始化为0

        getRandomMat(mat1, m, m);
        getRandomMat(mat2, m, m);

        cout << "mat1:" << endl;
        showMatCorner(mat1, m, show_size);
        cout << "mat2:" << endl;
        showMatCorner(mat2, m, show_size);
        cout << "ans:" << endl;
        showMatCorner(ans, m, show_size);

        double begin = MPI_Wtime();
        int begin_row, end_row, row_per_p = m;
        if(size > 1) {
            row_per_p = ceil(m*1.0 / size);
            for(int i = 1; i < size; ++i) {
                begin_row = row_per_p * i;
                end_row = min(row_per_p * (i + 1) - 1, m - 1);
                // msg[0] 处理行数, msg[1] 一行元素个数
                int msg[2] = {end_row - begin_row + 1, m};
                // 需要处理几行
                printf("process: %d, calculate %d rows(%d to %d)\n", i, msg[0], begin_row, end_row);
                MPI_Send(&msg[0], 2, MPI_INT, i, 1, MPI_COMM_WORLD);
                // 处理对应行mat1
                MPI_Send(&mat1[begin_row*m], msg[0] * m, MPI_DOUBLE, i, 2, MPI_COMM_WORLD);
                // 整个mat2
                MPI_Send(mat2, m * m, MPI_DOUBLE, i, 3, MPI_COMM_WORLD);
            }
        }
        begin_row = row_per_p * 0;
        end_row = min(row_per_p * (0 + 1) - 1, m - 1);
        printf("process: %d, calculate %d rows(%d to %d)\n", 0, end_row - begin_row + 1, begin_row, end_row);
        matMul(ans, mat1, mat2, end_row - begin_row + 1, m);

        if(size > 1) {
            for(int i = 1; i < size; ++i){
                begin_row = row_per_p * i;
                end_row = min(row_per_p * (i + 1), m - 1);
                MPI_Recv(&ans[begin_row*m], (end_row - begin_row + 1)*m, MPI_DOUBLE, i, 4, MPI_COMM_WORLD, &status);
            }
        }
        double end = MPI_Wtime();
        cout << "ans:" << endl;
        showMatCorner(ans, m, show_size);
        printf("running time: %.5fs\n", end - begin);

    }
    // 子进程
    if(rank != 0) {
        int msg[2];
        MPI_Recv(&msg, 2, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        double* mat1 = new double[msg[0] * msg[1]];
        double* mat2 = new double[msg[1] * msg[1]];
        double* ans = new double [msg[0] * msg[1]](); // 初始化为0
        // 接收 msg[0] 行mat1
        MPI_Recv(&mat1[0], msg[0] * msg[1], MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &status);
        // 接收 整个 mat2
        MPI_Recv(&mat2[0], msg[1] * msg[1], MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &status);

        matMul(ans, mat1, mat2, msg[0], msg[1]);

        MPI_Send(&ans[0], msg[0] * msg[1], MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);

        delete mat1;
        delete mat2;
        delete ans;
    }

    MPI_Finalize();
    return 0;
}