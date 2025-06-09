#include<iostream>
#include <random>
#include<stdio.h>
#include<cmath>
#include<mpi.h>

using namespace std;
const int show_size = 8;

void getRandomMat(double* mat, int m, int n) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(1, 10);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mat[i*n+j] = distrib(gen);
            //mat[i*n+j] = 1;
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

class Messages {
public:
    int avg_row;
    double mat2[];

    static MPI_Datatype message_type;

    static void build_mpi_type(int m) {
        int block_lengths[2] = {1, m * m};
        MPI_Datatype types[2] = {MPI_INT, MPI_DOUBLE};
        MPI_Aint displacements[2];

        Messages tmp;

        MPI_Aint base_address;
        MPI_Get_address(&tmp, &base_address);
        MPI_Get_address(&tmp.avg_row, &displacements[0]);
        MPI_Get_address(tmp.mat2, &displacements[1]);

        for(int i = 0; i < 2; ++i) {
            displacements[i] -= base_address;
        }
        
        MPI_Type_create_struct(2, block_lengths, displacements, types, &message_type);
        MPI_Type_commit(&message_type);
    }
};
// 静态变量类外声明
MPI_Datatype Messages::message_type;

int main(int argc, char* argv[]) {
    int rank = 0, size = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;
    
    double* mat1 = nullptr;
    double* mat2 = nullptr;
    double* ans = nullptr;
    Messages* message = nullptr;
    
    int m = 0;
    int avg_row = 0;

    // 主进程
    double begin, end;
    if(rank == 0) {
        cout << "input matrix size m: ";
        cin >> m;
        mat1 = new double[m*m];
        mat2 = new double[m*m];
        ans = new double [m*m](); // 初始化为0

        getRandomMat(mat1, m, m);
        getRandomMat(mat2, m, m);

        cout << "mat1:" << endl;
        showMatCorner(mat1, m, show_size);
        cout << "mat2:" << endl;
        showMatCorner(mat2, m, show_size);
        cout << "ans:" << endl;
        showMatCorner(ans, m, show_size);
        
        avg_row = m / size;
        begin = MPI_Wtime();
    }
    // 广播m的值
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 所有进程获得m值后进行构建
    Messages::build_mpi_type(m);
    // 分配空间
    message = (Messages*)malloc(sizeof(int) * 1 + sizeof(double) * m * m);

    // 主进程为message赋值
    if(rank == 0) {
        message->avg_row = avg_row;
        for(int i = 0; i < m; ++i) {
            for(int j = 0; j < m; ++j) {
                message->mat2[i * m + j] = mat2[i * m + j];
            }
        }
    }
    // 广播聚合消息
    MPI_Bcast(message, 1, Messages::message_type, 0, MPI_COMM_WORLD);

    // 非主进程从聚合消息中提取变量
    if(rank != 0) {
        avg_row = message->avg_row;
        // 非主进程要为mat2分配空间
        mat2 = new double[m * m];
        for(int i = 0; i < m; ++i) {
            for(int j = 0; j < m; ++j) {
                mat2[i * m + j] = message->mat2[i * m + j];
            }
        }
    }

    if(m % size != 0) {
        if(rank == 0) {
            cout << "size of matrix m mod process num != 0!" << endl;
        }
    } else {
        if(size == 1) {
            matMul(ans, mat1, mat2, m, m);
            end = MPI_Wtime();
            cout << "ans:" << endl;
            showMatCorner(ans, m, show_size);
            printf("running time: %.5fs\n", end - begin);
        } else {
            double* local_mat1 = new double[avg_row * m];
            double* local_ans = new double[avg_row * m]();
            // 接收 avg_row 行 mat1
            cout << "process " << rank << " scattering" << endl;
            MPI_Scatter(mat1, avg_row * m, MPI_DOUBLE, local_mat1, avg_row * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            // 计算
            cout << "process " << rank << " calculating" << endl;
            matMul(local_ans, local_mat1, mat2, avg_row, m);
            // 收集结果
            cout << "process " << rank << " gathering" << endl;
            MPI_Gather(local_ans, avg_row * m, MPI_DOUBLE, ans, avg_row * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if(rank == 0) {
                end = MPI_Wtime();
                cout << "ans:" << endl;
                showMatCorner(ans, m, show_size);
                printf("running time: %.5fs\n", end - begin);
                if(mat1) delete[] mat1;
                if(ans) delete[] ans;
            }
            if(message) delete message;
            if(local_mat1) delete[] local_mat1;
            if(mat2) delete[] mat2;
            if(local_ans) delete[] local_ans;
        }
    }
    MPI_Finalize();
    return 0;
}

