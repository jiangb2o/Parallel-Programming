#include <stdio.h>

#include <chrono>
#include <random>
#include <iostream>
#include <vector>

using namespace std;

void getRandomMat(int m, int n, vector<vector<double>>& mat) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mat[i][j] = (double)(rand() % 100 + 1);
        }
    }
}

void matMul(vector<vector<double>>& mat1, vector<vector<double>>& mat2,
            vector<vector<double>>& ans) {
    int m = mat1.size(), n = mat2.size(), k = mat2[0].size();
    for (int i = 0; i < m; ++i) {
        vector<double> mat1_row = mat1[i];
        for (int j = 0; j < k; ++j) {
            for (int l = 0; l < n; ++l) {
                ans[i][j] += mat1_row[l] * mat2[l][j];
            }
        }
    }
}

// 调整循环顺序
void matMul2(vector<vector<double>>& mat1, vector<vector<double>>& mat2,
             vector<vector<double>>& ans) {
    int m = mat1.size(), n = mat2.size(), k = mat2[0].size();
    for (int i = 0; i < m; ++i) {
        for (int l = 0; l < n; ++l) {
            double a = mat1[i][l];
            for (int j = 0; j < k; ++j) {
                ans[i][j] += a * mat2[l][j];
            }
        }
    }
}

// 循环展开
void matMul3(vector<vector<double>>& mat1, vector<vector<double>>& mat2,
             vector<vector<double>>& ans) {
    int m = mat1.size(), n = mat2.size(), k = mat2[0].size();
    int i, j, l;
    for (i = 0; i < m; ++i) {
        for (l = 0; l < n; ++l) {
            double a = mat1[i][l];
            for (j = 0; j < ((k) & (~3)); j += 4) {
                ans[i][j] += a * mat2[l][j];
                ans[i][j + 1] += a * mat2[l][j + 1];
                ans[i][j + 2] += a * mat2[l][j + 2];
                ans[i][j + 3] += a * mat2[l][j + 3];
            }
            // 不足四列的部分
            for (; j < k; ++j) {
                ans[i][j] += a * mat2[l][j];
            }
        }
    }
}

int main(void) {
    int m, n, k;
    cout << "input m, n, k:" << endl;
    cin >> m >> n >> k;
    vector<vector<double>> mat1(m, vector<double>(n)),
        mat2(n, vector<double>(k));
    vector<vector<double>> ans(m, vector<double>(k));
    srand(time(0));
    getRandomMat(m, n, mat1);
    getRandomMat(n, k, mat2);
    auto begin = chrono::high_resolution_clock::now();
    /*--------------------------------------------------------*/
    // matMul(mat1, mat2, ans);

    // 调整循环顺序
    // matMul2(mat1, mat2, ans);

    // 循环展开
    matMul3(mat1, mat2, ans);
    /*--------------------------------------------------------*/
    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);
    printf("running time: %.4f s\n", elapsed * 1e-9);

    printf(" Top left corner of mat1: \n");
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            printf("%12.3f", mat1[i][j]);
        }
        printf("\n");
    }

    printf("\n Top left corner of mat2: \n");
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            printf("%12.3f", mat2[i][j]);
        }
        printf("\n");
    }

    printf("\n Top left corner of ans: \n");
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            printf("%12.3f", ans[i][j]);
        }
        printf("\n");
    }

    return 0;
}
