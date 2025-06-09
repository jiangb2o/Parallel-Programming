#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include "mkl.h"

using namespace std;

int main()
{
    double *mat1, *mat2, *ans;
    int m, n, k, i, j;
    double alpha, beta;

    m = 1024, n = 1024, k = 1024;
    cout << "input m, n, k:" << endl;
    cin >> m >> n >> k;
    alpha = 1.0; beta = 0.0;
    mat1 = (double *)mkl_malloc( m*n*sizeof( double ), 64 );
    mat2 = (double *)mkl_malloc( n*k*sizeof( double ), 64 );
    ans = (double *)mkl_malloc( m*k*sizeof( double ), 64 );

    for (i = 0; i < (m*n); i++) {
        mat1[i] = (double)(i+1);
    }

    for (i = 0; i < (n*k); i++) {
        mat2[i] = (double)(-i-1);
    }

    for (i = 0; i < (m*k); i++) {
        ans[i] = 0.0;
    }

    auto begin = chrono::high_resolution_clock::now();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, n, k, alpha, mat1, k, mat2, n, beta, ans, n);
    printf ("\n Computations completed.\n\n");
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

    mkl_free(mat1);
    mkl_free(mat2);
    mkl_free(ans);

    return 0;
}