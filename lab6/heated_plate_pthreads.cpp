#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include "parallel_for.h"

int main(int argc, char *argv[]);

/******************************************************************************/

int main(int argc, char *argv[])
{
#define M 500
#define N 500

    const int MAX_THREAD = 4;
    double diff;
    double epsilon = 0.001;
    int i;
    int iterations;
    int iterations_print;
    int j;
    double mean;
    double my_diff;
    double u[M][N];
    double w[M][N];

    int threads_num = 4;

    if(argc >= 2) {
        threads_num = std::stoi(argv[1]);
    }

    printf("\n");
    printf("HEATED_PLATE_OPENMP\n");
    printf("  C/OpenMP version\n");
    printf("  A program to solve for the steady state temperature distribution\n");
    printf("  over a rectangular plate.\n");
    printf("\n");
    printf("  Spatial grid of %d by %d points.\n", M, N);
    printf("  The iteration will be repeated until the change is <= %e\n", epsilon);
    printf("  Number of processors available = %d\n", 4);
    printf("  Number of threads =              %d\n", threads_num);
    /*
        Set the boundary values, which don't change.
    */
    mean = 0.0;

    struct Args {
        double* mean;
        double* thread_mean;
        double** w;
        Args(double* _mean, double _w[M][N]): mean(_mean){
            thread_mean = new double[MAX_THREAD];
            set_thread_mean_zero();
            w = new double*[M];
            for(int i = 0; i < M; ++i) {
                w[i] = _w[i];
            }
        }
        void set_thread_mean_zero() {
            for(int i = 0; i < MAX_THREAD; ++i) {
                thread_mean[i] = 0;
            }
        }
        void reduce_mean() {
            for(int i = 0; i < MAX_THREAD; ++i) {
                *mean += thread_mean[i];
            }
        }
        ~Args() {
            delete [] w;
            delete [] thread_mean;
        }
    };
    Args* arg = new Args(&mean, w);
    parallel_for(1, M - 1, 1, [](int thread_id, int i, void* arg) -> void* {
        Args* args = static_cast<Args*>(arg);
        args->w[i][0] = 100.0;
    }, arg, MAX_THREAD);
    parallel_for(1, M - 1, 1, [](int thread_id, int i, void* arg) -> void* {
        Args* args = static_cast<Args*>(arg);
        args->w[i][N - 1] = 100.0;
    }, arg, MAX_THREAD);
    parallel_for(0, N, 1, [](int thread_id, int j, void* arg) -> void* {
        Args* args = static_cast<Args*>(arg);
        args->w[M - 1][j] = 100.0;
    }, arg, MAX_THREAD);
    parallel_for(0, N, 1, [](int thread_id, int j, void* arg) -> void* {
        Args* args = static_cast<Args*>(arg);
        args->w[0][j] = 0.0;
    }, arg, MAX_THREAD);


/*
    Average the boundary values, to come up with a reasonable
    initial value for the interior.
*/
    parallel_for(1, M - 1, 1, [](int thread_id, int i, void* arg) -> void* {
        Args* args = static_cast<Args*>(arg);
        args->thread_mean[thread_id] = args->thread_mean[thread_id] + args->w[i][0] + args->w[i][N - 1];
    }, arg, MAX_THREAD);

    arg->reduce_mean();
    arg->set_thread_mean_zero();

    parallel_for(0, N, 1, [](int thread_id, int j, void* arg) -> void* {
        Args* args = static_cast<Args*>(arg);
        args->thread_mean[thread_id] = args->thread_mean[thread_id] + args->w[M - 1][j] + args->w[0][j];
    }, arg, MAX_THREAD);

    arg->reduce_mean();
    

    mean = mean / (double)(2 * M + 2 * N - 4);
    printf("\n");
    printf("  MEAN = %f\n", mean);
/*
    Initialize the interior solution to the mean value.
*/

    parallel_for(1, M - 1, 1, [](int thread_id, int i, void* arg) -> void* {
        Args* args = static_cast<Args*>(arg);
        for(int j = 1; j < N - 1; j++) {
            args->w[i][j] = *(args->mean);
        }
    }, arg, MAX_THREAD);
    /*
        iterate until the  new solution W differs from the old solution U
        by no more than EPSILON.
    */
    iterations = 0;
    iterations_print = 1;
    printf("\n");
    printf(" Iteration  Change\n");
    printf("\n");

    struct Args2 {
        double** u;
        double** w;
        double* diff;
        double* my_diff;
        int thread_nums;
        Args2(double _u[M][N], double _w[M][N], double* _diff, int _thread_nums) {
            w = new double*[M];
            for(int i = 0; i < M; ++i) {
                w[i] = _w[i];
            }
            u = new double*[M];
            for(int i = 0; i < M; ++i) {
                u[i] = _u[i];
            }
            diff = _diff;
            thread_nums = _thread_nums;
            my_diff = new double[thread_nums];
        }
        void set_my_diff_zero() {
            for(int i = 0; i < thread_nums; ++i) {
                my_diff[i] = 0;
            }
        }
        void set_max_diff() {
            for(int i = 0; i < thread_nums; ++i) {
                *diff = std::max(my_diff[i], *diff);
            }
        }
        ~Args2(){
            delete []u;
            delete []w;
            delete [] my_diff;
        }
    };

    Args2* arg2 = new Args2(u, w, &diff, threads_num);

    auto begin = std::chrono::high_resolution_clock::now();

    diff = epsilon;

    while (epsilon <= diff) {
        /*
            Save the old solution in U.
        */
        parallel_for(0, M, 1, [](int thread_id, int i, void* arg) -> void* {
            Args2* args = static_cast<Args2*>(arg);
            for(int j = 0; j < N; ++j) {
                args->u[i][j] = args->w[i][j];
            }
        }, arg2, threads_num);

        /*
            Determine the new estimate of the solution at the interior points.
            The new solution W is the average of north, south, east and west neighbors.
        */
        parallel_for(1, M - 1, 1, [](int thread_id, int i, void* arg) -> void* {
            Args2* args = static_cast<Args2*>(arg);
            double** u = args->u;
            for(int j = 1; j < N - 1; ++j) {
                args->w[i][j] = (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1]) / 4.0;
            }
        }, arg2, threads_num);
        
        /*
            C and C++ cannot compute a maximum as a reduction operation.

            Therefore, we define a private variable MY_DIFF for each thread.
            Once they have all computed their values, we use a CRITICAL section
            to update DIFF.
        */
        diff = 0.0;
        arg2->set_my_diff_zero();

        parallel_for(1, M - 1, 1, [](int thread_id, int i, void* arg) -> void* {
            Args2* args = static_cast<Args2*>(arg);
            for(int j = 1; j < N - 1; ++j) {
                if(args->my_diff[thread_id] < fabs(args->w[i][j] - args->u[i][j])) {
                    args->my_diff[thread_id] = fabs(args->w[i][j] - args->u[i][j]);
                }
            }
        }, arg2, threads_num);

        arg2->set_max_diff();

        iterations++;
        if (iterations == iterations_print) {
            printf("  %8d  %f\n", iterations, diff);
            iterations_print = 2 * iterations_print;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    printf("\n");
    printf("  %8d  %f\n", iterations, diff);
    printf("\n");
    printf("  Error tolerance achieved.\n");
    printf("  Wallclock time = %f\n", elapsed * 1e-9);
    /*
      Terminate.
    */
    printf("\n");
    printf("HEATED_PLATE_OPENMP:\n");
    printf("  Normal end of execution.\n");

    return 0;

#undef M
#undef N
}