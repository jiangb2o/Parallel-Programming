#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include "parallel_for.h"

int main(int argc, char *argv[]);

/******************************************************************************/

int m, n;
int main(int argc, char *argv[])
{
    m = 512;
    n = 512;
    int threads_num = 4;
    if(argc >= 2) {
        threads_num = std::stoi(argv[1]);
    }
    if(argc >= 3) {
        m = std::stoi(argv[2]);
        n = std::stoi(argv[2]);
    }

    const int MAX_THREAD = 4;
    double diff;
    double epsilon = 0.001;
    int i;
    int iterations;
    int iterations_print;
    int j;
    double mean;
    double my_diff;
    double** u = new double* [m];
    double** w = new double* [m];
    for(int i = 0; i < m; ++i) {
        u[i] = new double[n];
        w[i] = new double[n];
    }

    FILE *fp = fopen("heated_plate_result.txt", "a+");
    fprintf(fp, "==================================================");
    fprintf(fp,"  \nSpatial grid of %d by %d points.\n", m, n);
    fprintf(fp, "  The iteration will be repeated until the change is <= %e\n", epsilon);
    fprintf(fp, "  Number of processors available = %d\n", 4);
    fprintf(fp, "  Number of threads =              %d\n", threads_num);
    /*
        Set the boundary values, which don't change.
    */
    mean = 0.0;

    struct Args {
        double* mean;
        double* thread_mean;
        double** w;
        Args(double* _mean, double** _w): mean(_mean){
            thread_mean = new double[MAX_THREAD];
            set_thread_mean_zero();
            w = new double*[m];
            for(int i = 0; i < m; ++i) {
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
    parallel_for(1, m - 1, 1, [](int thread_id, int i, void* arg) -> void* {
        Args* args = static_cast<Args*>(arg);
        args->w[i][0] = 100.0;
    }, arg, MAX_THREAD);
    parallel_for(1, m - 1, 1, [](int thread_id, int i, void* arg) -> void* {
        Args* args = static_cast<Args*>(arg);
        args->w[i][n - 1] = 100.0;
    }, arg, MAX_THREAD);
    parallel_for(0, n, 1, [](int thread_id, int j, void* arg) -> void* {
        Args* args = static_cast<Args*>(arg);
        args->w[m - 1][j] = 100.0;
    }, arg, MAX_THREAD);
    parallel_for(0, n, 1, [](int thread_id, int j, void* arg) -> void* {
        Args* args = static_cast<Args*>(arg);
        args->w[0][j] = 0.0;
    }, arg, MAX_THREAD);


/*
    Average the boundary values, to come up with a reasonable
    initial value for the interior.
*/
    parallel_for(1, m - 1, 1, [](int thread_id, int i, void* arg) -> void* {
        Args* args = static_cast<Args*>(arg);
        args->thread_mean[thread_id] = args->thread_mean[thread_id] + args->w[i][0] + args->w[i][n - 1];
    }, arg, MAX_THREAD);

    arg->reduce_mean();
    arg->set_thread_mean_zero();

    parallel_for(0, n, 1, [](int thread_id, int j, void* arg) -> void* {
        Args* args = static_cast<Args*>(arg);
        args->thread_mean[thread_id] = args->thread_mean[thread_id] + args->w[m - 1][j] + args->w[0][j];
    }, arg, MAX_THREAD);

    arg->reduce_mean();
    

    mean = mean / (double)(2 * m + 2 * n - 4);
    fprintf(fp, "\n");
    fprintf(fp, "  MEAN = %f\n", mean);
/*
    Initialize the interior solution to the mean value.
*/

    parallel_for(1, m - 1, 1, [](int thread_id, int i, void* arg) -> void* {
        Args* args = static_cast<Args*>(arg);
        for(int j = 1; j < n - 1; j++) {
            args->w[i][j] = *(args->mean);
        }
    }, arg, MAX_THREAD);
    /*
        iterate until the  new solution W differs from the old solution U
        by no more than EPSILON.
    */
    iterations = 0;
    iterations_print = 1;
    fprintf(fp, "\n");
    fprintf(fp, " Iteration  Change\n");
    fprintf(fp, "\n");

    struct Args2 {
        double** u;
        double** w;
        double* diff;
        double* my_diff;
        int thread_nums;
        Args2(double** _u, double** _w, double* _diff, int _thread_nums) {
            w = new double*[m];
            for(int i = 0; i < m; ++i) {
                w[i] = _w[i];
            }
            u = new double*[m];
            for(int i = 0; i < m; ++i) {
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
        parallel_for(0, m, 1, [](int thread_id, int i, void* arg) -> void* {
            Args2* args = static_cast<Args2*>(arg);
            for(int j = 0; j < n; ++j) {
                args->u[i][j] = args->w[i][j];
            }
        }, arg2, threads_num);

        /*
            Determine the new estimate of the solution at the interior points.
            The new solution W is the average of north, south, east and west neighbors.
        */
        parallel_for(1, m - 1, 1, [](int thread_id, int i, void* arg) -> void* {
            Args2* args = static_cast<Args2*>(arg);
            double** u = args->u;
            for(int j = 1; j < n - 1; ++j) {
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

        parallel_for(1, m - 1, 1, [](int thread_id, int i, void* arg) -> void* {
            Args2* args = static_cast<Args2*>(arg);
            for(int j = 1; j < n - 1; ++j) {
                if(args->my_diff[thread_id] < fabs(args->w[i][j] - args->u[i][j])) {
                    args->my_diff[thread_id] = fabs(args->w[i][j] - args->u[i][j]);
                }
            }
        }, arg2, threads_num);

        arg2->set_max_diff();

        iterations++;
        if (iterations == iterations_print) {
            fprintf(fp, "  %8d  %f\n", iterations, diff);
            iterations_print = 2 * iterations_print;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    fprintf(fp, "\n");
    fprintf(fp, "  %8d  %f\n", iterations, diff);
    fprintf(fp, "\n");
    fprintf(fp, "  Error tolerance achieved.\n");
    fprintf(fp, "  Wallclock time = %f\n", elapsed * 1e-9);

    return 0;
}