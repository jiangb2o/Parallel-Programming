#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include<stdio.h>

using namespace std;

void ccopy(int n, double x[], double y[]);
void cfft2(int n, double x[], double y[], double w[], double sgn);
void cffti(int n, double w[]);
double cpu_time(void);
double ggl(double *ds);
void step(int n, int mj, double a[], double b[], double c[], double d[], double w[], double sgn);

int prank, psize;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);

    double ctime1;
    int nits = 10000;
    double *w = nullptr;
    double *x = nullptr;
    double *y = nullptr;
    double *z = nullptr;

    freopen("fft_parallel_result.txt", "a+", stdout);
    if(prank == 0) {
        cout << "\n";
        cout << "FFT_SERIAL\n";
        cout << "  C++ parallel version\n";
        cout << "Process Conut: " << psize;
        cout << "\n";
        //
        //  Prepare for tests.
        //
        cout << "\n";
        cout << "  Accuracy check:\n";
        cout << "\n";
        cout << "    FFT ( FFT ( X(1:N) ) ) == N * X(1:N)\n";
        cout << "\n";
        cout << "             N      NITS    Error         Time          Time/Call "
                "    MFLOPS\n";
        cout << "\n";
    }

    double seed = 331.0;
    int n = 1;
    //
    //  LN2 is the log base 2 of N.  Each increase of LN2 doubles N.
    //
    for (int ln2 = 1; ln2 <= 20; ln2++) {
        n = 2 * n;
        //
        //  Allocate storage for the complex arrays W, X, Y, Z.
        //
        //  We handle the complex arithmetic,
        //  and store a complex number as a pair of doubles, a complex vector as
        //  a doubly dimensioned array whose second dimension is 2.
        //
        w = new double[n]();
        x = new double[2 * n]();
        y = new double[2 * n]();
        z = new double[2 * n]();

        int first = 1;

        for (int icase = 0; icase < 2; icase++) {
            // initialize in prank 0
            if(prank == 0) {
                if (first) {
                    for (int i = 0; i < 2 * n; i = i + 2) {
                        x[i] = ggl(&seed);
                        x[i + 1] = ggl(&seed);
                        z[i] = x[i];
                        z[i + 1] = x[i + 1];
                    }
                } else {
                    fill(x, x + 2 * n, 0);
                    fill(z, z + 2 * n, 0);
                }
                //
                //  Initialize the sine and cosine tables.
                //
                cffti(n, w);
            }

            // 广播w给其他进程
            MPI_Bcast(w, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            // 广播x
            MPI_Bcast(x, 2*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            //
            //  Transform forward, back
            //
            if (first) {
                double sgn = +1.0;
                cfft2(n, x, y, w, sgn);
                sgn = -1.0;
                cfft2(n, y, x, w, sgn);
                //
                //  Results should be same as initial multiplied by N.
                //
                if(prank == 0) {
                    double fnm1 = 1.0 / (double)n;
                    double error = 0.0;
                    for (int i = 0; i < 2 * n; i = i + 2) {
                        error = error + pow(z[i] - fnm1 * x[i], 2) +
                                pow(z[i + 1] - fnm1 * x[i + 1], 2);
                    }
                    error = sqrt(fnm1 * error);
                    cout << "  " << setw(12) << n << "  " << setw(8) << nits << "  " << setw(12) << error;
                }
                first = 0;
            } else {
                if(prank == 0) {
                    ctime1 = cpu_time();
                }
                for (int it = 0; it < nits; it++) {
                    double sgn = +1.0;
                    cfft2(n, x, y, w, sgn);
                    sgn = -1.0;
                    cfft2(n, y, x, w, sgn);
                }
                if(prank == 0) {
                    double ctime2 = cpu_time();
                    double ctime = ctime2 - ctime1;
    
                    double flops = 2.0 * (double)nits * (5.0 * (double)n * (double)ln2);
    
                    double mflops = flops / 1.0E+06 / ctime;
    
                    cout << "  " << setw(12) << ctime << "  " << setw(12)
                         << ctime / (double)(2 * nits) << "  " << setw(12) << mflops
                         << "\n";
                }
            }
        }
        if ((ln2 % 4) == 0) {
            nits = max(nits / 10, 1);
        }
        delete[] w;
        delete[] x;
        delete[] y;
        delete[] z;
    }

    if(prank == 0) {
        cout << "\n";
        cout << "FFT_SERIAL:\n";
        cout << "  Normal end of execution.\n";
        cout << "\n";
    }

    MPI_Finalize();

    return 0;
}

//  Purpose:
//    CCOPY copies a complex vector.
//
//  Discussion:
//    The "complex" vector A[N] is actually stored as a double vector B[2*N].
//    The "complex" vector entry A[I] is stored as:
//      B[I*2+0], the real part,
//      B[I*2+1], the imaginary part.
//
//  Parameters:
//    Input, int N, the length of the "complex" array.
//    Input, double X[2*N], the array to be copied.
//    Output, double Y[2*N], a copy of X.
//
void ccopy(int n, double x[], double y[]) {
    int i;

    for (i = 0; i < n; i++) {
        y[i * 2 + 0] = x[i * 2 + 0];
        y[i * 2 + 1] = x[i * 2 + 1];
    }
    return;
}


//  Purpose:
//    CFFT2 performs a complex Fast Fourier Transform.
//
//  Parameters:
//    Input, int N, the psize of the array to be transformed.
//    Input/output, double X[2*N], the data to be transformed.
//    On output, the contents of X have been overwritten by work information.
//    Output, double Y[2*N], the forward or backward FFT of X.
//    Input, double W[N], a table of sines and cosines.
//    Input, double SGN, is +1 for a "forward" FFT and -1 for a "backward" FFT.
//
void cfft2(int n, double x[], double y[], double w[], double sgn) {

    int m = (int)(log((double)n) / log(1.99));
    int mj = 1;
    //
    //  Toggling switch for work array.
    //
    int tgle = 1;
    
    step(n, mj, x, &x[(n / 2) * 2 + 0], y, &y[mj * 2 + 0], w, sgn);
    MPI_Bcast(y, 2 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    if (n == 2) {
        return;
    }

    for (int j = 0; j < m - 2; j++) {
        mj = mj * 2;
        if (tgle) {
            step(n, mj, y, &y[(n / 2) * 2], x, &x[mj * 2], w, sgn);
            MPI_Bcast(x, 2*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            tgle = 0;
        } else {
            step(n, mj, x, &x[(n / 2) * 2], y, &y[mj * 2], w, sgn);
            MPI_Bcast(y, 2*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            tgle = 1;
        }
    }


    //
    //  Last pass thru data: move y to x if needed
    //
    if (tgle) {
        ccopy(n, y, x);
    }

    mj = n / 2;
    step(n, mj, x, &x[(n / 2) * 2], y, &y[mj * 2], w, sgn);
    MPI_Bcast(y, 2*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return;
}

//  Purpose:
//    CFFTI sets up sine and cosine tables needed for the FFT calculation.
//
//  Parameters:
//    Input, int N, the psize of the array to be transformed.
//    Output, double W[N], a table of sines and cosines.
void cffti(int n, double w[]) {
    double arg;
    double aw;
    int i;
    int n2;
    const double pi = 3.141592653589793;

    n2 = n / 2;
    aw = 2.0 * pi / ((double)n);

    for (i = 0; i < n2; i++) {
        arg = aw * ((double)i);
        w[i * 2 + 0] = cos(arg);
        w[i * 2 + 1] = sin(arg);
    }
    return;
}

//  Purpose:
//
//    CPU_TIME reports the elapsed CPU time.
//
//  Parameters:
//    Output, double CPU_TIME, the current total elapsed CPU time in second.
double cpu_time(void) {
    double value;

    value = (double)clock() / (double)CLOCKS_PER_SEC;

    return value;
}

//  Purpose:
//    GGL generates uniformly distributed pseudorandom numbers.
//
//  Parameters:
//    Input/output, double *SEED, used as a seed for the sequence.
//    Output, double GGL, the next pseudorandom value.
double ggl(double *seed) {
    double d2 = 0.2147483647e10;
    double t;
    double value;

    t = *seed;
    t = fmod(16807.0 * t, d2);
    *seed = t;
    value = (t - 1.0) / (d2 - 1.0);

    return value;
}

//  Purpose:
//    STEP carries out one step of the workspace version of CFFT2.
void step(int n, int mj, double a[], double b[], double c[], double d[],
        double w[], double sgn) {
        
        int mj2 = 2 * mj;
        int lj = n / mj2;

        int local_n = lj / psize;
        int start = 0, end = 0;

        if(local_n == 0) {
            local_n = 1;
            if(prank < lj) {
                start = prank;
                end = start + 1;
            }
        } else {
            start = prank * local_n;
            end = start + local_n;
        }
        
        for (int j = start; j < end; j++) {
            int jw = j * mj;
            int ja = jw;
            int jb = ja;
            int jc = j * mj2;
            int jd = jc;
            
            double wjw[2] = {w[jw * 2 + 0], w[jw * 2 + 1]};

            if (sgn < 0.0) {
                wjw[1] = -wjw[1];
            }

            for (int k = 0; k < mj; k++) {
                c[(jc + k) * 2 + 0] = a[(ja + k) * 2 + 0] + b[(jb + k) * 2 + 0];
                c[(jc + k) * 2 + 1] = a[(ja + k) * 2 + 1] + b[(jb + k) * 2 + 1];

                double ambr = a[(ja + k) * 2 + 0] - b[(jb + k) * 2 + 0];
                double ambu = a[(ja + k) * 2 + 1] - b[(jb + k) * 2 + 1];

                d[(jd + k) * 2 + 0] = wjw[0] * ambr - wjw[1] * ambu;
                d[(jd + k) * 2 + 1] = wjw[1] * ambr + wjw[0] * ambu;
            }

            if(prank == 0) {
                for(int i = 1; i < min(lj, psize); ++i) {
                    MPI_Recv(&c[(i*local_n+j) * mj2 * 2], 2 * mj, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&d[(i*local_n+j) * mj2 * 2], 2 * mj, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
            else if(prank != 0 && prank < lj) {
                MPI_Send(&c[jc * 2], 2 * mj, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                MPI_Send(&d[jd * 2], 2 * mj, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            }
        }
    return;
}