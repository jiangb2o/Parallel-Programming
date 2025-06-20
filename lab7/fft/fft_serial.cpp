#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>

using namespace std;

int main();
void ccopy(int n, double x[], double y[]);
void cfft2(int n, double x[], double y[], double w[], double sgn);
void cffti(int n, double w[]);
double cpu_time(void);
double ggl(double *ds);
void step(int n, int mj, double a[], double b[], double c[], double d[], double w[], double sgn);


int main() {
    double ctime, ctime1, ctime2;
    double error;
    int first;
    double flops;
    double fnm1;
    int i;
    int icase;
    int it;
    int ln2;
    double mflops;
    int n;
    int nits = 10000;
    static double seed;
    double sgn;
    double *w = nullptr;
    double *x = nullptr;
    double *y = nullptr;
    double *z = nullptr;
    double z0, z1;

    cout << "\n";
    cout << "FFT_SERIAL\n";
    cout << "  C++ version\n";
    cout << "\n";
    cout << "  Demonstrate an implementation of the Fast Fourier Transform\n";
    cout << "  of a complex data vector.\n";
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

    seed = 331.0;
    n = 1;
    //
    //  LN2 is the log base 2 of N.  Each increase of LN2 doubles N.
    //
    for (ln2 = 1; ln2 <= 20; ln2++) {
        n = 2 * n;
        //
        //  Allocate storage for the complex arrays W, X, Y, Z.
        //
        //  We handle the complex arithmetic,
        //  and store a complex number as a pair of doubles, a complex vector as
        //  a doubly dimensioned array whose second dimension is 2.
        //
        w = new double[n];
        x = new double[2 * n];
        y = new double[2 * n];
        z = new double[2 * n];

        first = 1;

        for (icase = 0; icase < 2; icase++) {
            if (first) {
                for (i = 0; i < 2 * n; i = i + 2) {
                    z0 = ggl(&seed);
                    z1 = ggl(&seed);
                    x[i] = z0;
                    z[i] = z0;
                    x[i + 1] = z1;
                    z[i + 1] = z1;
                }
            } else {
                for (i = 0; i < 2 * n; i = i + 2) {
                    z0 = 0.0;
                    z1 = 0.0;
                    x[i] = z0;
                    z[i] = z0;
                    x[i + 1] = z1;
                    z[i + 1] = z1;
                }
            }
            //
            //  Initialize the sine and cosine tables.
            //
            cffti(n, w);
            //
            //  Transform forward, back
            //
            if (first) {
                sgn = +1.0;
                cfft2(n, x, y, w, sgn);
                sgn = -1.0;
                cfft2(n, y, x, w, sgn);
                //
                //  Results should be same as initial multiplied by N.
                //
                fnm1 = 1.0 / (double)n;
                error = 0.0;
                for (i = 0; i < 2 * n; i = i + 2) {
                    error = error + pow(z[i] - fnm1 * x[i], 2) +
                            pow(z[i + 1] - fnm1 * x[i + 1], 2);
                }
                error = sqrt(fnm1 * error);
                cout << "  " << setw(12) << n << "  " << setw(8) << nits << "  " << setw(12) << error;
                first = 0;
            } else {
                ctime1 = cpu_time();
                for (it = 0; it < nits; it++) {
                    sgn = +1.0;
                    cfft2(n, x, y, w, sgn);
                    sgn = -1.0;
                    cfft2(n, y, x, w, sgn);
                }
                ctime2 = cpu_time();
                ctime = ctime2 - ctime1;

                flops = 2.0 * (double)nits * (5.0 * (double)n * (double)ln2);

                mflops = flops / 1.0E+06 / ctime;

                cout << "  " << setw(12) << ctime << "  " << setw(12)
                     << ctime / (double)(2 * nits) << "  " << setw(12) << mflops
                     << "\n";
            }
        }
        if ((ln2 % 4) == 0) {
            nits = nits / 10;
        }
        if (nits < 1) {
            nits = 1;
        }
        delete[] w;
        delete[] x;
        delete[] y;
        delete[] z;
    }

    cout << "\n";
    cout << "FFT_SERIAL:\n";
    cout << "  Normal end of execution.\n";
    cout << "\n";

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
//    Input, int N, the size of the array to be transformed.
//    Input/output, double X[2*N], the data to be transformed.
//    On output, the contents of X have been overwritten by work information.
//    Output, double Y[2*N], the forward or backward FFT of X.
//    Input, double W[N], a table of sines and cosines.
//    Input, double SGN, is +1 for a "forward" FFT and -1 for a "backward" FFT.
//
void cfft2(int n, double x[], double y[], double w[], double sgn) {
    int j;
    int m;
    int mj;
    int tgle;

    m = (int)(log((double)n) / log(1.99));
    mj = 1;
    //
    //  Toggling switch for work array.
    //
    tgle = 1;
    step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0],
         &y[mj * 2 + 0], w, sgn);

    if (n == 2) {
        return;
    }

    for (j = 0; j < m - 2; j++) {
        mj = mj * 2;
        if (tgle) {
            step(n, mj, &y[0 * 2 + 0], &y[(n / 2) * 2 + 0], &x[0 * 2 + 0],
                 &x[mj * 2 + 0], w, sgn);
            tgle = 0;
        } else {
            step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0],
                 &y[mj * 2 + 0], w, sgn);
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
    step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0],
         &y[mj * 2 + 0], w, sgn);

    return;
}

//  Purpose:
//    CFFTI sets up sine and cosine tables needed for the FFT calculation.
//
//  Parameters:
//    Input, int N, the size of the array to be transformed.
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
    double ambr;
    double ambu;
    int j;
    int ja;
    int jb;
    int jc;
    int jd;
    int jw;
    int k;
    int lj;
    int mj2;
    double wjw[2];

    mj2 = 2 * mj;
    lj = n / mj2;

    for (j = 0; j < lj; j++) {
        jw = j * mj;
        ja = jw;
        jb = ja;
        jc = j * mj2;
        jd = jc;

        wjw[0] = w[jw * 2 + 0];
        wjw[1] = w[jw * 2 + 1];

        if (sgn < 0.0) {
            wjw[1] = -wjw[1];
        }

        for (k = 0; k < mj; k++) {
            c[(jc + k) * 2 + 0] = a[(ja + k) * 2 + 0] + b[(jb + k) * 2 + 0];
            c[(jc + k) * 2 + 1] = a[(ja + k) * 2 + 1] + b[(jb + k) * 2 + 1];

            ambr = a[(ja + k) * 2 + 0] - b[(jb + k) * 2 + 0];
            ambu = a[(ja + k) * 2 + 1] - b[(jb + k) * 2 + 1];

            d[(jd + k) * 2 + 0] = wjw[0] * ambr - wjw[1] * ambu;
            d[(jd + k) * 2 + 1] = wjw[1] * ambr + wjw[0] * ambu;
        }
    }
    return;
}