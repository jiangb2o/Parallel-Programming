// g++ -pthread filename.cpp -o filename
#include<pthread.h>
#include<iostream>
#include<stdio.h>
#include<string>
#include<random>
#include<chrono>
#include<algorithm>

using namespace std;

int *arr;
int ans;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

void usage(const char* program_name) {
    printf("useage:./%s <num of threads> <array size * 10^6>\n", program_name);
}

void getRandomArray(int* arr, int n) {
    random_device rd;
    mt19937 gen(rd());
    // 1*10^6 < n < 128 * 10^6
    // sum < 2147483647
    uniform_int_distribution<> distrib(1, 10);
    for (int i = 0; i < n; i++) {
        arr[i] = distrib(gen);
    }
}

struct arraySumPara{
    int n;
    int total;
    int thread_no;
    arraySumPara(){n = 0; total = 0; thread_no = 0;}
    arraySumPara(int _n, int _total, int _thread_no){
        n = _n;
        total = _total;
        thread_no = _thread_no;
    }
};

void* arraySum(void* args) {
    arraySumPara* para =  static_cast<arraySumPara*>(args);
    int n = para->n;
    int begin = para->thread_no * n;
    int end = min((para->thread_no + 1) * n, para->total);
    int tmp = 0;
    // for(int i = begin; i < end; ++i) {
    //     pthread_mutex_lock(&lock);
    //     ans += arr[i];
    //     pthread_mutex_unlock(&lock);
    // }
    for(int i = begin; i < end; ++i) {
        tmp += arr[i];
    }
    pthread_mutex_lock(&lock);
    ans += tmp;
    pthread_mutex_unlock(&lock);
    return NULL;
}

int main(int argc, char* argv[]) {
    int thread_num = 1;
    int array_size = 1;
    if(argc >= 2) {
        try {
            thread_num = std::stoi(argv[1]);
        } catch (std::invalid_argument const& ex) {
            usage(argv[0]);
        }
    }
    if(argc >= 3) {
        try {
            array_size = std::stoi(argv[2]);
        } catch (std::invalid_argument const& ex) {
            usage(argv[0]);
        }
    }
    FILE *fp = fopen("arraySum_resultv2.txt", "a+");
    fprintf(fp, "\n\n=====number of threads: %d, size of arr: %d x 10 ^ 6=====\n", thread_num, array_size);

    array_size *= (int)1e6;
    arr = new int[array_size];
    getRandomArray(arr, array_size);

    int n_each_thread = array_size / thread_num;
    pthread_t* threads = new pthread_t[thread_num];

    // 创建参数
    arraySumPara* threads_para = new arraySumPara[thread_num];
    for(int i = 0; i < thread_num; ++i) {
        threads_para[i] = arraySumPara(n_each_thread ,array_size, i);
    }

    // 创建线程
    auto begin = chrono::high_resolution_clock::now();
    for(int i = 0; i < thread_num; ++i) {
        pthread_create(&threads[i], NULL, arraySum, static_cast<void*>(&threads_para[i]));
    }
    // 回收线程
    for(int i = 0; i < thread_num; ++i) {
        pthread_join(threads[i], NULL);
    }
    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);

    int true_ans = std::accumulate(arr, arr + array_size, 0);
    fprintf(fp, "ans: %d\n", ans);
    fprintf(fp, "true ans: %d\n", true_ans);

    fprintf(fp,"running time: %.4f s\n", elapsed * 1e-9);
    fclose(fp);
    delete [] arr;
    delete [] threads;
    delete [] threads_para;

    return 0;
}

