// g++ -pthread filename.cpp -o filename
#include<pthread.h>
#include<iostream>
#include<stdio.h>
#include<string>
#include<random>
#include<chrono>
#include<algorithm>

using namespace std;

double ans = 0;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

void usage(const char* program_name) {
    printf("useage:./%s <num of threads> <num of sampling>\n", program_name);
}

random_device rd;
mt19937 gen(rd());
uniform_int_distribution<> distrib(0, 1);

inline double static getADouble() {
    return distrib(gen);
}


void* MonteCalro(void* args) {
    int* para = static_cast<int*>(args);
    int cnt = 0;
    
    double x, y;
    
    for(int i = 0; i <  *para; ++i) {
        x = getADouble();
        y = getADouble();

        if(pow(pow(x, 2) + pow(y, 2), 0.5) <= 1)
            cnt++;
    }
    pthread_mutex_lock(&lock);
    ans += cnt;
    pthread_mutex_unlock(&lock);
    return NULL;
}

int main(int argc, char* argv[]) {
    int thread_num = 1;
    int sampling_num = 1;
    if(argc >= 2) {
        try {
            thread_num = std::stoi(argv[1]);
        } catch (std::invalid_argument const& ex) {
            usage(argv[0]);
        }
    }
    if(argc >= 3) {
        try {
            sampling_num = std::stoi(argv[2]);
        } catch (std::invalid_argument const& ex) {
            usage(argv[0]);
        }
    }
    FILE *fp = fopen("montecalro.log", "a+");
    fprintf(fp, "\n\n=====number of threads: %d, point of sampling: %d=====\n", thread_num, sampling_num);

    int n_each_thread = sampling_num / thread_num;
    pthread_t* threads = new pthread_t[thread_num];

    // 创建线程
    auto begin = chrono::high_resolution_clock::now();
    for(int i = 0; i < thread_num; ++i) {
        pthread_create(&threads[i], NULL, MonteCalro, static_cast<void*>(&n_each_thread));
    }
    // 回收线程
    for(int i = 0; i < thread_num; ++i) {
        pthread_join(threads[i], NULL);
    }
    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);

    ans = ans / sampling_num * 4;

    double true_ans = M_PI;
    fprintf(fp, "ans: %f\n", ans);
    fprintf(fp, "true ans: %f\n", true_ans);
    fprintf(fp, "acc: %.2f%\n", (true_ans - abs(ans - true_ans)) / true_ans * 100);
    fprintf(fp,"running time: %.4f s\n", elapsed * 1e-9);

    fclose(fp);

    return 0;
}

