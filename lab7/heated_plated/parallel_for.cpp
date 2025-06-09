#include<iostream>
#include<pthread.h>
#include"parallel_for.h"

struct ThreadArgs{
    int thread_id;
    int start;
    int end;
    int inc;
    void* (*func)(int, int, void*);
    void* arg;
};

void* threadFunc(void* args) {
    ThreadArgs* thread_args = (ThreadArgs*)args;
    // std::cout << "Thread " << thread_args->thread_id << ": start and end is " << thread_args->start << "," << thread_args->end << std::endl;
    for (int i = thread_args->start; i < thread_args->end; i += thread_args->inc) {
        thread_args->func(thread_args->thread_id ,i, thread_args->arg);
    }

    delete thread_args;
    return nullptr;
}

void parallel_for(int start, int end, int inc, void* (*func)(int, int, void*), void* arg, int num_threads) {
    pthread_t threads[num_threads];
    // 迭代次数
    int iters = (end - start + inc - 1) / inc;
    // 每个线程处理的迭代次数
    int iters_per_thread = iters / num_threads;
    // [start, end)
    for (int i = 0; i < num_threads; ++i) {
        int thread_start = start + i * iters_per_thread * inc;
        int thread_end = (i == num_threads - 1) ? end : thread_start + iters_per_thread * inc;
        ThreadArgs* thread_args = new ThreadArgs{i, thread_start, thread_end, inc, func, arg};
        pthread_create(&threads[i], NULL, threadFunc, thread_args);
    }
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }
}