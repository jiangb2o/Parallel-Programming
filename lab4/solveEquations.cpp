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

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<> distrib(-1.0, 1.0);

inline double static getADouble() {
    return distrib(gen) * 100;
}

struct EquationData {
    double a, b, c;
    double b_square, ac4, b_square_sub_ac4, sqrt, a2;
    int count1, count2;
};

// 使用cond_1时需要的锁
pthread_mutex_t mutex_1 = PTHREAD_MUTEX_INITIALIZER;
// 使用cond_2时需要的锁
pthread_mutex_t mutex_2 = PTHREAD_MUTEX_INITIALIZER;
// 
pthread_cond_t cond_1 = PTHREAD_COND_INITIALIZER;
pthread_cond_t cond_2 = PTHREAD_COND_INITIALIZER;

void* compute_b_square(void* para) {
    EquationData* data = static_cast<EquationData*>(para);
    data->b_square = data->b * data->b;
    pthread_mutex_lock(&mutex_1);
    data->count1++;
    pthread_cond_signal(&cond_1);
    pthread_mutex_unlock(&mutex_1);
    return nullptr;
}

void* compute_ac4(void* para) {
    EquationData* data = static_cast<EquationData*>(para);
    data->ac4 = 4 * data->a * data->c;
    pthread_mutex_lock(&mutex_1);
    data->count1++;
    pthread_cond_signal(&cond_1);
    pthread_mutex_unlock(&mutex_1);
    return nullptr;
}

void* compute_sqrt(void* para) {
    EquationData* data = static_cast<EquationData*>(para);
    data->sqrt = sqrt(data->b_square_sub_ac4);
    pthread_mutex_lock(&mutex_2);
    data->count2++;
    pthread_cond_signal(&cond_2);
    pthread_mutex_unlock(&mutex_2);
    return nullptr;
}

void* compute_a2(void* para) {
    EquationData* data = static_cast<EquationData*>(para);
    data->a2 =  2 * data->a;
    pthread_mutex_lock(&mutex_2);
    data->count2++;
    pthread_cond_signal(&cond_2);
    pthread_mutex_unlock(&mutex_2);
    return nullptr;
}


int main(int argc, char* argv[]) {
    EquationData para {
        .a = getADouble(),
        .b = getADouble(),
        .c = getADouble(),
        .count1 = 0,
        .count2 = 0
    };

    printf("a: %.4f, b: %.4f, c: %.4f\n", para.a, para.b, para.c);

    pthread_t* threads = new pthread_t[4];

    auto begin = chrono::high_resolution_clock::now();
    
    // 创建线程
    pthread_create(&threads[0], NULL, compute_b_square, static_cast<void*>(&para));
    pthread_create(&threads[1], NULL, compute_ac4, static_cast<void*>(&para));

    // 等待 b^2 和 4ac 计算完毕  
    pthread_mutex_lock(&mutex_1);
    // 当某一个线程计算完毕时, 主线程被唤醒, 但此时count1<2, 再次进入阻塞状态
    // 知道晚计算出结果的线程计算完毕, 主线程被再次唤醒, 此时count1 = 2, 可以继续执行  
    while(para.count1 < 2) {
        pthread_cond_wait(&cond_1, &mutex_1);
    }
    pthread_mutex_unlock(&mutex_1);

    para.b_square_sub_ac4 = para.b_square - para.ac4;
    if(para.b_square_sub_ac4 < 0) {
        printf("No real roots\n");
        return 0;
    }

    pthread_create(&threads[2], NULL, compute_sqrt, static_cast<void*>(&para));
    pthread_create(&threads[3], NULL, compute_a2, static_cast<void*>(&para));

    // 等待 sqrt 和 2a 计算完毕  
    pthread_mutex_lock(&mutex_2);
    while(para.count2 < 2) {
        pthread_cond_wait(&cond_2, &mutex_2);
    }
    pthread_mutex_unlock(&mutex_2);

    double x1 = (-para.b + para.sqrt) / para.a2;
    double x2 = (-para.b - para.sqrt) / para.a2;

    // 回收线程
    for(int i = 0; i < 4; ++i) {
        pthread_join(threads[i], NULL);
    }
    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);

    printf("x1 = %.4f\nx2 = %.4f\n", x1, x2);
    printf("parallel running time: %.4f s\n", elapsed * 1e-9);

    pthread_mutex_destroy(&mutex_1);
    pthread_mutex_destroy(&mutex_2);
    pthread_cond_destroy(&cond_1);
    pthread_cond_destroy(&cond_2);

    // 串行
    begin = chrono::high_resolution_clock::now();
    para.b_square = para.b * para.b;
    para.ac4 = 4 * para.a * para.c;
    para.b_square_sub_ac4 = para.b_square - para.ac4;
    para.sqrt = sqrt(para.b_square_sub_ac4);
    para.a2 = 2 * para.a;
    x1 = (-para.b + para.sqrt) / para.a2;
    x2 = (-para.b - para.sqrt) / para.a2;

    end = chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);
    printf("x1 = %.4f\nx2 = %.4f\n", x1, x2);
    printf("serial running time: %.4f s\n", elapsed * 1e-9);

    return 0;
}
