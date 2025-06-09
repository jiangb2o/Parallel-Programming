#include<iostream>
#include<random>
#include<chrono>

#include"parallel_for.h"

using namespace std;

struct FunctorArgs {
    int* A;
    int* B;
    int* C;
};

void getRandomValue(int* const arr, int size) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(1, 10);
    for(int i = 0; i < size; ++i) {
        arr[i] = distrib(gen);
    }
}

void printArray(int* arr, int size) {
    cout << '[';
    for(int i = 0; i < size; ++i) {
        cout << arr[i] << ',';
    }
    cout << ']' << endl;
}

void* functor(int index, void* arg) {
    FunctorArgs* args = static_cast<FunctorArgs*>(arg);
    args->C[index] = args->A[index] + args->B[index];
    return nullptr;
}

int main(void) {

    int* A = new int[10];
    int* B = new int[10];
    int* C = new int[10]();
    getRandomValue(A, 10);
    getRandomValue(B, 10);

    printArray(A, 10);
    printArray(B, 10);

    FunctorArgs* arg = new FunctorArgs{A, B, C};

    auto begin = chrono::high_resolution_clock::now();
    parallel_for(0, 10, 1, functor, arg, 2);
    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);
    printArray(C, 10);

    return 0;
}