#pragma once

void parallel_for(int start, int end, int inc, void* (*func)(int, void*), void* arg, int num_threads);