#include<iostream>
#include<stdio.h>
#include<algorithm>
#include<vector>
#include<chrono>
#include<fstream>
#include<sstream>
#include<omp.h>

using namespace std;

static float INF = 1e+9;

/**
 * 读取邻接表和测试文件
 */
void read_data(int& n, vector<vector<float>>& dist, vector<pair<int, int>>& test, const string& adjacency_name, const string& test_name) {
    // 打开文件
    ifstream adj_file(adjacency_name);
    ifstream test_file(test_name);
    if (!adj_file.is_open() || !test_file.is_open()) {
        cerr << "Failed to open file." << endl;
        exit(1);
    }

    // 读取顶点总数
    adj_file >> n;
    n += 1;
    adj_file.ignore(); // 跳过换行

    // 初始化dist数组
    dist.resize(n , vector<float> (n, INF));
    for (int i = 0; i < n; ++i) {
        dist[i][i] = 0;
    }

    int i, j;
    float w;
    char comma;

    // 读取每一条边
    string line;
    while (getline(adj_file, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        iss >> i >> comma >> j >> comma >> w;
        dist[i][j] = w;
        dist[j][i] = w;
    }
    adj_file.close();

    // 读取测试文件
    while(getline(test_file, line)) {
        if(line.empty()) continue;
        istringstream iss(line);
        iss >> i >> comma >> j;
        test.push_back({i, j});
    }

    test_file.close();
}

void mode1(int n, vector<vector<float>>& dist) {
    for(int k = 0; k < n; ++k) {
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < n; ++i) {
            if(dist[i][k] >= INF) {
                continue;
            }
            for(int j = 0; j < n; ++j) {
                dist[i][j] = min({INF, dist[i][j], dist[i][k] + dist[k][j]});
            }
        }
    }
}

void mode2(int n, vector<vector<float>>& dist) {
    for(int k = 0; k < n; ++k) {
        for(int i = 0; i < n; ++i) {
            if(dist[i][k] >= INF) {
                continue;
            }
            #pragma omp parallel for schedule(static)
            for(int j = 0; j < n; ++j) {
                dist[i][j] = min({INF, dist[i][j], dist[i][k] + dist[k][j]});
            }
        }
    }
}

void floyd(int n, vector<vector<float>>& dist, int mode) {
    switch (mode)
    {
    case 1:
        mode1(n, dist);
        break;
    case 2:
        mode2(n, dist);
        break;
    }
}

int main(int argc, char* argv[]) {
    
    int n;
    vector<vector<float>> dist;
    vector<pair<int, int>> test;
    
    string adjacency_file = "updated_mouse.csv";
    string test_file = "test.txt";
    
    int thread_cnt = 4;
    int mode = 1;

    if(argc > 1) {
        thread_cnt = stoi(argv[1]);
    }
    if(argc > 2) {
        adjacency_file = argv[2];
    }
    if(argc > 3) {
        mode = stoi(argv[3]);
    }
    freopen("result.txt", "a+", stdout);

    cout << "thread num: " << thread_cnt << endl;
    cout << "parallel mode: " << mode << endl;
    cout << "data file: " << adjacency_file << endl;
    
    read_data(n, dist, test, adjacency_file, test_file);
    
    omp_set_num_threads(thread_cnt);
    auto begin = chrono::high_resolution_clock::now();
    floyd(n, dist, mode);
    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);
    
    cout << "running time: " << elapsed.count() * 1e-6 << " ms" << endl << endl;
    
    freopen("/dev/tty", "a", stdout);
    cout << "==============distance resutl===============" << endl;
    for(auto& t : test) {
        cout << t.first << ',' << t.second << ": ";
        cout << dist[t.first][t.second] << endl;
    }
    cout << endl;

    return 0;
}