#include <cassert>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>
#include <filesystem>

#include <omp.h>

#include "multicore_bfs.hpp"

void print(const std::vector<int>& arr) {
    int j = 0;
    for(const auto&i : arr) {
        std::cout << j++ <<" : " << i << std::endl;
    }
    std::cout << std::endl;
}

void read_edge(const std::string& filename, int& n, long& m, int*& srcs, int*& dsts) {

    std::ifstream infile(filename);
    if(!infile) {
        std::cerr <<"Unable to open file.\n";
        exit(EXIT_FAILURE);
    }

    int src, dst;
    long counter = 0;
    infile >> n >> m;

    srcs = new int[m];
    dsts = new int[m];

    for (long i = 0; i < m; ++i) {
        infile >> src >> dst;
        srcs[counter] = src;
        dsts[counter] = dst;
        ++counter;
    }
    assert(counter == m);

    return;
}

void create_csr(const int n, const long m, const int* srcs, const int* dsts, int*& out_array, long*& out_degree_list) {
    
    out_degree_list = new long[n + 1]{};
    out_array = new int[m];

    // Count edges per node
    for (long i = 0; i < m; ++i) {
        ++out_degree_list[srcs[i] + 1];
    }

    // Cumulative sum for row pointers
    for (int i = 1; i <= n; ++i) {
        out_degree_list[i] += out_degree_list[i - 1];
    }

    // Temporary array to help in filling out_array
    int* tempRowPtr = new int[n + 1];
    std::copy(out_degree_list, out_degree_list + n + 1, tempRowPtr);

    // Fill column indices array
    for (long i = 0; i < m; ++i) {
        int src = srcs[i];
        int dst = dsts[i];
        int index = tempRowPtr[src]++;
        out_array[index] = dst;
    }

    delete[] tempRowPtr;
}

bool verify_spanning_tree(int num_ver, int root, int* parent) {
    int* comp_num = new int[num_ver];
    int num_comp = num_ver;

    std::vector<std::vector<int>> vertices_in_comp(num_ver);

    for (int i = 0; i < num_ver; ++i) {
        comp_num[i] = i;
        vertices_in_comp[i].push_back(i);
    }

    for (int i = 0; i < num_ver; ++i) {
        if (i != root) {
            if (comp_num[i] != comp_num[parent[i]]) {
                int u = comp_num[i];
                int v = comp_num[parent[i]];
                int small, big;
                if (vertices_in_comp[u].size() < vertices_in_comp[v].size()) {
                    small = u;
                    big = v;
                }
                else {
                    small = v;
                    big = u;
                }

                for (int k = 0; k < vertices_in_comp[small].size(); ++k) {
                    int ver = vertices_in_comp[small][k];
                    comp_num[ver] = big;
                    vertices_in_comp[big].push_back(ver);
                }

                vertices_in_comp[small].clear();
                --num_comp;
            }
        }
    }

    delete[] comp_num;

    if (num_comp == 1) 
        return true;
    else 
    return false;
}

void print_CSR(const long* vertices, const int* edges, int numVertices) {

    for (int i = 0; i < numVertices; ++i) {
        std::cout << "Vertex " << i << " is connected to: ";
        for (int j = vertices[i]; j < vertices[i + 1]; ++j) {
            std::cout << edges[j] << " ";
        }
        std::cout << "\n";
    }
}

std::string get_file_extension(std::string filename) {

    std::filesystem::path file_path(filename);

    // Extracting filename with extension
    filename = file_path.filename().string();
    // std::cout << "Filename with extension: " << filename << std::endl;

    // Extracting filename without extension
    std::string filename_without_extension = file_path.stem().string();
    // std::cout << "Filename without extension: " << filename_without_extension << std::endl;

    return filename_without_extension;
}

int main(int argc, char* argv[]) {
    if(argc < 2) {
        std::cerr <<"Usage: " << argv[0] << " <filename>\n";
        exit(EXIT_FAILURE);
    }
    std::string filename = argv[1];
    int n;
    long m;
    int* srcs = NULL;
    int* dsts = NULL;
    int* out_array = NULL;
    long* out_degree_list = NULL;

    std::cout <<"\n\nReading graph " << get_file_extension(filename) <<" \n";
    read_edge(filename, n, m, srcs, dsts);
    std::cout <<"Reading graph successful.\n";
    std::cout <<"Creating csr\n";
    auto start = std::chrono::high_resolution_clock::now();
    create_csr(n, m, srcs, dsts, out_array, out_degree_list);
    std::cout <<"CSR creation successful.\n";
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout <<"csr creation took: " << dur <<" ms.\n";

    std::vector<int> parents(n);
    std::vector<int> levels(n);

    std::cout << "Doing MULTICORE-BFS init" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < n; ++i)
            levels[i] = -1;
        #pragma omp for nowait
        for (int i = 0; i < n; ++i)
            parents[i] = -1;
    }
    
    end = std::chrono::high_resolution_clock::now();
    dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout <<"multicore_bfs init took: " << dur <<" ms.\n";

    bool random_start = true;
    int root = -1;
    if (random_start)
        root = rand() % n;
    start = std::chrono::high_resolution_clock::now();
    multicore_bfs(n, m/2, out_array, out_degree_list, root, parents.data(), levels.data());
    end = std::chrono::high_resolution_clock::now();
    dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout <<"multicore_bfs took: " << dur <<" ms.\n";
    #ifdef DEBUG
        print_CSR(out_degree_list, out_array, n);
        std::cout <<"Parent array:\n";
        print(parents);
        std::cout <<"Level array:\n";
        print(levels);
    #endif

    if(verify_spanning_tree(n, root, parents.data())) 
        std::cout << "\n\nspanning tree is verified\n";
    else 
        std::cout << "The parent array will not represent spanning tree\n";

    return EXIT_SUCCESS;
}