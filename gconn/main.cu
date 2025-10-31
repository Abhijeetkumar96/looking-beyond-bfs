// In this version I'm optimizing the finding of the largest CC part
// To Compile: nvcc -O3 -arch=sm_80 -o main main.cu -std=c++17

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>

#include "common.hxx"
#include "graph.hxx"
#include "rst.hxx"

// #define DEBUG

// -----------------------------------------------------------------------------
// Main function
// -----------------------------------------------------------------------------
int main(int argc, char const *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <graph filename>" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::string filename = argv[1];
    
    // Read (or create) a graph on the host.
    undirected_graph g(filename);
    graph_data d_input;

    int num_vert = g.getNumVertices();
    long num_edges = g.getNumEdges() / 2;

    d_input.V = num_vert;
    d_input.E = num_edges;

    std::cout << "Number of vertices: " << num_vert << "\n";
    std::cout << "Number of edges: " << num_edges << "\n";

    // Allocate device memory and copy the edge list to the device
    size_t bytes = (num_edges) * sizeof(uint64_t);
    CUDA_CHECK(cudaMalloc((void**)&d_input.edgelist, bytes), "Failed to allocate device memory for edgelist");
    CUDA_CHECK(cudaMemcpy(d_input.edgelist, g.getEdgelist(), bytes, cudaMemcpyHostToDevice), "Failed to copy edgelist to device");

    auto st_time = construct_st(d_input);
    std::cout << "GCONN took: " << st_time << " ms\n";
    return EXIT_SUCCESS;
}
