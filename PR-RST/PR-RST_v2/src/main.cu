#include <iostream>
#include <fstream>
#include <vector>
#include <cuda.h>
#include <algorithm>
#include <random>
#include <chrono>
#include <utility>
#include <filesystem>

#include "rootedSpanningTreePR.h"
#include "utility.h"

// #define DEBUG

int numVert, numEdges;
std::vector<int> u_arr, v_arr;

static void csr_to_coo(const std::vector<long>& vertices,
                       const std::vector<int>& edges) {

    u_arr.reserve(edges.size() / 2);
    v_arr.reserve(edges.size() / 2);

    const int n = static_cast<int>(vertices.size()) - 1;
    for (int i = 0; i < n; ++i) {
        for (long j = vertices[i]; j < vertices[i + 1]; ++j) {
            int nbr = edges[j];
            if (i < nbr) {          // keep one direction
                u_arr.push_back(i);
                v_arr.push_back(nbr);
            }
        }
    }
}

void readECLgraph(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    size_t size = 0;
    std::vector<long> vertices;
    std::vector<int>  edges;

    inFile.read(reinterpret_cast<char*>(&size), sizeof(size));   // # of vertices entries
    vertices.resize(size);
    inFile.read(reinterpret_cast<char*>(&size), sizeof(size));   // # of edges entries
    edges.resize(size);

    inFile.read(reinterpret_cast<char*>(vertices.data()),
                vertices.size() * sizeof(long));
    inFile.read(reinterpret_cast<char*>(edges.data()),
                edges.size() * sizeof(int));

    numVert = static_cast<int>(vertices.size()) - 1;
    numEdges = static_cast<int>(edges.size());

    // Build COO once we have CSR
    csr_to_coo(vertices, edges);
}

void read_edgelist(std::string filename) {
	std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    inFile >> numVert >> numEdges;
    u_arr.resize(numEdges);
    v_arr.resize(numEdges);

    int u, v;

    for(long i = 0; i < numEdges; ++i) {
    	inFile >> u >> v;
    	if(u < v) {
            u_arr.push_back(u);
            v_arr.push_back(v);
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string filename = argv[1];
    std::filesystem::path file_path(filename);
    std::string ext = file_path.extension().string();

    try {
        if (ext == ".txt" || ext == ".edges" || ext == ".el") {
            std::cout << "Detected edge list format (" << ext << ").\n";
            read_edgelist(filename);
        } 
        else if (ext == ".graph" || ext == ".bin" || ext == ".egr") {
            std::cout << "Detected ECL graph format (" << ext << ").\n";
            readECLgraph(filename);
        } 
        else {
            std::cerr << "Unknown file extension '" << ext 
                      << "'. Expected .txt/.edges/.el for edge list or .graph/.bin/.egr for ECL format.\n";
            return EXIT_FAILURE;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "No. of vertices = " << numVert << std::endl;
    std::cout << "No. of edges = " << numEdges << std::endl;

	size_t size = numEdges * sizeof(int);

	int* d_u_arr = nullptr;
	int* d_v_arr = nullptr;

	cudaMalloc((void **)&d_u_arr, size);
	cudaMalloc((void **)&d_v_arr, size);

	cudaMemcpy(d_u_arr, u_arr.data(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_v_arr, v_arr.data(), size, cudaMemcpyHostToDevice);
	std::cout << "Filename: " << get_filename(filename) << "\n";
	RootedSpanningTree(d_u_arr, d_v_arr, n, e);
}