#include <iostream>
#include <fstream>
#include <vector>
#include <cuda.h>
#include <algorithm>
#include <random>
#include <chrono>
#include <utility>
#include <filesystem>
#include <thrust/device_vector.h>

#include "pr_rst/rootedSpanningTreePR.h"
#include "verification/verifySol.h"
#include "cc/connected_comp.h"
#include "util/utility.h"

// Global variables needed for the helper functions
std::filesystem::path filepath;

int numVert, numEdges;
std::vector<int> u_arr, v_arr;

void csr_to_coo() {
    size_t bytes = (numEdges / 2) * sizeof(uint64_t);

    for (int i = 0; i < numVert; ++i) {
        for (long j = vertices[i]; j < vertices[i + 1]; ++j) {
            if (i < edges[j]) {
                u_arr.push_back(i);
                v_arr.push_back(edges[j]);
            }
        }
    }
}

void readEdgeList() {
    std::ifstream inFile(filepath);
    if (!inFile) {
        throw std::runtime_error("Error opening file: " + filepath.string());
    }
    inFile >> numVert >> numEdges;

    size_t bytes = (numEdges / 2) * sizeof(uint64_t);
	u_arr.resize(numEdges / 2);
	v_arr.resize(numEdges / 2);
    long ctr = 0;
    int u, v;
    for (long i = 0; i < numEdges; ++i) {
        inFile >> u >> v;
        if (u < v) {
        	u_arr[ctr] = u;
        	v_arr[ctr] = v;
            ctr++;
        }
    }
    assert(ctr == numEdges / 2);
}   

void readECLgraph(std::string filename) {

    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        throw std::runtime_error("Error opening file: " + filename);
    }
    size_t size;
	
	std::vector<long> vertices;
	std::vector<int> edges;

    inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
    vertices.resize(size);
    inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
    edges.resize(size);

    inFile.read(reinterpret_cast<char*>(vertices.data()), vertices.size() * sizeof(long));
    inFile.read(reinterpret_cast<char*>(edges.data()), edges.size() * sizeof(int));

    numVert = vertices.size() - 1;
    numEdges = edges.size();

    csr_to_coo();
}

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
		return EXIT_FAILURE;
	}

	std::string filename = argv[1];
	readECLgraph(filename);
	
	std::cout << "No. of vertices = " << numVert << std::endl;
	std::cout << "No. of edges = " << numEdges << std::endl;

	// Compute rooted spanning tree
	std::cout << "Computing rooted spanning tree...\n";
	std::vector<int> parent = RootedSpanningTree(u_arr, v_arr, numVert);
	
	std::cout << "Rooted spanning tree computation completed\n";

	#ifdef DEBUG
		printArr(parent, n, 10);
	#endif

	return EXIT_SUCCESS;
}