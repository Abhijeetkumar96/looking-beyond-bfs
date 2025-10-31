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

#include "rootedSpanningTreePR.h"
#include "utility.h"

// #define DEBUG

std::string get_filename(std::filesystem::path file_path) {
	// Extracting filename with extension
    std::string filename = file_path.filename().stem().string();
    // std::cout << "Filename with extension: " << filename << std::endl;
    return filename;
}

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
		return EXIT_FAILURE;
	}
	cudaSetDevice(2);
	std::string filename = argv[1];
	std::ifstream inputFile(filename);
	if (!inputFile)
	{
		std::cerr << "Unable to open the file for reading.\n";
		return EXIT_FAILURE;
	}

	int n, e;
	inputFile >> n >> e;
	int u, v;

	std::vector<int> u_arr;
	std::vector<int> v_arr;
	
	for (int i = 0; i < e; ++i)
	{
		inputFile >> u >> v;
		// if(u < v) {
			u_arr.push_back(u);
			v_arr.push_back(v);
		// }
	}

	int numEdges = u_arr.size();
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