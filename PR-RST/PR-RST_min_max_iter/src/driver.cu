#include <iostream>
#include <fstream>
#include <vector>
#include <cuda.h>
#include <algorithm>
#include <random>
#include <chrono>
#include <utility>
#include <thrust/device_vector.h>

#include "pr_rst/rootedSpanningTreePR.h"
#include "verification/verifySol.h"
#include "cc/connected_comp.h"
#include "util/utility.h"

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
		return EXIT_FAILURE;
	}

	std::string filename = argv[1];
	std::ifstream inputFile(filename);
	if (!inputFile)
	{
		std::cerr << "Unable to open the file for reading.\n";
		return EXIT_FAILURE;
	}

	int n, e;
	inputFile >> n >> e;


	std::cout << "No. of vertices = " << n << std::endl;
	std::cout << "No. of edges = " << e << std::endl;

	int u, v;
	std::vector<int> u_arr;
	std::vector<int> v_arr;
	std::vector<std::vector<int>> adjlist(n);

	// std::vector<std::pair<int, int>> edge_stream;
	std::cout<<"Reading input\n";
	for (int i = 0; i < e; ++i)
	{
		inputFile >> u >> v;
		adjlist[u].push_back(v);
		u_arr.push_back(u);
		v_arr.push_back(v);
		// v_arr.push_back(u);
		// u_arr.push_back(v);
		// edge_stream.push_back(std::make_pair(v, u));
	}

	std::cout<<"Input reading done\n";

	// int comp_count = findConnected(adjlist, n);

	// std::cout<<"Number of connected components : "<<comp_count<<"\n";
	std::vector<int> parent = RootedSpanningTree(u_arr, v_arr, n);
	
	#ifdef DEBUG
		printArr(parent,n,10);
	#endif
	
	// if(validateRST(parent,comp_count))
	// {
	// 	std::cout<<"Validation success"<<std::endl;
	// }
	// else
	// {
	// 	std::cout<<"Validation failure"<<std::endl;
	// 	exit(1);
	// }
}