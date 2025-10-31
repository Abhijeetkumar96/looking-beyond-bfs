#ifndef __GRAPH__
#define __GRAPH__

#include<fstream>
#include <vector>

using namespace std;

typedef struct edge
{
	int u, v;
}edge;

typedef struct query
{
	int u, v;
}query;

class unweightedGraph
{
public:
	int totalVertices, totalEdges, * offset, * neighbour, * U, * V, root, * degree;
	std::vector<std::vector<int>> adjlist;

public:
	// Graph();
	unweightedGraph(ifstream& edgeList);
	void printCSR();
};

#define degree(G, n) (G.offset[n+1] - G.offset[n])

#endif