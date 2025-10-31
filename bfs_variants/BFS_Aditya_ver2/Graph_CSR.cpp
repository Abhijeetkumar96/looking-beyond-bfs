#include"Graph_CSR.h"
#include<fstream>
#include<iostream>
#include<vector>
#include<set>

unweightedGraph::unweightedGraph(ifstream& edgeList)
{
	// cout << "reached in unweightedGraph" << endl;
	edgeList >> totalVertices >> totalEdges;
	offset = new int[totalVertices + 1];
	degree = new int[totalVertices]();
	neighbour = new int[totalEdges];

	// storing the directed edges and calculating the degree
	U = new int[totalEdges];
	V = new int[totalEdges];

	adjlist.resize(totalVertices);

	for (int i = 0; i < totalEdges; i++)
	{
		int u, v;
		edgeList >> u >> v;
		adjlist[u].push_back(v);

		degree[u]++;

		U[i] = u;
		V[i] = v;
	}

	// updating the offset array
	vector<int> edgeCounter(totalVertices);
	offset[0] = 0;
	for (int i = 0; i < totalVertices; i++)
	{
		offset[i + 1] = degree[i] + offset[i];
		edgeCounter[i] = offset[i];
	}

	// updating the neighbour array
	for (int i = 0; i < totalEdges; i++)
	{
		int u, v;
		u = U[i];
		v = V[i];

		int currIndex = edgeCounter[u];
		edgeCounter[u]++;
		neighbour[currIndex] = v;
	}

	delete[] U;
	delete[] V;

	root = 0;
	int count = 0;
	int maxDegree = 0;
	U = new int[totalEdges / 2];
	V = new int[totalEdges / 2];
	for (int i = 0; i < totalVertices; i++)
	{
		int u = i;
		if (degree[i] > maxDegree)
		{
			maxDegree = degree[i];
			root = i;
		}
		for (int j = offset[i]; j < offset[i + 1]; j++)
		{
			int v = neighbour[j];
			if (u < v)
			{
				U[count] = u;
				V[count] = v;
				count++;
			}

		}
	}

}

void unweightedGraph::printCSR()
{
	cout << "Total Edges = " << totalEdges << endl;
	cout << "Total Vertices = " << totalVertices << endl;
	for (int i = 0; i < totalVertices; i++)
	{
		int u = i;
		cout << "Vertex " << u << " -> { ";
		for (int j = offset[i]; j < offset[i + 1]; j++)
		{
			int v = neighbour[j];
			cout << "( " << u << "," << v << ")";
			cout << ", ";
		}
		cout << "}" << endl;
	}
}