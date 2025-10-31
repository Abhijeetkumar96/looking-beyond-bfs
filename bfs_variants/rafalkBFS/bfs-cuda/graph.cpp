#include <ctime>
#include <iostream>
#include <fstream>
#include "graph.h"

void readGraph(Graph &G, int argc, char *argv[]) {
    int n;
    int m;

    //If no arguments then read graph from stdin
	std::string filename = argv[1];
	std::ifstream inputFile(filename);
	if(!inputFile)
		std::cerr <<"Unable to open file "<< std::endl;
	inputFile >> n >> m;

    std::vector<std::vector<int> > adjecancyLists(n);
    for (int i = 0; i < m; i++) {
        int u, v;
	inputFile >> u >> v;
	if(u == v)
		continue;
	else {
            adjecancyLists[u].push_back(v);
	    //adjecancyLists[v].push_back(u);
	    }
        } 

    for (int i = 0; i < n; i++) {
        G.edgesOffset.push_back(G.adjacencyList.size());
        G.edgesSize.push_back(adjecancyLists[i].size());
        for (auto &edge: adjecancyLists[i]) {
            G.adjacencyList.push_back(edge);
        }
    }

    G.numVertices = n;
    G.numEdges = G.adjacencyList.size();
}
