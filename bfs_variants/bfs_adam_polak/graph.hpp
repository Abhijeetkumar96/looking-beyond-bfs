/**
 * @file graph.hpp
 * @brief Graph data structure implementation for handling large graphs in CSR and edge list formats.
 * 
 * This implementation supports the following features:
 * - Reading graph data from .txt and .egr files into both edge list and CSR formats.
 * - Conversion between CSR format and edge list to facilitate different graph operations.
 * - The edge list representation includes the undirected edge in both directions
 * (e.g., for (u, v), (v, u)) is also stored.
 * - If only single orientation is needed, then do the following changes:
 *  if(u < v)
 *      then push_back
 * 
 * Usage:
 * To use this class, create an instance with the path to a graph file. The class automatically
 * reads the graph data based on the file extension:
 * - .txt: Assumes an adjacency list format.
 * - .egr: Assumes a binary CSR format.
 * 
 * The class provides functionality to print the graph in CSR format and internally handles conversions
 * between formats as necessary.
 * 
 * Example:
 *   graph g("path/to/graph.txt");
 *   g.print_CSR(); // Outputs the CSR representation of the graph.
 * 
 * @warning This implementation does not explicitly handle parallel edges and self-loops in input data; 
 *          these should be managed according to the specific needs of the application using this graph class.
 */

#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cassert>
#include <queue>
#include <random>
#include <algorithm>

class graph {
public:
    // csr ds
    std::vector<long> vertices;
    std::vector<int> edges;

    // edge_list
    std::vector<int> u_arr;
    std::vector<int> v_arr;
    
    int numVert = 0;
    long numEdges = 0;

    graph(const std::string& filename) {
        std::string extension = getFileExtension(filename);
        if (extension == ".txt") {
            readEdgesgraph(filename);
        } else if (extension == ".egr") {
            readECLgraph(filename);
        } else {
            std::cerr << "Unsupported file extension: " << extension << std::endl;
            return;
        }
    }

    void print_CSR() {
        std::cout << "CSR for graph G:\n";
        for (int i = 0; i < numVert; ++i) {
            std::cout << "Vertex " << i << " is connected to: ";
            for (int j = vertices[i]; j < vertices[i + 1]; ++j) {
                std::cout << edges[j] << " ";
            }
            std::cout << "\n";
        }
    }

private:
    void readECLgraph(const std::string& filepath) {
        std::ifstream inFile(filepath, std::ios::binary);
        if (!inFile) {
            throw std::runtime_error("Error opening file: " + filepath);
        }

        // Reading sizes
        size_t size;
        inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
        vertices.resize(size);
        inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
        edges.resize(size);

        // Reading data
        inFile.read(reinterpret_cast<char*>(vertices.data()), vertices.size() * sizeof(long));
        inFile.read(reinterpret_cast<char*>(edges.data()), edges.size() * sizeof(int));

        numVert = vertices.size() - 1;
        numEdges = edges.size();

        csrToList();
    }

    void readEdgesgraph(const std::string& filepath) {
        std::ifstream inFile(filepath);
        if (!inFile) {
            throw std::runtime_error("Error opening file: " + filepath);
        }
        inFile >> numVert >> numEdges;

        std::vector<std::vector<int>> adjlist(numVert);
        u_arr.reserve(numEdges);
        v_arr.reserve(numEdges);
        
        uint32_t u, v;
        for(long i = 0; i < numEdges; ++i) {
            inFile >> u >> v;
            adjlist[u].push_back(v);
                u_arr.push_back(u);
                v_arr.push_back(v);
        }

        createCSR(adjlist);  
    }

    void createCSR(const std::vector<std::vector<int>>& adjlist) {
    
        int numVert = adjlist.size();

        vertices.push_back(edges.size());
        for (int i = 0; i < numVert; i++) {
            edges.insert(edges.end(), adjlist[i].begin(), adjlist[i].end());
            vertices.push_back(edges.size());
        }
    }

    void csrToList() {

        u_arr.reserve(numEdges);
        v_arr.reserve(numEdges);
        long ctr = 0;

        for (int i = 0; i < numVert; ++i) {
            for (long j = vertices[i]; j < vertices[i + 1]; ++j) {
                if(i < edges[j]) {
                    uint32_t x = i;
                    uint32_t y = edges[j];
                    u_arr.push_back(x);
                    v_arr.push_back(y);
                    ctr++;
                }
            }
        }    

        assert(ctr == numEdges);
    }

    std::string getFileExtension(const std::string& filename) {
        auto pos = filename.find_last_of(".");
        if (pos != std::string::npos) {
            return filename.substr(pos);
        }
        return "";
    }
};

#endif // GRAPH_H
