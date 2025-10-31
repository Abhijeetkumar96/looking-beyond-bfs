#include <iostream>
#include <vector>

int numVert, numEdges;
std::vector<int> u_arr, v_arr;

// Convert CSR (vertices, edges) -> COO (u_arr, v_arr) for an undirected graph
// Keeps only (u,v) with u < v to store each undirected edge once.
static void csr_to_coo(const std::vector<long>& vertices,
                       const std::vector<int>& edges,
                       std::vector<int>& u_out,
                       std::vector<int>& v_out) {
    u_out.clear();
    v_out.clear();
    u_out.reserve(edges.size() / 2);
    v_out.reserve(edges.size() / 2);

    const int n = static_cast<int>(vertices.size()) - 1;
    for (int i = 0; i < n; ++i) {
        for (long j = vertices[i]; j < vertices[i + 1]; ++j) {
            int nbr = edges[j];
            if (i < nbr) {          // keep one direction
                u_out.push_back(i);
                v_out.push_back(nbr);
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
    csr_to_coo(vertices, edges, u_arr, v_arr);
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string filename = argv[1];
    readECLgraph(filename);

    std::cout << "No. of vertices = " << numVert << std::endl;
    std::cout << "No. of edges = " << numEdges << std::endl;

    std::cout << "Computing rooted spanning tree...\n";
    std::vector<int> parent = RootedSpanningTree(u_arr, v_arr, numVert);
    std::cout << "Rooted spanning tree computation completed\n";

#ifdef DEBUG
    // printArr(parent, numVert, 10); // if you have this helper, use numVert (not n)
#endif
    return EXIT_SUCCESS;
}
