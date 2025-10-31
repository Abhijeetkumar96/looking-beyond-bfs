#include <sstream>
#include <string>
#include <set>
#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <ctime>
#include <chrono>
#include <random>
#include <cmath>

#include <cuda_runtime.h>
#include <cuda.h>

#define FALSE 0
#define TRUE 1
#define INF INT_MAX

void serialbfs(int src, const std::vector<std::set<int>>& adjlist, std::vector<int>& dist_serial) {
    int n = adjlist.size();
    std::vector<bool> visited_serial(n, false);
    std::queue<int> q;

    // Initialize distances
    dist_serial = std::vector<int>(n, INT_MAX); // Use INT_MAX to represent infinity
    dist_serial[src] = 0;

    q.push(src);
    visited_serial[src] = true;

    while (!q.empty()) {
        int parent = q.front();
        q.pop();

        for (int neighbor : adjlist[parent]) {
            if (!visited_serial[neighbor]) {
                if (dist_serial[neighbor] > dist_serial[parent] + 1) {
                    dist_serial[neighbor] = dist_serial[parent] + 1;
                }
                q.push(neighbor);
                visited_serial[neighbor] = true;
            }
        }
    }
}

bool verify(const std::vector<int>& arr1, const std::vector<int>& arr2)
{
  bool flag = 1;
  if(arr1.size() == arr2.size())
    std::cout<<"Step 1, i.e. verifying of size is complete : \n";
  else
    std::cout<<"Size unequal : ";
  for(int i = 0; i<arr1.size(); ++i)
  {
    if(arr1[i] != arr2[i])
    {
      #ifdef DEBUG
        std::cout<<"\nVALUE AT "<<i<<"IS DIFFERENT FOR BOTH THE ARRAYS\n";
        std::cout << "Distance of parallel[" << i << "] = " << arr1[i] << " whereas distance of serial[" << i << "] = " << arr2[i] << std::endl;
      #endif
      flag = 0;
      return flag;
    }

  }
  std::cout<<"\nCongratulatiions.. The bfs results are correct..\n";

  return flag;
}

__global__ 
void BFS_KERNEL(int n, int *c_iteration_no, int *c_edgelist, int *c_csr_edge_range, int *c_dist, int *c_parent,int *c_visited, int* c_flag)
{
  // CUDA kernel implementation
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < n and c_visited[tid] == *c_iteration_no)
  {
    int vertex_no = tid;
    
    int start = c_csr_edge_range[vertex_no];
    int end = c_csr_edge_range[vertex_no+1];

    for(int j = start; j<end; ++j)
    {
      if(c_dist[c_edgelist[j]] > *c_iteration_no + 1)
      {
        int k = c_edgelist[j];
        c_dist[k] = *c_iteration_no+1;
        c_parent[k] = vertex_no;
        c_visited[k] = *c_iteration_no+1;
        *c_flag = 1;
      }
    }
  }
}

void BFS(const int s, const int n, const std::vector<std::set<int>> &adjlist, std::vector<int>& dist)
{
    // Your BFS implementation
    std::vector<int> edgelist;
    std::vector<int> csr_edge_range(n+1); 
    std::vector<int> parent(n); 
    std::vector<int> visited(n, -1);
    int iteration_no;

    auto csr_start = std::chrono::high_resolution_clock::now();
    // Build the CSR (Compressed Sparse Row)
    csr_edge_range[0] = 0;
    for (int i = 0; i < adjlist.size(); ++i) {
        csr_edge_range[i + 1] = csr_edge_range[i] + adjlist[i].size();
        for (auto adjnode : adjlist[i]) {
            edgelist.push_back(adjnode);
        }
    }

    auto csr_end = std::chrono::high_resolution_clock::now();
    auto csr_duration = std::chrono::duration_cast<std::chrono::milliseconds> (csr_end - csr_start).count();

    std::cout << "CSR preparation took : " << csr_duration <<" milliseconds\n.";

    //Updation of Source distance and iteration number
    visited[s] = 0;
    dist[s] = 0;
    iteration_no = 0;

    //CUDA Variable initialization
    int *c_edgelist, *c_csr_edge_range, *c_visited, *c_dist, *c_parent, *c_iteration_no;

    auto new_start = std::chrono::high_resolution_clock::now();

    cudaMalloc((void**)&c_edgelist, sizeof(int)*(edgelist.size()));
    cudaMalloc((void**)&c_csr_edge_range, sizeof(int)*(csr_edge_range.size()));
    cudaMalloc((void**)&c_visited, sizeof(int) * n);
    cudaMalloc((void**)&c_dist, sizeof(int) * n);
    cudaMalloc((void**)&c_parent, sizeof(int) * n);

    // Unified Memory for the flag
    int *flag;
    cudaMallocManaged(&flag, sizeof(int));

    // Initialize the flag
    *flag = 1;

    cudaMalloc((void**)&c_iteration_no, sizeof(int));

    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device); // get current device
    cudaGetDeviceProperties(&prop, device); // get the properties of the device

    int maxThreadsPerBlock = prop.maxThreadsPerBlock; // max threads that can be spawned per block

    // calculate the optimal number of threads and blocks
    int threadsPerBlock = (n < maxThreadsPerBlock) ? n : maxThreadsPerBlock;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaMemcpy(c_edgelist, edgelist.data(), sizeof(int)*(edgelist.size()), cudaMemcpyHostToDevice);
    cudaMemcpy(c_csr_edge_range, csr_edge_range.data(), sizeof(int)*(csr_edge_range.size()), cudaMemcpyHostToDevice);
    cudaMemcpy(c_visited, visited.data(), sizeof(int)*(visited.size()), cudaMemcpyHostToDevice);
    cudaMemcpy(c_dist, dist.data(), sizeof(int)*(dist.size()), cudaMemcpyHostToDevice);
    cudaMemcpy(c_parent, parent.data(), sizeof(int)*(parent.size()), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    while (*flag) {
      
      *flag = 0;
      cudaMemcpy(c_iteration_no, &iteration_no, sizeof(int), cudaMemcpyHostToDevice);

      BFS_KERNEL<<<blocksPerGrid, threadsPerBlock>>> 
      (n, c_iteration_no, c_edgelist, c_csr_edge_range, c_dist, c_parent, c_visited, flag);
      cudaDeviceSynchronize();

      iteration_no++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Time for parallel bfs without copying the data: " << duration.count() << " milliseconds.\n";


    auto new_end = std::chrono::high_resolution_clock::now();
    auto new_duration = std::chrono::duration_cast<std::chrono::milliseconds>(new_end - new_start);

    std::cout << "Time for parallel bfs with copying the data: " << new_duration.count() << " milliseconds.\n";
    
    std::cout << "\nDepth of graph is : " << iteration_no + 1 << std::endl;
    std::cout << "log2" << n <<" = " <<log2(n) << std::endl;
    
    cudaMemcpy(dist.data(), c_dist, sizeof(int)*(dist.size()), cudaMemcpyDeviceToHost);
    cudaMemcpy(parent.data(), c_parent, sizeof(int)*(parent.size()), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    parent[s] = s;

    int n_visited = 0;
    for(auto i : parent)
    {
      if(i != -1)
        n_visited++;
    }

    std::cout <<" The number of nodes that got visited are : "<< n_visited << "and the actual num of nodes are "<< n << "." << std::endl;

    cudaFree(c_edgelist);
    cudaFree(c_csr_edge_range);
    cudaFree(c_visited);
    cudaFree(c_dist);
    cudaFree(c_parent);
    cudaFree(flag);
    cudaFree(c_iteration_no);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string filename = argv[1];
    std::ifstream inputFile(filename);

    if (!inputFile) {
        std::cerr << "Unable to locate file." << std::endl;
        return EXIT_FAILURE;
    }

    std::string line;
    int n, e;

    // Read the first non-comment line for n and e
    while (std::getline(inputFile, line)) {
      if (line[0] != '%') {
          std::istringstream iss(line);
          if (!(iss >> n >> e)) {
              std::cerr << "Error reading node and edge counts" << std::endl;
              return EXIT_FAILURE;
          }
          break;
      }
  }

  auto start = std::chrono::high_resolution_clock::now();
  std::vector<std::set<int>> adjlist(n);
  int u, v;

  // Read the edges
  for (int i = 0; i < e; ++i) {
    if (!(inputFile >> u >> v)) {
        std::cerr << "Error reading an edge" << std::endl;
        return EXIT_FAILURE;
    }
    u--; // Adjusting for 0-based indexing
    v--;
    adjlist[u].insert(v);
    adjlist[v].insert(u);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  
  std::cout << "Time for creating csr took : " << duration << " milliseconds.\n";  

  std::vector<int> dist(n, INT_MAX);

  BFS(0, n, adjlist, dist);

  std::vector<int> serial_dist(n, INT_MAX);
  auto new_start = std::chrono::high_resolution_clock::now();
  serialbfs(0, adjlist, serial_dist);
  auto new_end = std::chrono::high_resolution_clock::now();
  auto new_duration = std::chrono::duration_cast<std::chrono::microseconds>(new_end - new_start);

  std::cout << "Time for serial bfs : " << new_duration.count() << " microseconds.\n";

  std::cout<<verify(dist, serial_dist);

  std::time_t currentTime = std::time(nullptr);

  // Convert the current time to a string representation
  char* timeString = std::ctime(&currentTime);

  // Print the current time
  std::cout << "Current time: " << timeString;


  return 0;
}