#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <cuda_runtime.h>

#include <cuda.h>
#include <cub/cub.cuh>

#include "graph.hpp"

#define DEBUG

inline void check_for_error(cudaError_t error, const std::string& message, const std::string& file, int line) noexcept {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << " - " << message << "\n";
        std::cerr << "CUDA Error description: " << cudaGetErrorString(error) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(err, msg) check_for_error(err, msg, __FILE__, __LINE__)

using namespace std;

__global__
void computesegments(long* d_nodes, int* d_edges, long d_m, long d_n, long d_nodefrontier_size , int* d_nodefrontier , long* d_segments) {

  long id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < d_nodefrontier_size) {
   
        int node = d_nodefrontier[id];
        long start = d_nodes[node];
        long end = (node == d_n - 1) ? d_m : d_nodes[node + 1];
        d_segments[id] = end - start;
    }
}

__global__
void computerank(long* d_rank , long* d_seg , long* d_segments , long d_edgefrontier_size) {
    long id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < d_edgefrontier_size) {
        d_rank[id] = id - d_segments[d_seg[id]];
    }
}

__global__
void computeedgefrontier(int* d_edgefrontier ,int* d_nodefrontier, long d_edgefrontier_size , long* d_rank , long* d_seg , long* d_nodes , int* d_edges) {
    long id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < d_edgefrontier_size){
        long seg = d_seg[id];
        long rank = d_rank[id];
        d_edgefrontier[id] = d_edges[d_nodes[d_nodefrontier[seg]] + rank];
    }
}


__global__
void markvisited(long d_edgefrontier_size , int level , int* d_edgefrontier , int* d_distance , long* d_seg) {
    long id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < d_edgefrontier_size){
        d_seg[id] = (-1 == atomicCAS(d_distance + d_edgefrontier[id], -1, level));
    }
}


__global__
void computenodefrontier(int* d_nodefrontier , int* d_edgefrontier , long* d_seg , long d_nodefrontier_size) {
    long id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < d_nodefrontier_size){
        d_nodefrontier[id] = d_edgefrontier[d_seg[id]];
    }
}


__global__ 
void custom_lbs(long *a, long n_a , long *b , long n_b) {
    long i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i< n_b) {
        long l = 1, r = n_a;
        while(l < r) {
            long mid = (l+r)/2;
            if(a[mid]<= i) {
                l = mid+1;
            } else {
                r = mid;
            }
        }
      
        b[i] = l-1;
    }
}

// excepts all directed edges
__global__ 
void update_parent(
    const int* row_indices, 
    const int* col_indices, 
    int* parent, 
    int* distance, 
    long M) {

    long index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < M) {
        int from = row_indices[index];
        int to = col_indices[index];
        if (distance[from] == distance[to] - 1) {
            parent[to] = from;
        }
    }
}

template <typename T1, typename T2>
__global__
void print_a(T1* arr, T2 n) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid == 0) {
        for(T2 i = 0; i < n; ++i) {
            printf("parent[%d]: %d\n", i, arr[i]);
        }
    }
}

void adam_polak_bfs(int n, long m, long* d_nodes, int* d_edges, int* u, int* v) {
  
    int* d_nodefrontier, *d_edgefrontier, *d_distance, *d_parent;
    long* d_rank;

    int root = 5;
    
    CUDA_CHECK(cudaMalloc(&d_nodefrontier, n * sizeof(int)),
        "Failed to allocate memory for d_nodefrontier");
    
    CUDA_CHECK(cudaMalloc(&d_edgefrontier, m * sizeof(int)),
        "Failed to allocate memory for d_edgefrontier");
    
    CUDA_CHECK(cudaMalloc(&d_rank, max(static_cast<long>(n), m) * sizeof(long)),
        "Failed to allocate memory for d_rank");
    
    CUDA_CHECK(cudaMalloc(&d_distance, n * sizeof(int)),
        "Failed to allocate memory for d_distance");

    CUDA_CHECK(cudaMalloc(&d_parent, n * sizeof(int)),
        "Failed to allocate memory for d_parent");

    long* d_segments, *d_seg, *d_seg1;

    CUDA_CHECK(cudaMalloc(&d_segments, (max(static_cast<long> (n), m) + 2) * sizeof(long)),
     "Failed to allocate memory for d_segments");
    
    CUDA_CHECK(cudaMalloc(&d_seg, (max(static_cast<long>(n), m) + 2) * sizeof(long)),
        "Failed to allocate memory for d_seg");
    
    CUDA_CHECK(cudaMalloc(&d_seg1,(max(static_cast<long> (n), m) + 2) * sizeof(long)),
        "Failed to allocate memory for d_seg1");

    //temp storage for cub scan
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_segments, d_segments, max(static_cast<long> (n) , m) + 2);
    
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes),
        "Failed to allocate memory for temp_storage_bytes");

    void *d_temp_storage1 = NULL;
    size_t temp_storage_bytes1 = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage1, temp_storage_bytes1, d_seg, d_seg1, max(static_cast<long> (n), m)+2);
    
    CUDA_CHECK(cudaMalloc(&d_temp_storage1, temp_storage_bytes1),
        "Failed to allocate memory for d_temp_storage1");

    int* d_row_indices;
    size_t size = m * sizeof(int);

    // Allocate memory on the device
    CUDA_CHECK(cudaMalloc((void**)&d_row_indices, size),
        "Failed to allocate device memory for row indices");

    int* level;
    long* h_nodefrontier_size;
    long* h_edgefrontier_size;
    
    CUDA_CHECK(cudaMallocHost((void**)&h_nodefrontier_size, sizeof(long)),
        "Failed to allocate memory for h_nodefrontier_size");
    
    CUDA_CHECK(cudaMallocHost((void**)&level, sizeof(int)),
        "Failed to allocate memory for level");

    CUDA_CHECK(cudaMallocHost((void**)&h_edgefrontier_size, sizeof(long)), 
        "Failed to allocate memory for h_edgefrontier_size");
    

    h_nodefrontier_size[0] = 1;
    h_edgefrontier_size[0] = 0;


    // Initialize the allocated memories
    CUDA_CHECK(cudaMemset(d_distance, -1, n * sizeof(int)),
        "Failed to set device memory for d_distance");
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

    CUDA_CHECK(cudaMemset(d_distance + root, 0, sizeof(int)),
        "Failed to set device memory for d_distance");
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");


    //kernel to init d_nodefrontier[0] = root
    //cudamemset donot work as expected here.
    //better to use cudamemcpy to initialize d_nodefrontier[0] = root
    CUDA_CHECK(cudaMemcpy(d_nodefrontier, &root, sizeof(int), cudaMemcpyHostToDevice),
        "Failed to copy root to d_nodefrontier");
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

    CUDA_CHECK(cudaMemset(d_row_indices, 0, size),
        "Failed to set device memory for d_row_indices");
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

    while(h_nodefrontier_size[0] > 0) {

        level[0]++;

        computesegments<<<(h_nodefrontier_size[0]+1023)/1024, 1024>>>(d_nodes, d_edges, m, n, h_nodefrontier_size[0], d_nodefrontier, d_segments);
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_segments, d_segments, h_nodefrontier_size[0]+1);

        CUDA_CHECK(cudaMemcpy(&h_edgefrontier_size[0], d_segments + h_nodefrontier_size[0], sizeof(long), cudaMemcpyDeviceToHost), "Failed to copy back");

        custom_lbs<<<(h_edgefrontier_size[0]+1023)/1024, 1024>>>(d_segments, h_nodefrontier_size[0],d_seg , h_edgefrontier_size[0]);
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");
          
        computerank<<<(h_edgefrontier_size[0]+1023)/1024, 1024>>>(d_rank, d_seg , d_segments, h_edgefrontier_size[0]);
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

        computeedgefrontier<<<(h_edgefrontier_size[0]+1023)/1024, 1024>>>(d_edgefrontier, d_nodefrontier, h_edgefrontier_size[0], d_rank, d_seg , d_nodes, d_edges);
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

        markvisited<<<(h_edgefrontier_size[0]+1023)/1024, 1024>>>(h_edgefrontier_size[0], level[0], d_edgefrontier, d_distance, d_seg);
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

        cub::DeviceScan::ExclusiveSum(d_temp_storage1, temp_storage_bytes1, d_seg, d_seg1, h_edgefrontier_size[0]+1);

        CUDA_CHECK(cudaMemcpy(&h_nodefrontier_size[0], d_seg1 + h_edgefrontier_size[0], sizeof(long), cudaMemcpyDeviceToHost), "Failed to copy back");

        custom_lbs<<<(h_nodefrontier_size[0]+1023)/1024, 1024>>>(d_seg1 , h_edgefrontier_size[0], d_seg , h_nodefrontier_size[0]);
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

        computenodefrontier<<<(h_nodefrontier_size[0]+1023)/1024, 1024>>>(d_nodefrontier, d_edgefrontier, d_seg , h_nodefrontier_size[0]);
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");
    }
    int threadsPerBlock = 1024;
    int blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;
    
    update_parent<<<blocksPerGrid, threadsPerBlock>>>(
        u, 
        v, 
        d_parent, 
        d_distance, 
        m);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");

    CUDA_CHECK(cudaMemcpy(d_parent + root, &root, sizeof(int), cudaMemcpyHostToDevice), 
        "Failed to copy d_parent value");

    // int threadsPerBlock = 1024;
    // int blocksPerGrid = (num_comps + threadsPerBlock - 1) / threadsPerBlock;
    // update_root_parent<<<blocksPerGrid, threadsPerBlock>>>(
    //     d_parent, d_roots, num_comps);

    std::cout << "Parent array from Adam_BFS:\n";
    print_a<<<1,1>>>(d_parent, n);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");
    // std::cout << "Total elapsed time for adam mgpu_BFS: " << myTimer.getElapsedMilliseconds() << " ms" << std::endl;    
    
    #ifdef DEBUG
        // print d_distance to output_updated.txt
        int* h_distance = (int*)malloc(n * sizeof(int));
        cudaMemcpy(h_distance, d_distance, n * sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "numNodes: " << n << "\n";
        std::cout << "distance array:\n";
        for(int i = 0; i < n; i++)
            cout << h_distance[i] << " ";
        std::cout << std::endl;
        delete[] h_distance;
    #endif

    CUDA_CHECK(cudaFree(d_nodefrontier),            "Error freeing d_nodefrontier");
    CUDA_CHECK(cudaFree(d_edgefrontier),            "Error freeing d_edgefrontier");
    CUDA_CHECK(cudaFree(d_rank),                    "Error freeing d_rank");
    CUDA_CHECK(cudaFree(d_distance),                "Error freeing d_distance");
    CUDA_CHECK(cudaFree(d_segments),                "Error freeing d_segments");
    CUDA_CHECK(cudaFree(d_seg),                     "Error freeing d_seg");
    CUDA_CHECK(cudaFree(d_seg1),                    "Error freeing d_seg1"); 
    CUDA_CHECK(cudaFree(d_temp_storage),            "Error freeing d_temp_storage");

    CUDA_CHECK(cudaFree(d_temp_storage1),           "Error freeing d_temp_storage1");
    CUDA_CHECK(cudaFree(d_row_indices),             "Error freeing d_row_indices");

    CUDA_CHECK(cudaFreeHost(h_nodefrontier_size),   "Error freeing h_nodefrontier_size");
    CUDA_CHECK(cudaFreeHost(level),                 "Error freeing level");
    CUDA_CHECK(cudaFreeHost(h_edgefrontier_size),   "Error freeing h_edgefrontier_size");
}

int main(int argc, char* argv[]) {
    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_graph>" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::string filename = argv[1];
    graph G(filename);

    std::cout << "\n\n";
    std::cout << "filename: " << filename << std::endl;
    
    std::cout << "numVertices : " << G.numVert << ", numEdges : " << G.numEdges << std::endl;
    int numVert = G.numVert;
    long numEdges = G.numEdges;

    int* d_u;
    int* d_v;

    cudaMalloc(&d_u, sizeof(int) * G.numEdges);
    cudaMalloc(&d_v, sizeof(int) * G.numEdges);

    long* d_vertices;
    int* d_edges;

    cudaMalloc(&d_vertices, (G.vertices.size()) * sizeof(long));
    cudaMalloc(&d_edges,    G.edges.size() * sizeof(int));

    CUDA_CHECK(cudaMemcpy(d_u, G.u_arr.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice), 
                "Unable to copy original_u array to device");
    CUDA_CHECK(cudaMemcpy(d_v, G.v_arr.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice),
                "Unable to copy original_v array to device");

    // Copy the csr graph
    CUDA_CHECK(cudaMemcpy(d_vertices, G.vertices.data(), G.vertices.size() * sizeof(long), cudaMemcpyHostToDevice), 
                "Unable to copy csr_edge_offset array to device");
    CUDA_CHECK(cudaMemcpy(d_edges, G.edges.data(), G.edges.size() * sizeof(int), cudaMemcpyHostToDevice),
                "Unable to copy csr_neighbour array to device");

    adam_polak_bfs(numVert, numEdges, d_vertices, d_edges, d_u, d_v);

}