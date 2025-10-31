#include "expand.cu"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <queue>
#include <stack>
#include <map>
#include <set>
#include <cmath>
#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>

#include "Graph_CSR.h"
#include "mytimer.h"

// #define DEBUG

using namespace std;

__global__
void computesegments(int* d_nodes , int* d_edges ,int* d_m ,int* d_n,  int d_nodefrontier_size , int* d_nodefrontier , int* d_segments){

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < d_nodefrontier_size){
    int node = d_nodefrontier[id];
    int start = d_nodes[node];
    int end = (node == *d_n - 1) ? *d_m : d_nodes[node + 1];
    d_segments[id] = end - start;
  }

}

__global__
void computerank(int* d_rank , int* d_seg , int* d_segments , int d_edgefrontier_size){
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < d_edgefrontier_size){
    d_rank[id] = id - d_segments[d_seg[id]];
  }
}

__global__
void computeedgefrontier(int* d_edgefrontier ,int* d_nodefrontier, int d_edgefrontier_size , int* d_rank , int* d_seg , int* d_nodes , int* d_edges){
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < d_edgefrontier_size){
    int seg = d_seg[id];
    int rank = d_rank[id];
    d_edgefrontier[id] = d_edges[d_nodes[d_nodefrontier[seg]] + rank];
  }

}


__global__
void markvisited(int d_edgefrontier_size , int level , int* d_edgefrontier , int* d_distance , int* d_seg){
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < d_edgefrontier_size){
    d_seg[id] = (-1 == atomicCAS(d_distance + d_edgefrontier[id], -1, level));
  }
}


__global__
void computenodefrontier(int* d_nodefrontier , int* d_edgefrontier , int* d_seg , int d_nodefrontier_size){
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < d_nodefrontier_size){
    d_nodefrontier[id] = d_edgefrontier[d_seg[id]];
  }
}


__global__ 
void custom_lbs(int *a, int n_a , int *b , int n_b){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i< n_b){
      int l = 1, r = n_a;
      while(l<r){
          int mid = (l+r)/2;
          if(a[mid]<= i){
              l = mid+1;
          }else{
              r = mid;
          }
      }
      b[i] = l-1;
  }
}

void serialbfs(int src, const std::vector<std::vector<int>> &adjlist, std::vector<int>& dist_serial) {   

    #ifdef DEBUG
        std::cout << "CSR Array:\n";
        
        std::cout << "Size of adjlist: " << adjlist.size() << " ";
        std::cout << "\nAdjlist from serial BFS:\n";
        for(int i = 0; i < adjlist.size(); ++i) {
            std::cout << i << ": ";
            for(int j = 0; j < adjlist[i].size(); ++j) {
                std::cout << adjlist[i][j] << " ";
            }
            std::cout << "\n";
        }
    #endif

    int n = adjlist.size();
    std::vector<bool> visited_serial;
    std::queue<int>q;

    visited_serial.resize(n);
    dist_serial[src] = 0;
    q.push(src);
    visited_serial[src] = 1;
    while(!q.empty())
    {
      int parent = q.front();
      q.pop();
      for(int i = 0; i < adjlist[parent].size(); i++) {
        if(visited_serial[adjlist[parent][i]] != 1) {
          q.push(adjlist[parent][i]);
          visited_serial[adjlist[parent][i]] = 1;
          
          if(dist_serial[adjlist[parent][i]] > dist_serial[parent] + 1)
            dist_serial[adjlist[parent][i]] = dist_serial[parent] + 1;
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
        std::cout<<"\nVALUE AT " << i << "IS DIFFERENT FOR BOTH THE ARRAYS\n";
        std::cout << "Distance of parallel[" << i << "] = " << arr1[i] << " whereas distance of serial[" << i << "] = " << arr2[i] << std::endl;
      #endif
      flag = 0;
      return flag;
    }

  }
  std::cout<<"\nCongratulations.. The bfs results are correct..\n";

  return flag;
}

int main(int argc, char* argv[])
{
  mytimer module_timer {};
  mytimer segment_timer {};

  cudaSetDevice(0);

  if(argc < 2) {cout<<"pass the dataset name as an argument, like this prog_name dataset_name\n";return 0;}

	cout << "=========================================================\n\n";
	cout << "parallel bfs for the file " << argv[1] << " started\n\n";

	//string datasetPath = "../datasets/";
	ifstream inputGraph;
	inputGraph.open(argv[1]);
	//inputGraph.open(argv[1]);

	if(inputGraph.is_open() == false) {
    cerr << "cannot found/open the dataset file";
    return 0;
  }
	
	unweightedGraph G(inputGraph);
	// cout << "Created unweightedGraph" << endl;

  //input is taken from input.txt csr format of graph
  //CSR initializations
  int* n, *m;
  int* nodes , *edges;
  cudaMallocHost((void**)&n, sizeof(int));
  cudaMallocHost((void**)&m, sizeof(int));
  n[0] = G.totalVertices;
	m[0] = G.totalEdges;
  int x = n[0];
  int y = m[0];
  cudaMallocHost((void**)&nodes, x * sizeof(int));
  cudaMallocHost((void**)&edges, y * sizeof(int));
  for(int i = 0; i < n[0]; i++)
    nodes[i] = G.offset[i];
  for(int i = 0; i < m[0]; i++)
    edges[i] = G.neighbour[i];

  
  // module_timer.timetaken_reset("reading and csr conversion time : ");

  int *d_nodes, *d_edges , *d_n , *d_m;
  cudaMalloc((void**)&d_n, sizeof(int));
  cudaMalloc((void**)&d_m, sizeof(int));
  cudaMalloc((void**)&d_nodes, n[0]*sizeof(int));
  cudaMalloc((void**)&d_edges, m[0]*sizeof(int));
  cudaMemcpy(d_n, n, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_m, m, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nodes, nodes, n[0]*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_edges, edges, m[0]*sizeof(int), cudaMemcpyHostToDevice);

  //BFS initializations
  int* d_nodefrontier, *d_edgefrontier, * d_tempcount, * d_rank ,* d_distance;
  cudaMalloc(&d_nodefrontier, n[0]*sizeof(int));
  cudaMalloc(&d_edgefrontier, m[0]*sizeof(int));
  cudaMalloc(&d_tempcount, sizeof(int));
  cudaMalloc(&d_rank, max(n[0],m[0])*sizeof(int));
  cudaMalloc(&d_distance, n[0]*sizeof(int));

  thrust::device_vector<int> d_segments(m[0]+2, 0);
  thrust::device_vector<int> d_seg(m[0]+2, 0);
  thrust::device_vector<int> d_seg1(m[0]+2, 0);
  
  //BFS

  cudaMemset(d_distance, -1, n[0] * sizeof(int));
  cudaMemset(d_distance, 0, sizeof(int));
  cudaMemset(d_nodefrontier, 0, sizeof(int));

  // module_timer.timetaken_reset("alloc and initialization time : ");

  //not pinned for now
  int level = 0;
  int h_nodefrontier_size = 1;
  int h_edgefrontier_size = 0;
  


  while(h_nodefrontier_size > 0) {
    
    level++;

    computesegments<<<(h_nodefrontier_size+1023)/1024, 1024>>>(d_nodes, d_edges, d_m, d_n, h_nodefrontier_size, d_nodefrontier, thrust::raw_pointer_cast(d_segments.data()));
    cudaDeviceSynchronize();

    // module_timer.timetaken_reset("segment computation time : ");
    
    thrust::exclusive_scan(d_segments.begin(), d_segments.begin() + h_nodefrontier_size + 1, d_segments.begin());

    // module_timer.timetaken_reset("segment scan time : ");

    cudaMemcpy(&h_edgefrontier_size, thrust::raw_pointer_cast(d_segments.data()) + h_nodefrontier_size, sizeof(int), cudaMemcpyDeviceToHost);

    // module_timer.timetaken_reset("edgefrontier size copy time : ");
    
    custom_lbs<<<(h_edgefrontier_size+1023)/1024, 1024>>>(thrust::raw_pointer_cast(d_segments.data()), h_nodefrontier_size, thrust::raw_pointer_cast(d_seg.data()) , h_edgefrontier_size);
    cudaDeviceSynchronize();

    // module_timer.timetaken_reset("custom lbs time : ");
      
    //create rank array rank[i] = i - exclusivescan[seg[i]]
    computerank<<<(h_edgefrontier_size+1023)/1024, 1024>>>(d_rank, thrust::raw_pointer_cast(d_seg.data()), thrust::raw_pointer_cast(d_segments.data()), h_edgefrontier_size);
    cudaDeviceSynchronize();

    // module_timer.timetaken_reset("rank computation time : ");

    computeedgefrontier<<<(h_edgefrontier_size+1023)/1024, 1024>>>(d_edgefrontier, d_nodefrontier, h_edgefrontier_size, d_rank, thrust::raw_pointer_cast(d_seg.data()), d_nodes, d_edges);
    cudaDeviceSynchronize();

    // module_timer.timetaken_reset("edgefrontier computation time : ");

    markvisited<<<(h_edgefrontier_size+1023)/1024, 1024>>>(h_edgefrontier_size, level, d_edgefrontier, d_distance, thrust::raw_pointer_cast(d_seg.data()));
    cudaDeviceSynchronize();

    // module_timer.timetaken_reset("mark visited time : ");

    //new node frontier size
    thrust::exclusive_scan(d_seg.begin(), d_seg.begin() + h_edgefrontier_size+1, d_seg1.begin());

    // module_timer.timetaken_reset("nodefrontier scan time : ");

    cudaMemcpy(&h_nodefrontier_size, thrust::raw_pointer_cast(d_seg1.data()) + h_edgefrontier_size, sizeof(int), cudaMemcpyDeviceToHost);

    // module_timer.timetaken_reset("nodefrontier size copy time : ");

    // cout<<"new nodefrontier size : "<<h_nodefrontier_size<<endl;
    mytimer segment_timer {};
    
    custom_lbs<<<(h_nodefrontier_size+1023)/1024, 1024>>>(thrust::raw_pointer_cast(d_seg1.data()), h_edgefrontier_size, thrust::raw_pointer_cast(d_seg.data()) , h_nodefrontier_size);
    cudaDeviceSynchronize();

    // segment_timer.timetaken_reset("custom lbs time : ");

    //new node frontier
    computenodefrontier<<<(h_nodefrontier_size+1023)/1024, 1024>>>(
      d_nodefrontier, 
      d_edgefrontier, 
      thrust::raw_pointer_cast(d_seg.data()), 
      h_nodefrontier_size);

    cudaDeviceSynchronize();

    // module_timer.timetaken_reset("nodefrontier computation time : ");
  }

  module_timer.timetaken_reset("BFS time : ");

  //print d_distance to output_updated.txt
  std::vector<int> h_distance(n[0]);
  std::vector<int> dist_serial(n[0], INT_MAX);

  cudaMemcpy(h_distance.data(), d_distance, n[0] * sizeof(int), cudaMemcpyDeviceToHost);

  serialbfs(0, G.adjlist, dist_serial);

  #ifdef DEBUG
      std::cout << "Adam BFS output:\n";

        for(int i = 0; i < n[0]; i++)
            std::cout << h_distance[i] << " ";
        std::cout << std::endl;
        

        std::cout << "Serial BFS output:\n";
        for(auto i : dist_serial)
            std::cout << i << " ";
        std::cout << std::endl;
    #endif

  return !(verify(h_distance, dist_serial));
}