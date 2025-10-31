#include "grafting.h"

__global__
void DetermineWinners(int *u_arr, int *v_arr, int *rep, int *winner, int edges, int *d_flag,int isMaxIteration) {
	
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < edges) {
		// for i from 1 to n:
		//for each neighbor j of vertex i:
		//Assuming u as vertex i and v as all neighbours of u
		int u = u_arr[tid];
		int v = v_arr[tid];

		int rep_u = rep[u], rep_v = rep[v];

		// isMaxIteration == 1 ==> child should be the maximum
		if(rep_u != rep_v) {
			*d_flag = 1;
			if(isMaxIteration){
				winner[max(rep_u,rep_v)] = tid;
			}
			else{
				winner[min(rep_u,rep_v)] = tid;
			}
		}
	}
}

__global__ 
void UpdateLabels(int *u_arr, int *v_arr, int *rep, int *winner, int edges, int *marked_parent, int *onPath,int isMaxIteration)
{

  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < edges)
  {
    int u = u_arr[tid];
    int v = v_arr[tid];

	int rep_u = rep[u], rep_v = rep[v], max_rep = max(rep_u,rep_v), min_rep = min(rep_u,rep_v);

	if(rep_u != rep_v){
		if(isMaxIteration && winner[max_rep] == tid)
		{
			int child_node = (rep_u == max_rep ? u : v);
			int parent_node = (rep_u == max_rep ? v : u);

			marked_parent[child_node] = parent_node;
			onPath[child_node] = 1;
		}
		if(!isMaxIteration && winner[min_rep] == tid)
		{
			int child_node = (rep_u == min_rep ? u : v);
			int parent_node = (rep_u == min_rep ? v : u);

			marked_parent[child_node] = parent_node;
			onPath[child_node] = 1;			
		}
	}
	// if(rep_u > rep_v && winner[rep_u] == tid)
	// {
	// 	marked_parent[u] = v;
	// 	onPath[u] = 1;
	// }
  }
}

void Graft(
	int vertices,
	int edges,
	int *d_u_ptr,
	int *d_v_ptr,
	int *d_ptr,
	int *d_winner_ptr,
	int *d_marked_parent,
	int *d_OnPath,
	int *d_flag,
	int isMaxIteration
)
{
		int numThreads = 1024;
		// int numBlocks_n = (vertices + numThreads - 1) / numThreads;
		int numBlocks_e = (edges + numThreads - 1) / numThreads;

		// Step 2.1: Determine potential winners for each vertex
		DetermineWinners<<<numBlocks_e, numThreads>>> (d_u_ptr, d_v_ptr, d_ptr, d_winner_ptr, edges, d_flag,isMaxIteration);
		cudaDeviceSynchronize();

		// Step 2.2: Update labels based on winners and mark parents
    	UpdateLabels<<<numBlocks_e, numThreads>>>(d_u_ptr, d_v_ptr, d_ptr, d_winner_ptr, edges, d_marked_parent, d_OnPath,isMaxIteration);
    	cudaDeviceSynchronize();	
}