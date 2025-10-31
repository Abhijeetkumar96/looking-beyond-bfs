
#include "verifySol.h"

int numberOfComponents(std::vector<int> &par)
{
	std::map<int,int> components;
	for(auto i : par)
	{
		components[i]++;
	}
	return components.size();
}

int markedComponents(std::vector<int> &marked)
{
	int count = 0;
	for(auto i : marked)
	{
		count += (i != -1);
	}
	return count;
}

int rootedComponents(std::vector<int> &par)
{
	int count = 0;
	for (int i = 0; i < (int)par.size(); i++)
	{
		count += (i == par[i]);
	}
	return count;
	
}

bool dfs_directed(int src,std::vector<int> &vis,std::vector<std::vector<int>> &adj)
{
	vis[src] = 1;
	for(auto child : adj[src]){
		if(vis[child] == 0){
			if(dfs_directed(child,vis,adj)){
				return true;
			}
		}
		else if(vis[child] == 1){
			return true;
		}
	}
	vis[src] = 2;
	return false;
}

bool validateRST(const std::vector<int> &parent,const int comp_count)
{
	int n = parent.size();
	std::vector<int> roots;

	for (int i = 0; i < n; i++)
	{
		if(parent[i] == i)
			roots.push_back(i);
	}

	int k = roots.size();

	if(k != comp_count)
	{
		std::cout<<"Wrong : found "<< k <<" roots instead of "<<comp_count<<std::endl;
		return false;		
	}

	std::vector<std::vector<int>> adj(n);
	for (int i = 0; i < n; i++)
	{
		if(parent[i] != i)
			adj[parent[i]].push_back(i);
	}

	std::vector<int> vis(n,0);
	
	for(auto root : roots)
	{
		if(dfs_directed(root,vis,adj))
		{
			std::cout<<"Wrong : Tree has cycles - node["<<root<<"]"<<std::endl;
			return false;
		}
	}

	for(int i=0;i<n;i++)
	{
		if(vis[i] != 2)
		{	
			std::cout<<"Wrong : Tree has cycles - node["<<i<<"]"<<std::endl;
			return false;
		}
	}

	std::vector<int> max_tree_depths;
	for(auto root : roots)
	{
		max_tree_depths.push_back(treeDepth(root,adj,n));
	}

	std::cout << "Max tree depth = " << *max_element(max_tree_depths.begin(),max_tree_depths.end()) << std::endl;
	std::cout << std::fixed << std::setprecision(2) << "Avg tree depth = " << accumulate(max_tree_depths.begin(),max_tree_depths.end(),(double)0) / (double)roots.size() << std::endl;
	return true;
}

int treeDepth(int root,std::vector<std::vector<int>> &adj,int n)
{
	std::vector<int> depth(n,1e9);
	std::queue<int> q;
	q.push(root);
	depth[root] = 0;

	while(!q.empty())
	{
		int cur = q.front();
		q.pop();

		for(auto child : adj[cur])
		{
			if(depth[child] > depth[cur] + 1)
			{
				depth[child] = depth[cur] + 1;
				q.push(child);
			}
		}
	}

	for(int i=0;i<n;i++){
		if(depth[i] == 1e9) depth[i] = -1;
	}

	return *max_element(depth.begin(),depth.end());
}