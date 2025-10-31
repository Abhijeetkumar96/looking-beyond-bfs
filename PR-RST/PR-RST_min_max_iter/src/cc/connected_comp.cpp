#include "connected_comp.h"

void dfs(int src,std::vector<bool> &vis,std::vector<std::vector<int>> &adj)
{
    vis[src] = true;
    for(auto child : adj[src])
    {
        if(!vis[child])
        {
            dfs(child,vis,adj);
        }
    }
}

int findConnected(std::vector<std::vector<int>> &adj,int n)
{
    std::vector<bool> vis(n,false);
    int count = 0;
    for (int i = 0; i < n; i++)
    {
        if(!vis[i])
        {
            dfs(i,vis,adj);
            count++;
        }
    }
    return count;
    
}