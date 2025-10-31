#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include <random>
#include <chrono>
#include <map>
#include <queue>
#include <iomanip>

int numberOfComponents(std::vector<int> &par);
int markedComponents(std::vector<int> &marked);
int rootedComponents(std::vector<int> &par);
bool dfs_directed(int src,std::vector<int> &vis,std::vector<std::vector<int>> &adj);
bool validateRST(const std::vector<int> &parent,const int comp_count);
int treeDepth(int root,std::vector<std::vector<int>> &adj,int n);