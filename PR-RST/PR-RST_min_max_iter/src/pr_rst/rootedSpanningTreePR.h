#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda.h>

// CUDA header files
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>

std::vector<int> RootedSpanningTree(const std::vector<int> &u_arr, 
                                    const std::vector<int> &v_arr,
                                    const int n);

