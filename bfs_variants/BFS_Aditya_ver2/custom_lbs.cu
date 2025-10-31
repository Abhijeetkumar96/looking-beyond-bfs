#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
#include<thrust/scan.h>
#include "mytimer.h"

using namespace std;




__global__ void custom_lbs(int *a, int n_a , int *b, int n_b){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n_b){
        int l = 0, r = n_a-1;
        while(l<r){
            int mid = (l+r)/2;
            if(a[mid]<= i){
                l = mid+1;
            }else{
                r = mid;
            }
        }
        b[i] = l;
    }
}


int main(){
    mytimer module_timer{};

    freopen("inputforlbs.txt", "r", stdin);
    int n;
    cin>>n;
    int a[n]; 
    for(int i=0;i<n;i++){
        cin>>a[i];
    }

    module_timer.timetaken_reset("time for reading input : ");

    thrust::inclusive_scan(a , a + n , a);

    module_timer.timetaken_reset("time for prefix sum : ");

    int n_b = a[n-1];
    int *d_a , *d_b ;

    cudaMalloc((void **)&d_a, n*sizeof(int));
    cudaMalloc((void **)&d_b, n_b*sizeof(int));
    cudaMemcpy(d_a, a, n*sizeof(int), cudaMemcpyHostToDevice);
    
    module_timer.timetaken_reset("time for memory transfer : ");

    custom_lbs<<<(n_b+255)/256, 256>>>(d_a, n, d_b, n_b);
    cudaDeviceSynchronize();

    module_timer.timetaken_reset("time for kernel execution : ");

    int *b;
    cudaMallocHost((void **)&b, n_b*sizeof(int));
    cudaMemcpy(b, d_b, n_b*sizeof(int), cudaMemcpyDeviceToHost);
    // for(int i=0;i<n_b;i++){
    //     cout<<b[i]<<" ";
    // }
    // cout<<endl;

}




