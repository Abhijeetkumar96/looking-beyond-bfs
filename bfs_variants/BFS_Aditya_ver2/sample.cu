#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
#include<thrust/scan.h>



using namespace std;

int main(){
    int a[5] = { 1, 2, 3, 4, 5};

    int* sum = thrust::inclusive_scan(a , a + 5 , a);

    for(int i=0;i<5;i++){
        cout<<sum[i]<<" ";
    }
}