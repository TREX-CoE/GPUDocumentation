#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


//__device__ int ret[100];

__device__ float d_test;
__global__ void kernel1() { d_test = 1.0; }

__global__ void my_kernel(int *ret){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < 100)
		ret[idx] = idx;
	
}

int main(int argc, char **argv){
	
	int *ret, *ret_map;
	
	cudaSetDeviceFlags(cudaDeviceMapHost);
	
	cudaHostAlloc((void **)&ret, 100 * sizeof(int), cudaHostAllocMapped);
	cudaHostGetDevicePointer((void **)&ret_map, ret, 0);
	
	for(int i = 0; i < 100; ++i){
		ret[i] = 0;
	}
	
	my_kernel<<<5,32>>>(ret_map);
	
	for(int i = 60; i < 70; ++i){
		std::cout << i << " ";
	}
	std::cout << "\n" << std::flush;
	
	// initialise variables
	float h_test = 0.0;
	cudaMemset(&d_test,0,sizeof(float));

	// invoke kernel
	kernel1 <<<1,1>>> ();

	// Copy device variable to host and print
	cudaMemcpyFromSymbol(&h_test, d_test, sizeof(float), 0, cudaMemcpyDeviceToHost);
	printf("%f\n",h_test);  
	
	return EXIT_SUCCESS;
}