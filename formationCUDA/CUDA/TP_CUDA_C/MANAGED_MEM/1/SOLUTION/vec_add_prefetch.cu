#include <iostream>
#include <math.h>
#include <random>
#include <stdlib.h>
#include <omp.h>

//computes a vector addition of two vectors a and b. Result stored in res
__global__ void vec_add(float *a, float *b, float *res, int n){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(idx < n)
	res[idx] = a[idx] + b[idx];
	
}

int main(void){
	int n =1<<20;
	
	float *a, *b, *res;
	cudaMallocManaged(&a, n * sizeof(float));
	cudaMallocManaged(&b, n * sizeof(float));
	cudaMallocManaged(&res, n * sizeof(float)); 
	
	for(int  i = 0; i < n; ++i){		//init
		a[i] = -1.0f;
		b[i] = 2.0f;
		res[i] = 0.0f;
	}
	
	//timing
	double start = omp_get_wtime();
	
	//prefetching
	int device = -1;
	cudaGetDevice(&device);
	cudaMemPrefetchAsync(a, n * sizeof(float), device);
	cudaMemPrefetchAsync(b, n * sizeof(float), device);
	cudaMemPrefetchAsync(res, n * sizeof(float), device);
	
	int blocksize = 256;
	int gridsize = (n + blocksize - 1) / blocksize;
	vec_add<<<gridsize, blocksize>>>(a,b,res, n);
	cudaDeviceSynchronize();
	
	double elapsed = omp_get_wtime() -start;
	
	//check
	float c = 0;
	for(int i = 0; i < n; ++i){
		c += res[i];
	}
	if(c == n){
		std::cout << "success \n";
	}else{
		std::cout << "error, c = " << c << " expected: " << n << "\n";
	}
	
	cudaFree(a);
	cudaFree(b);
	cudaFree(res);
	std::cout << "Elapsed time: " << elapsed << "\n";
	return EXIT_SUCCESS;
}