#include <iostream>
#include <math.h>
#include <random>
#include <stdlib.h>

//computes a vector addition of two vectors a and b. Result stored in res
__global__ void vec_add(float *a, float *b, float *res, int n){
	it i = threadIdx.x + blockIdx.x * blockDim.x;
	res[i] = a[i] + b[i];
	
}

int main(void){
	int n =1<<20;
	
	float *a = new float[n];
	float *b = new float[n];
	float *res = new float[n];
	
	cudaMallocManaged(&a, n*sizeof(float));
	cudaMallocManaged(&b, n*sizeof(float));
	cudaMallocManaged(&res, n*sizeof(float));
	
	for(int  i = 0; i < n; ++i){		//init
		a[i] = -1.0f;
		b[i] = 2.0f;
		res[i] = 0.0f;
	}
	
	int blocksize = 256;
	int gridsize = (n + blocksize -1) / blocksize;
	vec_add<<<gridsize,blocksize>>>(a,b,res, n);
	cudaDeviceSynchronize();	


	//check
	float c = 0;
	for(int i = 0; i < n; ++i){
		c += res[i];
	}
	if(c == n){
		std::cout << "success \n";
	}
	
	delete [] a;
	delete [] b;
	delete [] res;
	
	return EXIT_SUCCESS;
}
