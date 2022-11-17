#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void vec_add(float *a, const float *b, size_t n){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(idx < n){
		a[idx] = a[idx] + b[idx];
	}
}


int main(int argc, char **argv){
	const int timesteps = 100;
	const int N = 2<<19;	//524288, will spin the kernels for a couple of ys 
	float *h_a, *h_b;
	float *d_a, *d_b;
	
	h_a = (float *) malloc(sizeof(float) * N);
	h_b = (float *) malloc(sizeof(float) * N);
	
	CUDA_SAFE_CALL( cudaMalloc(&d_a, sizeof(float) * N) );
	CUDA_SAFE_CALL( cudaMalloc(&d_b, sizeof(float) * N) );
	
	for(int i = 0; i < N; i++){
		h_a[i] = 2.0f;
		h_b[i] = 4.0f;
	}
	
	CUDA_SAFE_CALL( cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice) );	//from here, we keep the data on the GPU
	
	cudaStream_t computeStream;
	CUDA_SAFE_CALL( cudaStreamCreate(&computeStream) );

	int blocksize = 256;
	int gridsize = (N + blocksize-1)/blocksize;

	double start = omp_get_wtime();
	
	//compute loop
	for(int i = 0; i < timesteps; i++){
		//lauch a series of small compute kernels
		for(int j = 0; j < 20; j++){
			h_a[0] = h_a[0] + h_b[0];
			vec_add<<<gridsize, blocksize, 0, computeStream>>>(d_a, d_b, N);
			//CUDA_SAFE_CALL( cudaStreamSynchronize(computeStream) );
		}
		CUDA_SAFE_CALL( cudaStreamSynchronize(computeStream) );
	}

	double elapsed = omp_get_wtime() - start;

	//test the result
	float test = .0f;
	CUDA_SAFE_CALL( cudaMemcpy(&test, d_a, sizeof(float), cudaMemcpyDeviceToHost) );
	if(test == h_a[0]){
		std::cout << "Success! elapsed " << elapsed << "\n";
	}else{
		std::cout << "Error! test: " << test << " h_a[0]: " << h_a[0] << "\n";
	}

	return EXIT_SUCCESS;
}