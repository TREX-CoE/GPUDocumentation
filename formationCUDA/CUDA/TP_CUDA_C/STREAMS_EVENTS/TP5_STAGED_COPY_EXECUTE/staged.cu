#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define REAL double
#define N 1024
#define DEBUG

using clock_value_t = long long;

#define CUDA_SAFE_CALL(call) { gpuAssert((call), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false){
   if (code != cudaSuccess){
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ void sleep(clock_value_t sleep_cycles)
{
    clock_value_t start = clock64();
    clock_value_t cycles_elapsed;
    do { cycles_elapsed = clock64() - start; } 
    while (cycles_elapsed < sleep_cycles);
}

__global__ void testStaged(const REAL * A, REAL * C){
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	//C[y * N + x] = A[y * N + x] +1;
	sleep(50000);
	C[y * N + x] = A[y * N + x] * -1.0;
}

void performStagedLoadZeroCopy(REAL *A, REAL * C) {
	int nStreams = 4;
	//Task 2: perform a staged concurrent copy and execute with zero copy
}

void performStagedLoadPinned(REAL *A, REAL * C) {
	int nStreams = 4;
	cudaStream_t stream[nStreams];
	for(int i=0; i < nStreams: i++){
		CUDA_SAFE_CALL(cudaStreamCreate(&stream[i]));
	}
	REAL *bufA, *bufA;
	CUDA_SAFE_CALL(cudaMalloc((void**)&bufA, N*N,sizeof(REAL)));	
	CUDA_SAFE_CALL(cudaMalloc((void**)&bufB, N*N,sizeof(REAL)));
	int chunk_size = N*N*sizeof(REAL)/nStreams;
	for(int i = 0;i< nStreams; i++){
		int offset = i*N/nStreams;
		cudaMemcpyAsync(bufA+offset,A+offset, size, cudaMemcpyHostToDevice, stream[i]);
			
	}	

}


int main(int argc, char ** argv){
	cudaSetDeviceFlags(cudaDeviceMapHost);
	
	//zero-copy host memory
	REAL *A, *C;
	CUDA_SAFE_CALL( cudaHostAlloc((void **)&A, N * N * sizeof(REAL), cudaHostAllocMapped) ); 
	CUDA_SAFE_CALL( cudaHostAlloc((void **)&C, N * N * sizeof(REAL), cudaHostAllocMapped) ); 
	
	//pinned host memory
	REAL *X, *Z;
	cudaMallocHost(&X, N * N * sizeof(REAL));
	cudaMallocHost(&Z, N * N * sizeof(REAL));
	
	for(int i = 0; i < N*N; ++i){
		A[i] = (REAL) i;
		X[i] = (REAL) i;
		C[i] = (REAL) 0;
		Z[i] = (REAL) 0;
	}
	performStagedLoadPinned(X, Z);
	performStagedLoadZeroCopy(A, C);
	
	
#ifdef DEBUG
	for(int i = 0; i < 10; ++i)
		std::cout << Z[i] << " ";
	std::cout << "\n";
	for(int i = N*N-10; i < N*N; ++i)
		std::cout << Z[i] << " ";
	
	std::cout << "\n";
#endif
	CUDA_SAFE_CALL( cudaFreeHost(A) );
	CUDA_SAFE_CALL( cudaFreeHost(C) );
	CUDA_SAFE_CALL( cudaFreeHost(X) );
	CUDA_SAFE_CALL( cudaFreeHost(Z) );
	
	return EXIT_SUCCESS;
}
