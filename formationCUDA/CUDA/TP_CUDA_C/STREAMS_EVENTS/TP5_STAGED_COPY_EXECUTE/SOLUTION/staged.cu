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
	cudaStream_t stream[4];
	CUDA_SAFE_CALL(cudaStreamCreate(&stream[0]));
	CUDA_SAFE_CALL(cudaStreamCreate(&stream[1]));
	CUDA_SAFE_CALL(cudaStreamCreate(&stream[2]));
	CUDA_SAFE_CALL(cudaStreamCreate(&stream[3]));
	REAL *devPtrA, *devPtrC;
	CUDA_SAFE_CALL(cudaHostGetDevicePointer(&devPtrA, A, 0) );
	CUDA_SAFE_CALL(cudaHostGetDevicePointer(&devPtrC, C, 0) );
	int size=N*N*sizeof(REAL)/nStreams;		//okay (it is an int-op)
	for (int i=0; i<nStreams; i++) {
		int offset = i*N*N/nStreams;
#ifdef DEBUG
		std::cout<< "offset: " << offset << " (in elements), A[offset]: " << *(A + offset) << ", copy size: " << size <<  " bytes\n";
#endif
		cudaMemcpyAsync(devPtrA+offset, A+offset, size, cudaMemcpyHostToDevice, stream[i]);
		dim3 block(TILE_SIZE,TILE_SIZE);
		dim3 grid(N / (block.x), N / (block.y * nStreams));
		testStaged<<<grid, block, 0, stream[i]>>>(devPtrA+offset, devPtrC+offset);
	}
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	for(int i = 0; i < nStreams; i++){
		cudaStreamDestroy(stream[i]);
	}
}

void performStagedLoadPinned(REAL *A, REAL * C) {
	int nStreams = 4;
	cudaStream_t stream[4];
	CUDA_SAFE_CALL(cudaStreamCreate(&stream[0]));
	CUDA_SAFE_CALL(cudaStreamCreate(&stream[1]));
	CUDA_SAFE_CALL(cudaStreamCreate(&stream[2]));
	CUDA_SAFE_CALL(cudaStreamCreate(&stream[3]));
	REAL *devPtrA, *devPtrC;
	CUDA_SAFE_CALL(cudaMalloc((void **)&devPtrA, N * N * sizeof(REAL)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&devPtrC, N * N * sizeof(REAL)));
	int size=N*N*sizeof(REAL)/nStreams;		//okay (it is an int-op)
	for (int i=0; i<nStreams; i++) {
		int offset = i*N*N/nStreams;
#ifdef DEBUG
		std::cout<< "offset: " << offset << " (in elements), A[offset]: " << *(A + offset) << ", copy size: " << size <<  " bytes\n";
#endif
		cudaMemcpyAsync(devPtrA+offset, A+offset, size, cudaMemcpyHostToDevice, stream[i]);
		dim3 block(TILE_SIZE,TILE_SIZE);
		dim3 grid(N / (block.x), N / (block.y * nStreams));
		testStaged<<<grid, block, 0, stream[i]>>>(devPtrA+offset, devPtrC+offset);
	}
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	CUDA_SAFE_CALL(cudaMemcpy(C, devPtrC, N * N * sizeof(REAL), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(devPtrA));
	CUDA_SAFE_CALL(cudaFree(devPtrC));
	for(int i = 0; i < nStreams; i++){
		cudaStreamDestroy(stream[i]);
	}
}


int main(int argc, char ** argv){
	cudaSetDeviceFlags(cudaDeviceMapHost);
	
	REAL *A, *C;
	CUDA_SAFE_CALL( cudaHostAlloc((void **)&A, N * N * sizeof(REAL), cudaHostAllocDefault) ); //cudaHostAllocMapped
	CUDA_SAFE_CALL( cudaHostAlloc((void **)&C, N * N * sizeof(REAL), cudaHostAllocDefault) ); //cudaHostAllocDefault
	
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
