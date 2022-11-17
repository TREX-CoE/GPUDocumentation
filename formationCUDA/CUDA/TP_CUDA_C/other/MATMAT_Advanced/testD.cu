#include "test.h"

#define TILE_SIZE 16

#define CUDA_SAFE_CALL(call) { gpuAssert((call), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false){
   if (code != cudaSuccess){
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void dotProduct(const REAL * a, const REAL *b, REAL *result){
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ REAL temp[16];
	temp[idx] = a[idx] * b[idx];
	__syncthreads();
	if(idx == 0){
		REAL sum = 0; 
		for(int i = 0; i < 16; i++){
			sum += temp[i];
		}
		*result = sum;
	}
}

void TestD::initLibs(REAL *r){
	REAL x[16] = {0, 0, 0, 0, 0, 0, 0, 0, -28, 0, 41, -15, 0, 20, -55.744289, -76.930283};
	REAL y[16] = {0, 0, 0, 0, 0, 0, 0, 0, -54.654705, 0, 23.670898, -4.533447, 0, 79.964569, -65.6026, -4};

	
	REAL expected = .0f, actual = .0f;
    for (int i = 0; i < 16; i++) {
        expected += x[i] * y[i];
    }
	
	REAL *devPtrA, *devPtrB, *devRes;
	
    CUDA_SAFE_CALL(cudaMalloc((void **)&devPtrA, 16 * sizeof(REAL)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&devPtrB, 16 * sizeof(REAL)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&devRes, 1 * sizeof(REAL)));
	
	CUDA_SAFE_CALL(cudaMemcpy(devPtrA, x, 16*sizeof(REAL), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(devPtrB, y, 16*sizeof(REAL), cudaMemcpyHostToDevice));
	
	dotProduct<<<1,16>>>(devPtrA, devPtrB, devRes);
	
	CUDA_SAFE_CALL(cudaMemcpy(&actual, devRes, 1*sizeof(REAL), cudaMemcpyDeviceToHost));
	
	cudaDeviceSynchronize();
	
	std::cout << " actual : " << actual << "\n";
	
    (*r) = fabs(expected -  actual);
}

void TestD::execTest(const REAL * __restrict__ A, const REAL * __restrict__ B, REAL * __restrict__ C) const {
	//naive implementation
	//this->naiveImpl(A, B, C);
	//zero copy
	this->zeroCopy(A, B, C);
}

__global__ void sharedMemMatMatMult(const REAL * __restrict__ A, const REAL * __restrict__ B, REAL * __restrict__ C){
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	__shared__ REAL sA[TILE_SIZE * TILE_SIZE];
	__shared__ REAL sB[TILE_SIZE * TILE_SIZE];
	
	REAL temp = .0f;
	
	for(int i = 0; i < (N / TILE_SIZE); ++i){
		sA[threadIdx.y * TILE_SIZE + threadIdx.x] = A[i * TILE_SIZE + threadIdx.x + row * N];
		sB[threadIdx.y * TILE_SIZE + threadIdx.x] = B[threadIdx.y * N + i * TILE_SIZE * N + col];
		//sB[threadIdx.x * TILE_SIZE + threadIdx.y] = B[threadIdx.y * N + i * TILE_SIZE * N + col]; //variante A slower
		
		__syncthreads();
		
		for(int j = 0; j < TILE_SIZE; ++j){
			temp += sA[j + threadIdx.y * TILE_SIZE] * sB[j * TILE_SIZE + threadIdx.x];
			//temp += sA[j + threadIdx.y * TILE_SIZE] * sB[j + TILE_SIZE * threadIdx.x]; //variante A
		}
		
		__syncthreads();
	}
	C[row * N + col] = temp;
}

__global__ void naiveMatMatMult(const REAL * __restrict__ A, const REAL * __restrict__ B, REAL * __restrict__ C){
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	for(int k = 0; k < N; ++k){
		C[x * N + y] += A[x * N + k] * B[y + k * N];
	}
}

void TestD::naiveImpl(const REAL * __restrict__ A, const REAL * __restrict__ B, REAL * __restrict__ C) const {
	REAL *devPtrA, *devPtrB, *devPtrC;
	
	CUDA_SAFE_CALL(cudaMalloc((void **)&devPtrA, N * N * sizeof(REAL)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&devPtrB, N * N * sizeof(REAL)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&devPtrC, N * N * sizeof(REAL)));
	
	CUDA_SAFE_CALL(cudaMemcpy(devPtrA, A, N * N * sizeof(REAL), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(devPtrB, B, N * N * sizeof(REAL), cudaMemcpyHostToDevice));
	
	dim3 block(TILE_SIZE, TILE_SIZE);
	dim3 grid(N / block.x, N / block.y);
	//naiveMatMatMult<<<grid, block>>>(devPtrA, devPtrB, devPtrC);
	sharedMemMatMatMult<<<grid, block>>>(devPtrA, devPtrB, devPtrC);
	cudaDeviceSynchronize();
	cudaError_t err;
	if((err = cudaGetLastError()) != cudaSuccess){
		std::cout << "Error\n";
		std::cout << cudaGetErrorString(err) << "\n";
	}
	CUDA_SAFE_CALL(cudaMemcpy(C, devPtrC, N * N * sizeof(REAL), cudaMemcpyDeviceToHost));
}

void TestD::stagedLoad(const REAL * __restrict__ A, const REAL * __restrict__ B, REAL * __restrict__ C) const {
	//does not work for matrix matrix multiplication:(
}


void TestD::zeroCopy(const REAL *A, const REAL *B, REAL *C) const {
	REAL *devPtrA, *devPtrB;
	//REAL *pinnedC;
	REAL *dMapC;
	
	CUDA_SAFE_CALL(cudaMalloc((void **)&devPtrA, N * N * sizeof(REAL)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&devPtrB, N * N * sizeof(REAL)));
	//CUDA_SAFE_CALL(cudaHostAlloc((void **)&pinnedC, N * N * sizeof(REAL), cudaHostAllocMapped));
	
	CUDA_SAFE_CALL(cudaMemcpy(devPtrA, A, N * N * sizeof(REAL), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(devPtrB, B, N * N * sizeof(REAL), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaHostGetDevicePointer(&dMapC, C, 0));
	
	dim3 block(TILE_SIZE, TILE_SIZE);
	dim3 grid(N / block.x, N / block.y);
	//naiveMatMatMult<<<grid, block>>>(devPtrA, devPtrB, dMapC);
	sharedMemMatMatMult<<<grid, block>>>(devPtrA, devPtrB, dMapC);
	
	cudaDeviceSynchronize();
	cudaError_t err;
	if((err = cudaGetLastError()) != cudaSuccess){
		std::cout << "Error 0x0: " << cudaGetErrorString(err) << "\n";
	}
	
	CUDA_SAFE_CALL(cudaFree(devPtrA));
	CUDA_SAFE_CALL(cudaFree(devPtrB));
	
	//memcpy(C, pinnedC, N * N * sizeof(REAL));
}

__global__ void tranMat(const REAL * A,  REAL * C, int n){
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	C[x * n + y] = A[y * n + x];
}

void TestD::transposeMat(const REAL * A, REAL * C) const {
	REAL *devPtrA, *devPtrC;
	CUDA_SAFE_CALL(cudaMalloc((void **)&devPtrA, N * N * sizeof(REAL)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&devPtrC, N * N * sizeof(REAL)));
	CUDA_SAFE_CALL(cudaMemcpy(devPtrA, A, N * N * sizeof(REAL), cudaMemcpyHostToDevice));
	dim3 block(TILE_SIZE, TILE_SIZE);
	dim3 grid(N / block.x, N / block.y);
	tranMat<<<grid, block>>>(devPtrA,devPtrC,N);
	CUDA_SAFE_CALL(cudaMemcpy(C, devPtrC, N * N * sizeof(REAL), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(devPtrA));
	CUDA_SAFE_CALL(cudaFree(devPtrC));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

using clock_value_t = long long;

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
	sleep(150000);
	C[y * N + x] = A[y * N + x] +1;
}

void TestD::stagedTransposeMat(const REAL *A, REAL * C) const {
	int nStreams = 4;
	cudaStream_t stream[4];
	CUDA_SAFE_CALL(cudaStreamCreate(&stream[0]));
	CUDA_SAFE_CALL(cudaStreamCreate(&stream[1]));
	CUDA_SAFE_CALL(cudaStreamCreate(&stream[2]));
	CUDA_SAFE_CALL(cudaStreamCreate(&stream[3]));
	REAL *devPtrA, *devPtrC;
	CUDA_SAFE_CALL(cudaMalloc((void **)&devPtrA, N * N * sizeof(REAL)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&devPtrC, N * N * sizeof(REAL)));
	int size=N*N*sizeof(REAL)/nStreams;
	for (int i=0; i<nStreams; i++) {
		int offset = i*N*N/nStreams;
		//std::cout<< "offset: " << offset << " " << *(A + offset) << " size: " << size <<  "\n";
		cudaMemcpyAsync(devPtrA+offset, A+offset, size, cudaMemcpyHostToDevice, stream[i]);
		dim3 block(16,16);
		dim3 grid(N / (block.x), N / (block.y * nStreams));
		testStaged<<<grid, block, 0, stream[i]>>>(devPtrA+offset, devPtrC+offset);
	}
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	CUDA_SAFE_CALL(cudaMemcpy(C, devPtrC, N * N * sizeof(REAL), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(devPtrA));
	CUDA_SAFE_CALL(cudaFree(devPtrC));
}


