#include "test.h"


/**
 * This can be set to true to let cuBlas handle some data transfer.
 * Should not make much of a difference (cuBlas routines use CUDA Runtime routines)
 */
#define USE_CUBLAS_COPY false

//CUDA_SAVE_CALL (and cuBLAS)
#define CUDA_SAFE_CALL(call) { gpuAssert((call), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false){
   if (code != cudaSuccess){
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
inline void gpuAssert(cublasStatus_t code, const char *file, int line, bool abort=false){
	if(code != CUBLAS_STATUS_SUCCESS){
		std::cout << "something went wrong in cublas" << "\n";
	}
}

void TestC::printCudaStats(){
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	//std::cout << "number of devices: " << nDevices << "\n";
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		
		//std::cout << "Shared memory per block: " << (prop.sharedMemPerBlock / 1024) << " KB\n";
		//std::cout << "canMapHostMemory: " << (prop.canMapHostMemory ? "yes" : "no") << "\n";
	}
}

void TestC::initLibs(REAL *r){
	printCudaStats();
	REAL x[16] = {0, 0, 0, 0, 0, 0, 0, 0, -28, 0, 41, -15, 0, 20, -55.744289, -76.930283};
	REAL y[16] = {0, 0, 0, 0, 0, 0, 0, 0, -54.654705, 0, 23.670898, -4.533447, 0, 79.964569, -65.6026, -4};

	
	REAL expected = .0f, actual = .0f;
    for (int i = 0; i < 16; i++) {
        expected += x[i] * y[i];
    }
	
	handle = new cublasHandle_t();
	REAL *devPtrA, *devPtrB;
	
    CUDA_SAFE_CALL(cudaMalloc((void **)&devPtrA, 16 * sizeof(REAL)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&devPtrB, 16 * sizeof(REAL)));
	
	CUDA_SAFE_CALL(cudaMemcpy(devPtrA, x, 16*sizeof(REAL), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(devPtrB, y, 16*sizeof(REAL), cudaMemcpyHostToDevice));
	
	CUDA_SAFE_CALL(cublasCreate(handle));
	
	CUDA_SAFE_CALL(cublasDdot(*handle, 16, devPtrA, 1, devPtrB, 1, &actual));
	
    (*r) = fabs(expected -  actual);
}

void TestC::execTest(const REAL * __restrict__ A, const REAL * __restrict__ B, REAL * __restrict__ C) const {
	cudaEvent_t timer_start, timer_stop;
	float time;
    REAL *devPtrA, *devPtrB, *devPtrC;
	
	CUDA_SAFE_CALL(cudaEventCreate(&timer_start));
	CUDA_SAFE_CALL(cudaEventCreate(&timer_stop));
	
	CUDA_SAFE_CALL(cudaEventRecord( timer_start, 0 ));
	
    CUDA_SAFE_CALL(cudaMalloc ((void**)&devPtrA, N*N*sizeof(REAL)));
	CUDA_SAFE_CALL(cudaMalloc ((void**)&devPtrB, N*N*sizeof(REAL)));
	CUDA_SAFE_CALL(cudaMalloc ((void**)&devPtrC, N*N*sizeof(REAL)));
	
	if(USE_CUBLAS_COPY){
		CUDA_SAFE_CALL(cublasSetMatrix(N, N, sizeof(REAL), A, N, devPtrA, N));
		CUDA_SAFE_CALL(cublasSetMatrix(N, N, sizeof(REAL), B, N, devPtrB, N));
	}else{
		CUDA_SAFE_CALL(cudaMemcpy(devPtrA, A, N*N*sizeof(REAL), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(devPtrB, B, N*N*sizeof(REAL), cudaMemcpyHostToDevice));
	}
	
	if(handle == nullptr)
		CUDA_SAFE_CALL(cublasCreate(handle));
	
	REAL alpha = 1.0f;
	REAL beta = 0.0f;
	CUDA_SAFE_CALL(cublasDgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha, devPtrA, N, devPtrB, N, &beta, devPtrC, N));
	CUDA_SAFE_CALL(cublasDgeam(*handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, &alpha, devPtrC, N, &beta, devPtrB, N, devPtrB, N));	//transform from c-major to r-major
	
	if(USE_CUBLAS_COPY){//devPtrB contains the transpose
		//CUDA_SAFE_CALL(cublasGetMatrix(N,N,sizeof(REAL), devPtrB, N, C, N));	
		CUDA_SAFE_CALL(cublasGetMatrix(N,N,sizeof(REAL), devPtrB, N, C, N));	
	}else{
		CUDA_SAFE_CALL(cudaMemcpy(C, devPtrB, N * N * sizeof(REAL), cudaMemcpyDeviceToHost));
	}
	
	CUDA_SAFE_CALL(cudaEventRecord( timer_stop, 0 ));
	CUDA_SAFE_CALL(cudaEventSynchronize( timer_stop ));

	CUDA_SAFE_CALL(cudaEventElapsedTime( &time, timer_start, timer_stop ));
	
	//std::cout << "cuda time: " << time << "\n";
	
	cudaFree(devPtrA);
	cudaFree(devPtrB);
	cudaFree(devPtrC);
	
	cudaEventDestroy(timer_start);
	cudaEventDestroy(timer_stop);
	
    cublasDestroy(*handle);
	cudaDeviceSynchronize();
}

