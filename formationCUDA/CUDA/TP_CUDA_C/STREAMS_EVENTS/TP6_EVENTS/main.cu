#include <stdlib.h>
#include <stdio.h>


#define BLOCKSIZE 256


__global__ void myKernel_a(int nbchunk, int * __restrict__ d_a){

	int nbthreads = blockDim.x * gridDim.x;

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for(int i = 0; i < nbchunk; i++){

		double val = cos( (double)  d_a[index + nbthreads * i] );

		d_a[index + nbthreads * i] = index + nbthreads * i + val;
	}

}


__global__ void myKernel_b(int nbchunk, int * __restrict__ d_b){

	int nbthreads = blockDim.x * gridDim.x;

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for(int i = 0; i < nbchunk; i++)
		d_b[index + nbthreads * i] += 1;

}


__global__ void myKernel_c(int size, int * __restrict__ d_a, int * __restrict__ d_b, int * __restrict__ d_c){

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	d_c[index] = d_b[index] + d_a[index];

}



int main(int argc, char **argv){


	//current device
	int device;
	cudaGetDevice(&device);


	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	//total of global memory of the current device (in Bytes)
	size_t totalGlobalMem = prop.totalGlobalMem;
	//number of multiprocessors of the current device
	int nbMultipro = prop.multiProcessorCount;
	printf("number of multipro: %d\n", nbMultipro);

	//we will fix the number of threads to have only half multiprocessors 
	//working for mykernel_a and the other half for mykernel_b
	int nb_threads = (nbMultipro/2) * BLOCKSIZE;
	size_t size = ( (totalGlobalMem / 40) / nb_threads) * nb_threads; 
	int nbchunk = size / nb_threads;

	printf("nb_threads: %d    size: %d    nbchunk:%d\n",nb_threads,size,nbchunk);

	//allocation and initialization on the host
	int *h_a = (int *) malloc(size * sizeof(int));
	int *h_b = (int *) malloc(size * sizeof(int));
	int *h_c = (int *) malloc(size * sizeof(int));

	for(int i=0; i<size; i++){
		h_a[i] = 0;
		h_b[i] = 0;
		h_c[i] = 0;
	}


	//allocation on the device
	int *d_a;
	int *d_b;
	int *d_c;
	cudaMalloc( (void**) &d_a, size * sizeof(int));
	cudaMalloc( (void**) &d_b, size * sizeof(int));
	cudaMalloc( (void**) &d_c, size * sizeof(int));


	//inialization on the device (copy of the host value)
	cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, h_c, size * sizeof(int), cudaMemcpyHostToDevice);


	dim3 threadsPerBlock;
	dim3 nbBlocks; 
	threadsPerBlock.x = BLOCKSIZE;
	nbBlocks.x = (nbMultipro/2);



	//call myKernel_a in a stream
	//call myKernel_b then myKernel_c in an other stream (than the one used for myKernel_a)
	//ensure than myKernel_c start only after myKernel_a complete using event
	
	cudaStream_t stream[2];
	cudaStreamCreate(&stream[0]);
	cudaStreamCreate(&stream[1]);

	cudaEvent_t event;
	cudaEventCreateWithFlags(&event, cudaEventDisableTiming);	

	myKernel_a<<< nbBlocks, threadsPerBlock, 0, stream[0] >>>(nbchunk, d_a);

	myKernel_b<<< nbBlocks, threadsPerBlock, 0, stream[1] >>>(nbchunk, d_b);

	cudaEventRecord(event, stream[0]);
	cudaStreamWaitEvent(stream[1], event, 0);

	dim3 threadsPerBlock2;
	dim3 nbBlocks2;
	threadsPerBlock2.x = BLOCKSIZE;
	nbBlocks2.x = size / threadsPerBlock.x;

	myKernel_c<<< nbBlocks2, threadsPerBlock2 >>>(size, d_a, d_b, d_c);


	//copy the value of d_i in h_i
	cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);


	//print the result
	if(h_c[0] == 2 && h_c[257] == 259 && h_c[561]== 563){
		printf("SUCCESS\n");
	}else{
		printf("ERROR: h_c[  0] = %d expected    2\n",h_c[  0]);
		printf("       h_c[257] = %d expected  259\n",h_c[257]);
		printf("ERROR: h_c[561] = %d expected  563\n",h_c[561]);
	}


	//free device and host memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(h_a);
	free(h_b);
	free(h_c);

	return 0;
}
