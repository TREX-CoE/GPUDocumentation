#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>



double wallclock(){
  struct timeval timer;
  gettimeofday(&timer, NULL);
  double time = timer.tv_sec + timer.tv_usec * 1.0E-6;
  return time;
}





__global__ void myKernel(int size, int *d_i){


  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid < size){
    for(int loop=0; loop<10000; loop++)
      d_i[tid] += tid + loop;
  }
}

int main(int argc, char **argv){


  int i,size;
  size = 100000000;

  //allocation and initialization on the host
  int *h_array1;
  int *h_array2;

  cudaMallocHost( (void **) &h_array1, size * sizeof(int));
  cudaMallocHost( (void **) &h_array2, size * sizeof(int));


  for(i=0;i<size;i++){
    h_array1[i] = 0;
    h_array2[i] = 0;
  }


  cudaStream_t mystream;
  //stream creation
  cudaStreamCreateWithFlags(&mystream, cudaStreamNonBlocking);


  dim3 threadsPerBlock;
  dim3 numBlocks;
  threadsPerBlock.x = 256;
  numBlocks.x = (size + threadsPerBlock.x -1) / threadsPerBlock.x;


  //allocation on the device
  int *d_array1;
  int *d_array2;
  cudaMalloc( (void**) &d_array1, size * sizeof(int));
  cudaMalloc( (void**) &d_array2, size * sizeof(int));


  double t0 = wallclock();

  cudaMemcpyAsync(d_array1, h_array1, size * sizeof(int), cudaMemcpyHostToDevice, 0);
  myKernel<<< numBlocks, threadsPerBlock, 0, 0 >>>(size, d_array1);
  cudaMemcpyAsync(h_array1, d_array1, size * sizeof(int), cudaMemcpyDeviceToHost, 0);

  double t1  = wallclock();

  cudaMemcpyAsync(d_array2, h_array2, size * sizeof(int), cudaMemcpyHostToDevice, mystream);
  myKernel<<< numBlocks, threadsPerBlock, 0, mystream >>>(size, d_array2);
  cudaMemcpyAsync(h_array2, d_array2, size * sizeof(int), cudaMemcpyDeviceToHost, mystream);

  double t2  = wallclock();

  cudaDeviceSynchronize();
  //stream destruction
  cudaStreamDestroy(mystream);

  double t3  = wallclock();



  //print the result
  printf("value: %d     t1-t0: %lf    t2-t1: %lf  t3-t2: %lf    t3-t0: %lf  \n", h_array1[50], t1-t0,t2-t1,t3-t2,t3-t0);


  //free device and host memory
  cudaFree(d_array1);
  cudaFree(d_array2);
  cudaFreeHost(h_array1);
  cudaFreeHost(h_array2);

  return 0;
}
