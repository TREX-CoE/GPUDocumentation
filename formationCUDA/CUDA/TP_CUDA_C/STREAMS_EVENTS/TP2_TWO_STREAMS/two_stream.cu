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

  //pinned memory allocation for h_array1 and h_array2
  ...
  ...


  for(i=0;i<size;i++){
    h_array1[i] = 0;
    h_array2[i] = 0;
  }


  cudaStream_t stream[2];
  //stream creation
  cudaStreamCreate(&stream[0]);
  cudaStreamCreate(&stream[1]);
  


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

  //do an async copy of h_array1 in d_array1 in the stream[0]
  cudaMemcpyAsync(
  //call myKernel with the stream[0]
  myKernel<<< numBlocks, threadsPerBlock ...>>>(size, d_array1);
  //do an async copy of d_array1 in h_array1 in the stream[0]
  ...

  double t1  = wallclock();

  //do an async copy of h_array2 in d_array2 in the stream[1]
  ...
  //call myKernel with the stream[1]
  myKernel<<< numBlocks, threadsPerBlock ...>>>(size, d_array2);
  //do an async copy of d_array2 in h_array2 in the stream[1]
  ...

  double t2  = wallclock();


  //Synchronize device before stream destruction
  ...


  //stream destruction
  ...
  ...

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
