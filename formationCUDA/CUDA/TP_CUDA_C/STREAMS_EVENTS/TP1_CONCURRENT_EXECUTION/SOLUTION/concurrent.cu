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
  int *h_i, *h_c;

  h_i = (int *) malloc( size * sizeof(int));
  h_c = (int *) malloc(  100 * sizeof(int));

  for(i=0;i<size;i++){
	h_i[i] = 0;
  }
  for(i=0;i<100;i++){
        h_c[i] = 0;
  }



  dim3 threadsPerBlock;
  dim3 numBlocks;
  threadsPerBlock.x = 256;
  numBlocks.x = (size + threadsPerBlock.x -1) / threadsPerBlock.x;


  //allocation on the device
  int *d_a, *d_b, *d_c, *d_i;
  cudaMalloc( (void**) &d_a, size * sizeof(int));
  cudaMalloc( (void**) &d_b, size * sizeof(int));
  cudaMalloc( (void**) &d_c, 1000 * sizeof(int));
  cudaMalloc( (void**) &d_i, size * sizeof(int));

  double t0 = wallclock();

  //copy h_i in d_i
  cudaMemcpy(d_i, h_i, size * sizeof(int), cudaMemcpyHostToDevice);

  double t1 = wallclock();

  //copy h_c in d_c
  cudaMemcpy(d_c, h_c, 1000 * sizeof(int), cudaMemcpyHostToDevice);


  double t2 = wallclock();

  //copy d_b in d_a
  cudaMemcpy(d_a, d_b, size*sizeof(int), cudaMemcpyDeviceToDevice);

  double t3 = wallclock();

  //call mykernel using numBlocks and threadsPerBlock for the grid and size and d_i as parameters
  myKernel<<< numBlocks, threadsPerBlock >>>(size, d_i);

  double t4  = wallclock();

  //copy d_i in h_i
  cudaMemcpy(h_i, d_i, size * sizeof(int), cudaMemcpyDeviceToHost);

  double t5  = wallclock();

  //print the result
  printf("value: %d     h_i->d_i: %lf    h_c->d_c: %lf   d_b->d_a: %lf    myKernel: %lf  d_i->h_i: %lf    t4-t1:%lf\n", h_i[50], t1-t0,t2-t1,t3-t2,t4-t3,t5-t4, t4-t1);


  //free device and host memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_i);
  free(h_c);
  free(h_i);


  return 0;
}
