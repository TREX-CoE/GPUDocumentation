#include <stdlib.h>
#include <stdio.h>


//Complete the kernel
__global__ void myKernel(int size, int stride, float *d_in, float *d_out){

  int tid = ...
  if(tid < size){
    //1D acces with stride
    d_out[...] = d_in[...];
  }

}



int main(int argc, char **argv){

  int stride = 1;

  if(argc == 2){
    int tmp_stride = atoi(argv[1]);

    if(tmp_stride < 1 || tmp_stride > 32){
      printf("Error stride must be > 0 and < 33\n");
      return -1;
    }
    stride = tmp_stride;
  }


  int size;
  // 100 000 256
  size = 100000256;

  //allocation and initialization on the host
  float *h_in, *h_out;

  h_in  = (float *) malloc( size * sizeof(float));
  h_out = (float *) malloc( size * sizeof(float));


  int i;
  for(i=0;i<size;i++){
    h_in[i]  = i;
    h_out[i] = 0;
  }


  dim3 threadsPerBlock;
  dim3 numBlocks;
  threadsPerBlock.x = 256;
  numBlocks.x = ((size/32) + threadsPerBlock.x -1) / threadsPerBlock.x;


  //allocation on the device
  float *d_in, *d_out;
  cudaMalloc( (void**) &d_in , size * sizeof(float));
  cudaMalloc( (void**) &d_out, size * sizeof(float));


  //inialization on the device (copy of the host value)
  cudaMemcpy(...);
  cudaMemcpy(...);

  //create two events for timing
  cudaEvent_t start, stop;
  ...
  ...
 
  //first time measure
  ...

  //launch mykernel
  ...

  //second time measure
  ...
  
  //synchronize and compute the elasped time between the two events
  ...
  float elapsedTime;
  ...

  printf("stride: %d  time: %lf\n",stride, elapsedTime);


  //copy the value of d_i in h_i
  cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);


  //print a result
  //printf("value: %lf\n", h_out[50]);


  //free device and host memory
  cudaFree(d_in );
  cudaFree(d_out);
  free(h_in);
  free(h_out);


  return 0;
}
