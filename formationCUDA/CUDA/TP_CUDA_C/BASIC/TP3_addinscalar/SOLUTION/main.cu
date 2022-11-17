#include <stdlib.h>
#include <stdio.h>


__global__ void myKernel(int *d_i){

  *d_i = *d_i + 1;
  
}



int main(int argc, char **argv){


  //allocation and initialization on the host
  int *h_i;
  h_i = (int *) malloc(sizeof(int));
  *h_i = 0;


  //allocation on the device
  int *d_i;
  cudaMalloc( (void**) &d_i, sizeof(int));


  //inialization on the device (copy of the host value)
  cudaMemcpy(d_i, h_i,  sizeof(int), cudaMemcpyHostToDevice);



  dim3 threadsPerBlock;
  dim3 numBlocks; 

  threadsPerBlock.x = 1024;
  numBlocks.x = 1024;

  myKernel<<< numBlocks, threadsPerBlock >>>(d_i);


  //copy the value of d_i in h_i
  cudaMemcpy(h_i, d_i,  sizeof(int), cudaMemcpyDeviceToHost);


  //print the result
  printf("value: %d\n", *h_i);


  //free device and host memory
  cudaFree(d_i);
  free(h_i);


  return 0;
}
