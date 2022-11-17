#include <stdlib.h>
#include <stdio.h>



__global__ void myKernel(int size, int *d_i){

  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if( index < size)
    d_i[index] = d_i[index] + 1;

}



int main(int argc, char **argv){

  int size = 1000000;


  //allocation and initialization on the host
  int *h_i=NULL;
  cudaHostAlloc((void **) &h_i, size *  sizeof(int), cudaHostAllocMapped | cudaHostRegisterPortable);
  
  for(int i=0; i<size; i++)
    h_i[i] = i;


  dim3 threadsPerBlock;
  dim3 numBlocks;
  threadsPerBlock.x = 256;
  numBlocks.x = (size + threadsPerBlock.x -1)/ threadsPerBlock.x;

  //get the current device
  int device;
  cudaGetDevice(&device);

  //get its properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  //in value1 put the value of the unifiedAddressing property
  int value1;
  value1 = prop.unifiedAddressing;
  printf("Device property unifiedAddressing value: %d\n", value1);





  //in value2 put the value of the canUseHostPointerForRegisteredMem property
  int value2;
  value2 = prop.canUseHostPointerForRegisteredMem;
  printf("Device property canUseHostPointerForRegisteredMem value: %d\n", value2);


  if(value1 || value2){

    myKernel<<< numBlocks, threadsPerBlock >>>(size, h_i);

  }else{

    //call cudaHostGetDevicePointer to ensure translation    
    int *d_i;
    cudaHostGetDevicePointer((void**)&d_i,(void*) h_i ,0);

    myKernel<<< numBlocks, threadsPerBlock >>>(size, d_i);
  }


  cudaDeviceSynchronize();

  //print the result
  printf("h[1000] = %d   expected 1001\n", h_i[1000]);

  //free host memory
  cudaFreeHost(h_i);

  return 0;
}
