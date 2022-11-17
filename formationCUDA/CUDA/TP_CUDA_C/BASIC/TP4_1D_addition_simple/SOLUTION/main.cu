#include <stdlib.h>
#include <stdio.h>


__global__ void myKernel(int *d_a, int *d_b, int *d_c){

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  d_c[index] = d_b[index] + d_a[index];
  
}



int main(int argc, char **argv){

  int size = 512;

  //allocation and initialization on the host
  int *h_a = (int *) malloc(size * sizeof(int));
  int *h_b = (int *) malloc(size * sizeof(int));
  int *h_c = (int *) malloc(size * sizeof(int));

  for(int i=0; i<size; i++){
    h_a[i] = i;
    h_b[i] = i%4;
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


  dim3 threadsPerBlock;
  dim3 nbBlocks; 

  threadsPerBlock.x = 32;
  nbBlocks.x = size / threadsPerBlock.x;

  myKernel<<< nbBlocks, threadsPerBlock >>>(d_a, d_b, d_c);


  //copy the value of d_i in h_i
  cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);


  //print the result
  int success = 1;
  int first_index_error = -1;

  for(int i=0; i<size; i++){
    if(h_c[i] != (i+i%4)){
      success = 0;
      first_index_error = i;
      break;
    }
  }


  if(success){
    printf("SUCCESS\n");
  }else{
    printf("ERROR: h_c[%d] = %d expected %d\n",first_index_error, h_c[first_index_error], first_index_error + first_index_error%4 );
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
