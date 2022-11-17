#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>


//warp
#define WARPSIZE 32

extern "C" __global__ void myKernel(int size_x, int size_y, const int* in, int* out){

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int idx = tx + blockIdx.x * blockDim.x;
  int idy = ty + blockIdx.y * blockDim.y;

  //complete to have the thread ID
  int threadID = idx + idy * gridDim.x;

  //complete to have the thread rank inside its warp
  int threadrank_inwarp = tx;
  

  if( (idx<size_x) && (idy<size_y) ){
    out[idx + idy *size_x] = threadrank_inwarp; 
  }

}



int main(int argc, char **argv){

  int i,j;

  int block_size_x = 64;
  int block_size_y = 4;

  if(argc == 3){
    int tmpx = atoi(argv[1]);
    int tmpy = atoi(argv[2]);

    if(block_size_x * block_size_y > 1024){
      printf("Error block_size_x * block_size_y must be lower or equal to 1024\n");
      return -1;
    }   
    block_size_x = tmpx;
    block_size_y = tmpy;
  }

  printf("BLOCK SIZE X: %d\n", block_size_x);
  printf("BLOCK SIZE Y: %d\n", block_size_y);
  printf("\n");


  int size_x = 256;
  int size_y = 32;
  int size_all = size_x * size_y;

  int *in   = (int *) malloc(size_all * sizeof(int));
  int *out  = (int *) malloc(size_all * sizeof(int));

  for(i=0; i<size_all; i++){
    in  [i] = i;
    out [i] = 0;
  }


  //CUDA PART
  int* d_in;
  int* d_out;

  cudaMalloc((void **)&d_in  ,size_all*sizeof(int));
  cudaMalloc((void **)&d_out ,size_all*sizeof(int));

  cudaMemcpy(d_in , in , size_all*sizeof(int), cudaMemcpyHostToDevice);

  dim3 dimBlock;
  dim3 dimGrid;

  dimBlock.x = block_size_x;
  dimBlock.y = block_size_y;

  dimGrid.x = (size_x + dimBlock.x - 1) / dimBlock.x;
  dimGrid.y = (size_y + dimBlock.y - 1) / dimBlock.y;

  myKernel    <<<dimGrid, dimBlock>>>(size_x, size_y, d_in, d_out);
  
  cudaMemcpy(out , d_out , size_all*sizeof(int), cudaMemcpyDeviceToHost);

  //print the first 32 columns
  for(j=0; j<size_y; j++){
    for(i=0; i<32/*size_x*/; i++){
      printf("%2d ",out[i + j * size_x]);
    }
    printf("\n");
  }
  

  cudaFree(d_in  );
  cudaFree(d_out );

  free(in  );
  free(out );


  return 0;

}
