#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>




extern "C" __global__ void myKernel(int size_x, int size_y, int* in, int* out){

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int idx = tx + blockIdx.x * blockDim.x;
  int idy = ty + blockIdx.y * blockDim.y;

  if( (idx<size_x) && (idy<size_y) ){
     for(int loop = 0; loop < 10; loop++)
       out[idx + idy *size_x] += in[idx + idy *size_x];
  }

}


//declare in as const restrict and ou as restrict
extern "C" __global__ void myKernel_opt(int size_x, int size_y, int* in, int* out){

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int idx = tx + blockIdx.x * blockDim.x;
  int idy = ty + blockIdx.y * blockDim.y;

  if( (idx<size_x) && (idy<size_y) ){
     for(int loop = 0; loop < 10; loop++)
       out[idx + idy *size_x] += in[idx + idy *size_x];
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


  int size_x = 512;
  int size_y = 512;
  int size_all = size_x * size_y;

  int *in   = (int *) malloc(size_all * sizeof(int));
  int *out  = (int *) malloc(size_all * sizeof(int));
  int *out2 = (int *) malloc(size_all * sizeof(int));

  for(i=0; i<size_all; i++){
    in  [i] = i;
    out [i] = 0;
    out2[i] = 0;
  }


  //CUDA PART
  int* d_in;
  int* d_out;
  int* d_out2;

  cudaMalloc((void **)&d_in  ,size_all*sizeof(int));
  cudaMalloc((void **)&d_out ,size_all*sizeof(int));
  cudaMalloc((void **)&d_out2,size_all*sizeof(int));

  cudaMemcpy(d_in , in , size_all*sizeof(int), cudaMemcpyHostToDevice);

  dim3 dimBlock;
  dim3 dimGrid;

  dimBlock.x = block_size_x;
  dimBlock.y = block_size_y;

  dimGrid.x = (size_x + dimBlock.x - 1) / dimBlock.x;
  dimGrid.y = (size_y + dimBlock.y - 1) / dimBlock.y;

  //WARMUP
  myKernel    <<<dimGrid, dimBlock>>>(size_x, size_y, d_in, d_out);
  myKernel_opt<<<dimGrid, dimBlock>>>(size_x, size_y, d_in, d_out2);

  myKernel    <<<dimGrid, dimBlock>>>(size_x, size_y, d_in, d_out);
  myKernel_opt<<<dimGrid, dimBlock>>>(size_x, size_y, d_in, d_out2);

  
  cudaMemcpy(out , d_out , size_all*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(out2, d_out2, size_all*sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_in  );
  cudaFree(d_out );
  cudaFree(d_out2);

  free(in  );
  free(out );
  free(out2);


  return 0;

}
